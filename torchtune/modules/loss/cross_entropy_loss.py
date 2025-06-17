# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor
from typing import Optional, Union

from torchtune.modules.loss.loss_types import SFTLoss
from torchtune.utils import get_logger

log = get_logger()


class LinearCrossEntropyLoss(nn.Module, SFTLoss):
    """Memory efficient Cross-entropy loss that incrementally computes loss for chunks of tokens
    by masking ignored tokens, calculating logits and then applying cross-entropy loss. Combines
    the linear projection with the cross-entropy calculation for further memory savings.

    Linear cross entropy masks out ignored tokens before the projection layer to save memory.
    You therefore need to skip the final projection layer in your model and pass it to the loss instead.
    You can setup the loss with the model and compile it as shown below.

    >>> model = Transformer(...)
    >>> loss = LinearCrossEntropyLoss(...)
    >>> loss.set_model_output(model)
    >>> loss.apply_compile_strategy()
    """

    def __init__(
        self,
        num_output_chunks: int = 8,
        ignore_index: int = -100,
    ):
        super().__init__()
        """
        Args:
            num_output_chunks (int): Number of chunks to split the output tensor into. Default is 8.
            ignore_index (int): Index to ignore in the target tensor. Default is -100.
        """
        self.linear_projection = None
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index

    def apply_compile_strategy(self, *args, **kwargs):
        """Applies compile only to the compute_cross_entropy function.
        If compiling CE + chunking operation together, memory requirement is higher."""
        log.warning("Skipping compile loss, as it is not supported at this time")
        # TODO fix compile and re-enable
        # self.compute_cross_entropy = torch.compile(
        #     self.compute_cross_entropy, *args, **kwargs
        # )
        return self

    def set_model_output(self, model: nn.Module) -> None:
        """Modify model output to match the expected input for the loss function."""
        model.skip_output_layer = True
        self.linear_projection = model.output

    def compute_cross_entropy(
        self,
        hidden_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Computes cross-entropy by masking tokens, calculating logits and then applying cross-entropy loss.

        Args:
            hidden_chunk (torch.Tensor): [batch_size, chunk_size, embed_dim]
            target_chunk (torch.Tensor): [batch_size, chunk_size]

        Returns:
            torch.Tensor: Sum of cross-entropy loss for non-ignored tokens in the chunk

        Raises:
            AttributeError: if called before update_model
        """
        # Select hidden states and targets where mask is True
        mask_chunk = target_chunk != self.ignore_index
        if mask_chunk.sum() == 0:
            # Unmask 1 token to allow loss to sync with all data parallel workers
            mask_chunk[0] = True

        target_chunk = target_chunk[mask_chunk]  # [num_valid]
        if isinstance(hidden_chunk, DTensor):
            # DTensor doesn't support masks so we have to mask locally
            mesh = hidden_chunk.device_mesh
            placements = hidden_chunk.placements
            local_hidden_chunk = hidden_chunk.to_local()[mask_chunk]
            hidden_chunk = DTensor.from_local(
                local_hidden_chunk, mesh, placements
            )  # [num_valid, embed_dim]
        else:
            hidden_chunk = hidden_chunk[mask_chunk]  # [num_valid, embed_dim]

        # [num_valid, embed_dim] @ [embed_dim, vocab_size]
        if self.linear_projection is None:
            raise AttributeError("forward called before update_model")
        logits = self.linear_projection(hidden_chunk)  # [num_valid, vocab_size]
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()

        return F.cross_entropy(
            logits.float(),
            target_chunk,
            reduction="sum",
            ignore_index=self.ignore_index,
        )

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            outputs (torch.Tensor): Hidden state of the model, pre projection. Shape ``[bsz, seq_len, emb_dim]``
            targets (torch.Tensor): Labels for the model. Shape ``[bsz, seq_len]``

        Returns:
            torch.Tensor: loss tensor
        """
        # Total number of non-ignored tokens across the entire batch
        mask = targets != self.ignore_index
        total_elements = mask.sum()

        # Chunk along sequence dimension
        hidden_chunks = outputs.tensor_split(self.num_output_chunks, dim=1)
        target_chunks = targets.tensor_split(self.num_output_chunks, dim=1)

        # Compute cross-entropy loss for the chunks
        total_loss = 0.0
        for idx in range(len(hidden_chunks)):
            total_loss += self.compute_cross_entropy(
                hidden_chunks[idx],
                target_chunks[idx],
            )

        if total_elements == 0:
            # must return after calling compute_cross_entropy to not hang during data parallel training
            return total_loss
        else:
            return total_loss / total_elements

class WeightedCrossEntropyLoss(nn.Module, SFTLoss):
    """
    Weighted Cross-entropy loss for Medusa multi-token prediction heads.
    
    This loss function handles multiple prediction heads from MedusaTransformerDecoder,
    where each head predicts k-th next tokens. The loss applies different weights
    to different prediction heads, typically with decreasing weights for further predictions.
    
    Args:
        num_medusa_heads (int): Number of Medusa prediction heads. Default is 4.
        medusa_weights (list[float], optional): Weights for each Medusa head. If None,
            uses exponentially decreasing weights: [1.0, 0.5, 0.25, 0.125, ...]
        main_head_weight (float): Weight for the main LM head. Default is 1.0.
        ignore_index (int): Index to ignore in the target tensor. Default is -100.
        num_output_chunks (int): Number of chunks to split the output tensor into for memory efficiency. Default is 8.
    """
    
    def __init__(
        self,
        num_medusa_heads: int = 3,
        medusa_weights: Optional[list[float]] = None,
        main_head_weight: float = 0.0,
        ignore_index: int = -100,
        num_output_chunks: int = 8,
    ):
        super().__init__()
        self.num_medusa_heads = num_medusa_heads
        self.main_head_weight = main_head_weight
        self.ignore_index = ignore_index
        self.num_output_chunks = num_output_chunks
        
        # Set default exponentially decreasing weights if not provided
        if medusa_weights is None:
            self.medusa_weights = [1.0 / (2 ** i) for i in range(num_medusa_heads)]
        else:
            if len(medusa_weights) != num_medusa_heads:
                raise ValueError(f"medusa_weights length ({len(medusa_weights)}) must match num_medusa_heads ({num_medusa_heads})")
            self.medusa_weights = medusa_weights
        
        # Initialize model output references (set by set_model_output)
        self.main_output_layer = None
        self.medusa_heads = None
        
        log.info(f"WeightedCrossEntropyLoss initialized with main_head_weight: {main_head_weight}, medusa_weights: {self.medusa_weights}")

    def set_model_output(self, model: nn.Module) -> None:
        """
        Modify model output to match the expected input for the WeightedCrossEntropyLoss function.
        
        For MedusaTransformerDecoder, this sets up the model to return hidden states instead
        of final logits, and stores references to both the main output layer and medusa heads
        for computing losses in the loss function.
        
        Args:
            model (nn.Module): The model, expected to be MedusaTransformerDecoder
            
        Raises:
            ValueError: If the model is not a MedusaTransformerDecoder or has mismatched medusa heads
        """
        from torchtune.modules import MedusaTransformerDecoder
        
        if not isinstance(model, MedusaTransformerDecoder):
            raise ValueError(
                "WeightedCrossEntropyLoss.set_model_output expects a MedusaTransformerDecoder, "
                f"but got {type(model)}"
            )
        
        # Set model to skip output layer and return hidden states
        model.skip_output_layer = True
        
        # Store references to output layers for loss computation
        self.main_output_layer = model.output
        self.medusa_heads = model.medusa_heads
        
        # Sync chunking configuration between model and loss
        if hasattr(model, 'num_output_chunks'):
            model.num_output_chunks = self.num_output_chunks
        
        # Verify we have the expected number of medusa heads
        if len(self.medusa_heads) != self.num_medusa_heads:
            raise ValueError(
                f"Model has {len(self.medusa_heads)} medusa heads, but loss expects {self.num_medusa_heads}"
            )

    def compute_head_logits(
        self,
        hidden_states: torch.Tensor,
        head_idx: int,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """
        Compute logits for a specific head from hidden states.
        
        Args:
            hidden_states (torch.Tensor): Hidden states [batch_size, seq_len, hidden_dim]
            head_idx (int): Head index. 0 for main head, 1+ for medusa heads
            
        Returns:
            Union[torch.Tensor, list[torch.Tensor]]: Logits tensor or list of chunked logits
        """
        if head_idx == 0:
            # Main head
            if self.main_output_layer is None:
                raise AttributeError("set_model_output must be called before computing head logits")
            
            if self.num_output_chunks > 1:
                # Chunked computation
                hidden_chunks = hidden_states.tensor_split(self.num_output_chunks, dim=1)
                logits_chunks = []
                for chunk in hidden_chunks:
                    logits_chunks.append(self.main_output_layer(chunk).float())
                return logits_chunks
            else:
                return self.main_output_layer(hidden_states).float()
        else:
            # Medusa head
            medusa_head_idx = head_idx - 1
            if self.medusa_heads is None or medusa_head_idx >= len(self.medusa_heads):
                raise AttributeError("set_model_output must be called before computing medusa head logits")
            
            medusa_head = self.medusa_heads[medusa_head_idx]
            
            if self.num_output_chunks > 1:
                # Chunked computation
                hidden_chunks = hidden_states.tensor_split(self.num_output_chunks, dim=1)
                logits_chunks = []
                for chunk in hidden_chunks:
                    logits_chunks.append(medusa_head(chunk).float())
                return logits_chunks
            else:
                return medusa_head(hidden_states).float()

    def compute_cross_entropy_chunk(
        self,
        logits_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes cross-entropy loss for a single chunk of logits and targets.
        
        Args:
            logits_chunk (torch.Tensor): [batch_size, chunk_size, vocab_size]
            target_chunk (torch.Tensor): [batch_size, chunk_size]
            
        Returns:
            torch.Tensor: Sum of cross-entropy loss for non-ignored tokens in the chunk
        """
        # Flatten logits and targets
        logits_flat = logits_chunk.view(-1, logits_chunk.size(-1))  # [batch_size * chunk_size, vocab_size]
        target_flat = target_chunk.view(-1)  # [batch_size * chunk_size]
        
        # Compute cross-entropy loss
        return F.cross_entropy(
            logits_flat.float(),
            target_flat,
            reduction="sum",
            ignore_index=self.ignore_index,
        )

    def forward(
        self,
        outputs: Union[torch.Tensor, list[torch.Tensor]],
        targets: Union[torch.Tensor, dict[str, Union[torch.Tensor, list[torch.Tensor]]]],
    ) -> torch.Tensor:
        """
        Computes weighted cross-entropy loss for Medusa multi-token prediction.
        
        Args:
            outputs (Union[torch.Tensor, list[torch.Tensor]]): Model outputs from MedusaTransformerDecoder.
                After set_model_output is called, this will be hidden states [batch_size, seq_len, hidden_dim].
                Before set_model_output, this can be a list of logits tensors (main + medusa heads).
            targets (Union[torch.Tensor, dict[str, Union[torch.Tensor, list[torch.Tensor]]]]): Target labels.
                Can be either:
                - A single tensor [batch_size, seq_len] for backward compatibility
                - A dict with keys "labels" and "medusa_labels" for full medusa training
                
        Returns:
            torch.Tensor: Weighted cross-entropy loss
        """
        # Check if set_model_output has been called (hidden states mode)
        hidden_states_mode = hasattr(self, 'main_output_layer') and self.main_output_layer is not None
        
        if hidden_states_mode:
            # Working with hidden states - compute logits for each head
            if not isinstance(outputs, torch.Tensor):
                raise ValueError("In hidden states mode, outputs must be a single tensor of hidden states")
            
            # Handle targets format
            if isinstance(targets, torch.Tensor):
                # Simple case: single target tensor for all heads
                main_targets = targets
                medusa_targets = [targets] * self.num_medusa_heads
            elif isinstance(targets, dict):
                if "labels" not in targets or "medusa_labels" not in targets:
                    raise ValueError("targets dict must contain 'labels' and 'medusa_labels' keys")
                main_targets = targets["labels"]
                medusa_targets = targets["medusa_labels"]
                if len(medusa_targets) != self.num_medusa_heads:
                    raise ValueError(f"Expected {self.num_medusa_heads} medusa label sets, got {len(medusa_targets)}")
            else:
                raise ValueError("targets must be a tensor or dict")
            
            total_loss = 0.0
            total_elements = 0
            
            # Main head loss
            if self.main_head_weight > 0:
                main_logits = self.compute_head_logits(outputs, 0)
                main_loss, main_elements = self._compute_head_loss(main_logits, main_targets)
                total_loss += self.main_head_weight * main_loss
                total_elements += main_elements
            
            # Medusa heads loss
            for i, medusa_weight in enumerate(self.medusa_weights):
                if medusa_weight > 0:
                    medusa_logits = self.compute_head_logits(outputs, i + 1)
                    medusa_loss, medusa_elements = self._compute_head_loss(medusa_logits, medusa_targets[i])
                    total_loss += medusa_weight * medusa_loss
                    total_elements += medusa_elements
            
            # Average loss by total number of elements
            if total_elements == 0:
                return torch.tensor(0.0, device=outputs.device, requires_grad=True)
            
            return total_loss / total_elements
        
        else:
            # Legacy mode: working with pre-computed logits
            # Handle single tensor output (fallback to standard cross-entropy)
            if isinstance(outputs, torch.Tensor):
                if isinstance(targets, dict) and "labels" in targets:
                    return self._compute_single_head_loss(outputs, targets["labels"])
                elif isinstance(targets, torch.Tensor):
                    return self._compute_single_head_loss(outputs, targets)
                else:
                    raise ValueError("targets must contain 'labels' key for single tensor output")
            
            # Handle multiple outputs (main + medusa heads)
            if not isinstance(outputs, list):
                raise ValueError("outputs must be a torch.Tensor or list of torch.Tensor")
            
            if len(outputs) != (1 + self.num_medusa_heads):
                raise ValueError(f"Expected {1 + self.num_medusa_heads} outputs (1 main + {self.num_medusa_heads} medusa), got {len(outputs)}")
            
            # Handle targets format
            if isinstance(targets, dict):
                if "labels" not in targets or "medusa_labels" not in targets:
                    raise ValueError("targets must contain 'labels' and 'medusa_labels' keys")
                main_targets = targets["labels"]
                medusa_targets_list = targets["medusa_labels"]
            else:
                raise ValueError("With multiple outputs, targets must be a dict with 'labels' and 'medusa_labels' keys")
            
            if len(medusa_targets_list) != self.num_medusa_heads:
                raise ValueError(f"Expected {self.num_medusa_heads} medusa label sets, got {len(medusa_targets_list)}")
            
            total_loss = 0.0
            total_elements = 0
            
            # Main LM head loss
            main_logits = outputs[0]
            
            if isinstance(main_targets, list):
                main_targets = torch.tensor(main_targets, device=self._get_device_from_output(main_logits))
            
            main_loss, main_elements = self._compute_head_loss(main_logits, main_targets)
            total_loss += self.main_head_weight * main_loss
            total_elements += main_elements
            
            # Medusa heads loss
            for i, (medusa_logits, medusa_weight) in enumerate(zip(outputs[1:], self.medusa_weights)):
                medusa_targets = medusa_targets_list[i]
                
                if isinstance(medusa_targets, list):
                    medusa_targets = torch.tensor(medusa_targets, device=self._get_device_from_output(medusa_logits))
                
                medusa_loss, medusa_elements = self._compute_head_loss(medusa_logits, medusa_targets)
                total_loss += medusa_weight * medusa_loss
                total_elements += medusa_elements
            
            # Average loss by total number of elements
            if total_elements == 0:
                return torch.tensor(0.0, device=self._get_device_from_output(outputs[0]), requires_grad=True)
            
            return total_loss / total_elements
    
    def _get_device_from_output(self, output: Union[torch.Tensor, list[torch.Tensor]]) -> torch.device:
        """Get device from output, handling both single tensors and lists of chunks."""
        if isinstance(output, torch.Tensor):
            return output.device
        elif isinstance(output, list) and len(output) > 0:
            return output[0].device
        else:
            raise ValueError("Cannot determine device from empty output")
    
    def _compute_head_loss(self, logits: Union[torch.Tensor, list[torch.Tensor]], targets: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Computes cross-entropy loss for a single head, handling both regular and chunked outputs.
        
        Args:
            logits (Union[torch.Tensor, list[torch.Tensor]]): Either a single tensor [batch_size, seq_len, vocab_size]
                or a list of chunk tensors when using chunked output.
            targets (torch.Tensor): Target labels [batch_size, seq_len]
            
        Returns:
            tuple[torch.Tensor, int]: (total_loss, total_elements)
        """
        # Handle chunked output (list of tensors)
        if isinstance(logits, list):
            return self._compute_chunked_head_loss(logits, targets)
        
        # Handle single tensor output
        return self._compute_chunked_loss(logits, targets)
    
    def _compute_chunked_head_loss(self, logits_chunks: list[torch.Tensor], targets: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Computes cross-entropy loss for chunked output from medusa heads.
        
        Args:
            logits_chunks (list[torch.Tensor]): List of logit chunks, each [batch_size, chunk_size, vocab_size]
            targets (torch.Tensor): Target labels [batch_size, seq_len]
            
        Returns:
            tuple[torch.Tensor, int]: (total_loss, total_elements)
        """
        # Count total non-ignored elements
        mask = targets != self.ignore_index
        total_elements = mask.sum().item()
        
        # Split targets to match logits chunks
        target_chunks = targets.tensor_split(len(logits_chunks), dim=1)
        
        # Ensure we have matching number of chunks
        if len(logits_chunks) != len(target_chunks):
            raise ValueError(f"Mismatch between logits chunks ({len(logits_chunks)}) and target chunks ({len(target_chunks)})")
        
        # Compute loss for each chunk
        total_loss = 0.0
        for logits_chunk, target_chunk in zip(logits_chunks, target_chunks):
            chunk_loss = self.compute_cross_entropy_chunk(logits_chunk, target_chunk)
            total_loss += chunk_loss
        
        return total_loss, total_elements

    def _compute_single_head_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Fallback method for single head loss computation."""
        loss, elements = self._compute_chunked_loss(logits, targets)
        return loss / elements if elements > 0 else torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    def _compute_chunked_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Computes cross-entropy loss using chunking for memory efficiency.
        
        Args:
            logits (torch.Tensor): [batch_size, seq_len, vocab_size]
            targets (torch.Tensor): [batch_size, seq_len]
            
        Returns:
            tuple[torch.Tensor, int]: (total_loss, total_elements)
        """
        # Count total non-ignored elements
        mask = targets != self.ignore_index
        total_elements = mask.sum().item()
        
        # Chunk along sequence dimension
        logits_chunks = logits.tensor_split(self.num_output_chunks, dim=1)
        target_chunks = targets.tensor_split(self.num_output_chunks, dim=1)
        
        # Compute loss for each chunk
        total_loss = 0.0
        for logits_chunk, target_chunk in zip(logits_chunks, target_chunks):
            chunk_loss = self.compute_cross_entropy_chunk(logits_chunk, target_chunk)
            total_loss += chunk_loss
        
        return total_loss, total_elements