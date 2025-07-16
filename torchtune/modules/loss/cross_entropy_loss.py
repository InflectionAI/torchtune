# Here, outputs is assumed to be the entire [*hidden, base_hidden_state, medusa_hidden_states] outputted by MedusaTransformerDecoder

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor
from typing import Optional, Union

from torchtune.modules.loss.loss_types import SFTLoss
from torchtune.utils import get_logger

log = get_logger()

class MedusaCrossEntropyLoss(nn.Module, SFTLoss):
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
        self.medusa_linear_layers = None
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index
        self.medusa_num_heads = None
        self.hidden_size = None

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
        self.medusa_linear_layers = model.medusa_heads.medusa_linear_layers
        self.medusa_num_heads = model.medusa_heads.medusa_num_heads
        self.medusa_loss_weights = [0.8**i for i in range(self.medusa_num_heads)]
        self.hidden_size = model.hidden_size

    def compute_cross_entropy(
        self,
        hidden_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
        head_index : int,
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
        if self.medusa_linear_layers is None:
            raise AttributeError("forward called before update_model")
        logits = self.medusa_linear_layers[head_index](hidden_chunk)  # [num_valid, vocab_size]
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
        # Here, outputs is assumed to be the entire [*hidden, base_hidden_state, medusa_hidden_states] outputted by MedusaTransformerDecoder

        # Total number of non-ignored tokens across the entire batch
        mask = targets != self.ignore_index
        total_elements = mask.sum()
        total_loss = 0.0
        for i in range(self.medusa_num_heads):
            # It's been assumed that the for input sequences [0,n] the target is [1, n+1]
            if 1+i >= targets.shape[1]:
                break
            hidden_state = outputs[1+i][:,:-(1+i)].contiguous()
            medusa_targets = targets[:,i+1:].contiguous()
            # Flatten the tensors since cross-entropy expects 2D hidden_chunks
            hidden_state = hidden_state.view(-1, self.hidden_size)
            medusa_targets = medusa_targets.view(-1)

            # Chunk along sequence dimension
            hidden_chunks = hidden_state.tensor_split(self.num_output_chunks, dim=0)
            target_chunks = medusa_targets.tensor_split(self.num_output_chunks, dim=0)

            # Compute cross-entropy loss for the chunks
            loss_per_head = 0.0
            for idx in range(len(hidden_chunks)):
                print("head, idx: ", i, idx)
                h = hidden_chunks[idx]
                t = target_chunks[idx]
                if target_chunks[idx].shape[0] == 0:
                    device = hidden_chunks[0].device
                    t = torch.zeros_like(target_chunks[0]).to(device)
                    h = torch.zeros_like(hidden_chunks[0]).to(device)

                # if i == 2 and idx == 7:
                #     import pdb; pdb.set_trace()
                loss_per_head += self.compute_cross_entropy(h,t, head_index = i)
            total_loss += loss_per_head * self.medusa_loss_weights[i]
        if total_elements == 0:
            # must return after calling compute_cross_entropy to not hang during data parallel training
            return total_loss
        else:
            return total_loss / total_elements