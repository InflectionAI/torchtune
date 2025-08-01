# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
from typing import Callable, Optional, Union, List

import torch
from torch import nn
from torchtune.modules import MultiHeadAttention
from torchtune.modules.attention_utils import _MaskType

from torchtune.utils import deprecated

class TransformerSelfAttentionLayer(nn.Module):
    """
    Transformer layer derived from the Llama2 model. Normalization is applied before the attention **and** FF layer.

    Args:
        attn (MultiHeadAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        sa_norm (Optional[nn.Module]): Normalization to be applied before self-attention.
        mlp_norm (Optional[nn.Module]): Normalization to be applied before the feed-forward layer.
        sa_scale (Optional[nn.Module]): Module to scale self-attention output.
        mlp_scale (Optional[nn.Module]): Module to scale the feed-forward output.
        mask_mod (Optional[Callable[[_MaskType, int, int, int], _MaskType]]): A callable
            taking a _MaskType, bsz, and seq_len, and modifying the mask (e.g. for chunked attention).
    """

    def __init__(
        self,
        attn: MultiHeadAttention,
        mlp: nn.Module,
        *,
        sa_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        sa_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
        mask_mod: Optional[Callable[[_MaskType, int, int, int], _MaskType]] = None,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.mlp = mlp
        self.sa_norm = sa_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()
        self.sa_scale = sa_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()
        self.mask_mod = mask_mod or None

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: int,
        decoder_max_seq_len: int,
    ) -> None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (int): this parameter is ignored in this layer.
            decoder_max_seq_len (int): maximum cache sequence length.
        """
        self.attn.setup_cache(batch_size, dtype, max_seq_len=decoder_max_seq_len)

    def caches_are_setup(self) -> bool:
        """
        Check if the key value caches are setup on ``self.attn``.
        See :func:~torchtune.modules.TransformerDecoder.caches_are_setup`.
        """
        return self.attn.kv_cache is not None

    def caches_are_enabled(self) -> bool:
        """
        Checks if the key value caches on ``self.attn`` are enabled.
        See :func:~torchtune.modules.TransformerDecoder.caches_are_enabled`.
        """
        return self.attn.cache_enabled
    
    def reset_cache(self):
        """Reset the key value caches."""
        self.attn.reset_cache()

    def revert_cache_to_valid_length(self, valid_cache_len):
        self.attn.revert_cache_to_valid_length(valid_cache_len)

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
                and before the softmax. Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.encoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.
            **kwargs (dict): transformer layer inputs not relevant to self attention.

        Returns:
            torch.Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]
        """
        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        h = self.sa_norm(x)
        if self.mask_mod is not None:
            # With TP we need to use a replicated tensor here
            bsz, seq_len, *_ = h.shape
            mask = self.mask_mod(mask=mask, bsz=bsz, seq_len=seq_len)
        attn_out = self.attn(h, h, mask=mask, input_pos=input_pos)
        
        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = self.sa_scale(attn_out) + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + self.mlp_scale(mlp_out)
        return out


class TransformerCrossAttentionLayer(nn.Module):
    """
    Cross attention Transformer layer following the same conventions as the TransformerSelfAttentionLayer.
    Normalization is applied before the attention **and** FF layer.

    Args:
        attn (MultiHeadAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        ca_norm (Optional[nn.Module]): Normalization to be applied before cross-attention.
        mlp_norm (Optional[nn.Module]): Normalization to be applied before the feed-forward layer.
        ca_scale (Optional[nn.Module]): Module to scale cross-attention output.
        mlp_scale (Optional[nn.Module]): Module to scale the feed-forward output.

    Raises:
        AssertionError: if attn.pos_embeddings is set.
    """

    def __init__(
        self,
        attn: MultiHeadAttention,
        mlp: nn.Module,
        *,
        ca_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        ca_scale: Optional[nn.Module] = None,
        mlp_scale: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if attn.pos_embeddings is not None:
            raise AssertionError(
                "Doesn't support positional embeddings for cross attention, \
                because q and k are different sequences."
            )
        self.attn = attn
        self.mlp = mlp
        self.ca_norm = ca_norm or nn.Identity()
        self.mlp_norm = mlp_norm or nn.Identity()
        self.ca_scale = ca_scale or nn.Identity()
        self.mlp_scale = mlp_scale or nn.Identity()

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: int,
        decoder_max_seq_len: int,
    ) -> None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (int): maximum cache sequence length.
            decoder_max_seq_len (int): this parameter is ignored in this layer.
        """
        self.attn.setup_cache(batch_size, dtype, encoder_max_seq_len)

    def caches_are_setup(self) -> bool:
        """
        Check if the key value caches are setup on ``self.attn``.
        See :func:~torchtune.modules.TransformerDecoder.caches_are_setup`.
        """
        return self.attn.kv_cache is not None

    def caches_are_enabled(self) -> bool:
        """
        Checks if the key value caches on ``self.attn`` are enabled.
        See :func:~torchtune.modules.TransformerDecoder.caches_are_enabled`.
        """
        return self.attn.cache_enabled

    def reset_cache(self):
        """Reset the key value caches."""
        self.attn.reset_cache()

    def revert_cache_to_valid_length(self, valid_cache_len):
        self.attn.revert_cache_to_valid_length(valid_cache_len)

    def _skip_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Some tokens in x may not attend to any encoder inputs
        due to the cross attention mask (encoder_mask). This results in
        a full row of the attention matrix being masked out.

        In the example below, the word "the" is masked from every embedding.
        The False value means a token can't attend to an embedding.

        .. code-block:: text

            |emb||emb||emb|
        |The| F    F    F
        |red| T    F    T
        |car| F    T    T

        This results in no inputs into the softmax layer which causes a NaN.
        The skip mask is used to mask the outputs of attention and
        mlp resulting in the token being skipped.

        The above example would result in a skip mask of: [[True], [False], [False]]
        which specifies which tokens to fully mask out.

        """
        # no skip_mask if no masking
        if mask is None:
            return None
        # negate mask and convert to boolean mask
        if mask.dtype == torch.bool:
            mask = ~mask
        else:
            mask = torch.isneginf(mask)
        # True where all elements in a row are True
        mask = torch.all(mask, dim=-1, keepdim=True)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        *,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder. Shape
                [batch_size x token_sequence x embed_dim]
            encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position i,j means token i can attend
                to embedding j in the decoder. Mask has shape [batch_size x token_sequence x embed_sequence].
                Default is None.
            **kwargs (dict): transformer layer inputs not relevant to self attention.

        Returns:
            torch.Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]
        """
        # During decoding, it's possible encoder_input is None because the embeds
        # are already stored in the kv cache.
        empty_cache = not self.caches_are_enabled() or self.attn.kv_cache.size == 0
        # Skip cross attention when no secondary input as it's primary purpose
        # is to attend between x and encoder_input.
        if encoder_input is None and empty_cache:
            return x

        # A mask of tokens (x) with no encoder_input
        skip_mask = self._skip_mask(encoder_mask)
        if encoder_mask is not None:
            # TODO: remove after PyTorch 2.5 is released
            # This unmasks the skipped rows to avoid NaNs in SDPA Softmax backward
            # This doesn't affect the output since outputs are masked out later
            encoder_mask = encoder_mask.masked_fill(skip_mask, True)

        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        # TODO: Add support for sample packing and bring back input_pos
        attn_out = self.attn(self.ca_norm(x), encoder_input, mask=encoder_mask)
        if skip_mask is not None:
            attn_out = attn_out.masked_fill(skip_mask, 0)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        h = self.ca_scale(attn_out) + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))
        if skip_mask is not None:
            mlp_out = mlp_out.masked_fill(skip_mask, 0)

        # Residual connection; shape: [batch_size, seq_length, embed_dim]
        out = h + self.mlp_scale(mlp_out)
        return out

def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Return a list of ``n`` identical layers.

    Args:
        module (nn.Module): module to be cloned
        n (int): number of clones

    Returns:
        nn.ModuleList: list of ``n`` identical layers
    """
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder derived from the Llama2 architecture.

    Args:
        tok_embeddings (nn.Embedding): PyTorch embedding layer, to be used to move
            tokens to an embedding space.
        layers (Union[nn.Module, list[nn.Module], nn.ModuleList]): A single transformer Decoder layer, an
            nn.ModuleList of layers or a list of layers. It is recommended to use an nn.ModuleList.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value. This is used to setup the
            :func:`~torchtune.modules.KVCache`
        head_dim (int): embedding dimension for each head in self-attention. This is used
            to setup the :func:`~torchtune.modules.KVCache`
        norm (nn.Module): Callable that applies normalization to the output of the decoder,
            before final MLP.
        output (Union[nn.Linear, Callable]): Callable that applies a linear transformation to the output of
            the decoder.
        num_layers (Optional[int]): Number of Transformer Decoder layers, only define when
            layers is not a list.
        output_hidden_states (Optional[list[int]]): list of layers (indices) to include in the output

    Raises:
        AssertionError:
            If ``num_layers`` is set and layer is a list, **or**
            ``num_layers`` is not set and layer is an ``nn.Module``.

    Note:
        Arg values are checked for correctness (eg: ``attn_dropout`` belongs to [0,1])
        in the module where they are used. This helps reduces the number of raise
        statements in code and improves readability.
    """

    def __init__(
        self,
        *,
        tok_embeddings: nn.Embedding,
        layers: Union[nn.Module, list[nn.Module], nn.ModuleList],
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        norm: nn.Module,
        output: Union[nn.Linear, Callable],
        num_layers: Optional[int] = None,
        output_hidden_states: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        if isinstance(layers, nn.ModuleList):
            pass
        elif isinstance(layers, list):
            layers = nn.ModuleList(layers)
        else:
            if not isinstance(layers, nn.Module):
                raise AssertionError("num_layers is defined, layers must be a module")
            if num_layers is None:
                raise AssertionError("num_layers is not defined, layers must be a list")
            layers = _get_clones(layers, num_layers)

        self.tok_embeddings = tok_embeddings
        self.layers = layers
        self.norm = norm
        self.output = output
        self.output_hidden_states = output_hidden_states or []
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal_mask = None
        self.num_output_chunks = 0
        self.skip_output_layer = False
        
        # Store the number of layers before activation checkpointing wraps them
        self.num_layers = len(layers) if hasattr(layers, '__len__') else num_layers

        # attributes for KV caches during inference
        self.encoder_max_cache_seq_len = None
        self.decoder_max_cache_seq_len = None

    @deprecated("Please use LinearCrossEntropyLoss instead")
    def set_num_output_chunks(self, num_output_chunks: int) -> None:
        """Used to save memory in combination with :class:`~torchtune.modules.loss.CEWithChunkedOutputLoss`.
        This should be called before the first forward pass, in the recipe."""
        self.num_output_chunks = num_output_chunks

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: Optional[int] = None,
        decoder_max_seq_len: Optional[int] = None,
    ):
        """
        Sets up key-value attention caches for inference. For each layer in ``self.layers``:
            - :class:`~torchtune.modules.TransformerSelfAttentionLayer` will use ``decoder_max_seq_len``.
            - :class:`~torchtune.modules.TransformerCrossAttentionLayer` will use ``encoder_max_seq_len``.
            - :class:`~torchtune.modules.model_fusion.FusionLayer` will use ``decoder_max_seq_len`` and ``encoder_max_seq_len``.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (Optional[int]): maximum encoder cache sequence length.
            decoder_max_seq_len (Optional[int]): maximum decoder cache sequence length.
        """
        has_encoder_layers = any(
            isinstance(m, TransformerCrossAttentionLayer) for m in self.modules()
        )
        has_decoder_layers = any(
            isinstance(m, TransformerSelfAttentionLayer) for m in self.modules()
        )

        if has_encoder_layers:
            if encoder_max_seq_len is not None:
                self.encoder_max_cache_seq_len = encoder_max_seq_len
            else:
                self.encoder_max_cache_seq_len = self.max_seq_len

        if has_decoder_layers:
            if decoder_max_seq_len is not None:
                self.decoder_max_cache_seq_len = decoder_max_seq_len
            else:
                self.decoder_max_cache_seq_len = self.max_seq_len
        for layer in self.layers:
            layer.setup_caches(
                batch_size,
                dtype,
                encoder_max_seq_len=self.encoder_max_cache_seq_len,
                decoder_max_seq_len=self.decoder_max_cache_seq_len,
            )

    def caches_are_setup(self) -> bool:
        """
        Check if the key value caches are setup. This means ``setup_caches`` has been called, and
        the relevant attention modules in the model have created their ``KVCache``.
        """
        return self.layers[0].caches_are_setup()

    def caches_are_enabled(self) -> bool:
        """
        Checks if the key value caches are enabled. Once KV-caches have been setup, the relevant
        attention modules will be "enabled" and all forward passes will update the caches. This behaviour
        can be disabled without altering the state of the KV-caches by "disabling" the KV-caches
        using :func:`torchtune.modules.common_utils.disable_kv_cache`, upon which ``caches_are_enabled`` would return False.
        """
        return self.layers[0].caches_are_enabled()

    def reset_caches(self):
        """
        Resets KV-cache buffers on relevant attention modules to zero, and reset cache positions to zero,
        without deleting or reallocating cache tensors.

        Raises:
            RuntimeError: if KV-caches are not setup. Use :func:`~torchtune.modules.TransformerDecoder.setup_caches` to
                setup caches first.
        """
        if not self.caches_are_enabled():
            raise RuntimeError(
                "Key value caches are not setup. Call model.setup_caches first."
            )

        for layer in self.layers:
            layer.reset_cache()
    
    def revert_cache_to_valid_length(self, valid_cache_len):
        for layer in self.layers:
            layer.revert_cache_to_valid_length(valid_cache_len)

    @deprecated("Please use self.skip_output_layer=True and use a linear loss instead")
    def chunked_output(self, last_hidden_state: torch.Tensor) -> list[torch.Tensor]:
        """
        Apply output projection in chunks. This should be applied in conjunction with
        :class:`~torchtune.modules.loss.CEWithChunkedOutputLoss` as upcasting to fp32 is done there.

        To use this method, you should first call
        :func:`~torchtune.modules.TransformerDecoder.set_num_output_chunks`.

        Args:
            last_hidden_state (torch.Tensor): last hidden state of the decoder, having shape
                [b, seq_len, embed_dim].

        Returns:
            list[torch.Tensor]: List of num_chunks output tensors, each with shape
                [b, seq_len/num_chunks, out_dim], where out_dim is usually the vocab size.
        """
        return [
            self.output(chunk)
            for chunk in last_hidden_state.tensor_split(self.num_output_chunks, dim=1)
        ]

    def _validate_inputs(
        self,
        tokens: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        """
        Validates inputs for ``forward``.
        Args:
            tokens (Optional[torch.Tensor]): input tensor with shape ``[b x s]``
            mask (Optional[torch.Tensor]): Attention mask used for inference and for sequence packing.
            encoder_input (Optional[torch.Tensor]): Encoder input for cross-attention.
            encoder_mask (Optional[torch.Tensor]): Encoder attention mask for cross-embedding attention.
            input_pos (Optional[torch.Tensor]): Input tensor position IDs.
            input_embeds (Optional[torch.Tensor]): Input tensor embeddings (if short-circuiting token embeddings).

        Raises:
            ValueError:
                If neither tokens nor input_embeds are passed **or**
                If seq_len of x is bigger than max_seq_len, **or**
                if the model has caches which have been setup with self-attention layers and ``mask`` is not provided, **or**
                if the model has caches which have been setup with encoder layers and ``encoder_mask`` is not provided, **or**
                if the model has caches which have been setup ``input_pos`` is not provided.
        """

        if tokens is None and input_embeds is None:
            raise ValueError(
                "Either tokens or input_embeds must be provided to the decoder."
            )

        # input tensor of shape [b, s]
        seq_len = tokens.shape[1] if tokens is not None else input_embeds.shape[1]

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        if self.caches_are_enabled():
            if mask is None:
                raise ValueError(
                    "KV-caches for self-attention layers are setup for inference mode, causal masks must be provided!"
                    " Use the `mask` arg to provide a causal mask."
                )

            if encoder_input is not None and encoder_mask is None:
                raise ValueError(
                    "KV-caches for cross-attention/fusion layers are setup for inference mode and you seem to be using"
                    " encoder_input, causal masks must be provided! Use the `encoder_mask` arg to provide a causal mask."
                )

            if input_pos is None:
                raise ValueError(
                    "KV-caches are setup for inference mode, input positions must be provided!"
                )

    def forward(
        self,
        tokens: Optional[torch.Tensor],
        *,
        mask: Optional[_MaskType] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            tokens (Optional[torch.Tensor]): input tensor with shape ``[b x s]``
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
                and before the softmax. This parameter is required during inference if caches have been setup.
                Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.encoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            encoder_input (Optional[torch.Tensor]): Optional input embeds from the encoder. Shape ``[b x s_e x d_e]``
            encoder_mask (Optional[torch.Tensor]):  Boolean tensor defining a relational matrix between
                tokens and encoder embeddings. A True value at position ``i,j`` means token ``i`` can attend
                to embedding ``j`` in the decoder. Mask has shape ``[b x s x s_e]``. Default is None,
                but this is required during inference if the model has been setup with any layers
                which use encoder embeddings and caches have been setup.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape ``[b x s]``.
                During inference, this indicates the position of the current token.
                This parameter is required during inference if caches have been setup. Default is None.
            input_embeds (Optional[torch.Tensor]): Pass these instead of tokens to short-circuit token embeddings
                and skip straight to the transformer layers. Shape ``[b x s x d]``. Default: None

        Returns:
            Union[torch.Tensor, list[torch.Tensor]]: output tensor with shape ``[b x s x v]`` if `self.skip_output_layer=False`
            and ``[b x s x d]`` otherwise, or a list of layer output tensors defined by ``output_hidden_states`` with the
            final output tensor appended to the list.

        Note:
            At the very first step of inference, when the model is provided with a prompt,
            ``input_pos`` should contain the positions of all of the tokens in the prompt.
            For a single-batch prompt, or a batch of prompts with identical lengths, this
            will be ``torch.arange(prompt_length)``. For a batch of varying-length prompts,
            shorter prompts are left-padded and position ids are correspondingly right-shifted,
            thus positional ids should be of shape ``[b, padded_prompt_length]``.
            This is because we will need to retrieve the positional embeddings for each input id.
            In the subsequent steps, if the model has been setup with KV-caches, ``input_pos`` will contain
            the position(s) of the current token(s) ``torch.tensor([padded_prompt_length])``. Otherwise,
            ``input_pos`` will contain all the position ids up to the current token.

        Shape notation:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """
        self._validate_inputs(
            tokens=tokens,
            mask=mask,
            encoder_input=encoder_input,
            encoder_mask=encoder_mask,
            input_pos=input_pos,
            input_embeds=input_embeds,
        )

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens) if input_embeds is None else input_embeds

        hidden = []
        for i, layer in enumerate(self.layers):
            if i in self.output_hidden_states:
                hidden.append(h)
            # shape: [b, s, d]
            h = layer(
                h,
                mask=mask,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                input_pos=input_pos,
            )

        if self.num_layers in self.output_hidden_states:
            hidden.append(h)

        # shape: [b, seq_len, out_dim]
        output = self.unembed(h)

        # Output list if hidden states are requested, otherwise just the output
        # TODO: always output a list to have a consistent output type
        output = output if not hidden else [*hidden, output]
        return output

    def unembed(self, h):
        # shape: [b, s, d]
        h = self.norm(h)
        if self.skip_output_layer:
            output = h
        elif self.num_output_chunks > 0:
            output = self.chunked_output(h)
        else:
            # shape: [b, seq_len, out_dim]
            output = self.output(h).float()

        return output

class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))

class MedusaHeads(nn.Module):
    def __init__(
        self, 
        hidden_size : int, 
        vocab_size : int, 
        medusa_num_layers: int = 3, 
        medusa_num_heads: int = 3):
        """
        This implements the entire MedusaHeads forward pass and returns the stacked_medusa_logits in the shape: [medusa_num_heads, batch_size, vocab_size]
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.medusa_num_layers = medusa_num_layers
        self.medusa_num_heads = medusa_num_heads

        # Note Medusa's original repo names this variable as medusa_head not medusa_heads
        self.medusa_base = nn.ModuleList()
        self.medusa_linear_layers = nn.ModuleList()
        for _ in range(medusa_num_heads):
            res_blocks = nn.Sequential(*([ResBlock(self.hidden_size)] * medusa_num_layers))
            linear_layer = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            self.medusa_base.append(res_blocks)
            self.medusa_linear_layers.append(linear_layer)
        

    def forward(self, base_hidden_state):
        """
        This only returns the stacked_medusa_hidden in shape [num_heads, batch_size, embed_dim]
        """
        hidden_states = base_hidden_state.clone()
        medusa_hidden = []
        # TODO: Consider parallelizing this loop for efficiency?
        for i in range(self.medusa_num_heads):
            medusa_hidden.append(self.medusa_base[i](hidden_states))
        
        # stacked_medusa_hidden = torch.stack(medusa_hidden, dim=0)
        return medusa_hidden

    def get_medusa_output(self, hidden_state, chunked = False, num_output_chunks = None):
        medusa_output = []
        for i in range(self.medusa_num_heads):
            hidden_state_i = hidden_state[i]
            if chunked == True:
                output = [self.medusa_linear_layers[i](chunk).float() for chunk in hidden_state_i.tensor_split(num_output_chunks, dim=1)]
            else:
                output = self.medusa_linear_layers[i](hidden_state_i).float()
            medusa_output.append(output)
        return medusa_output
        
    def __iter__(self):
        """Allow iteration over the heads for backward compatibility."""
        return iter(self.medusa_base)
    
    def __getitem__(self, idx):
        """Allow indexing for backward compatibility."""
        if idx == 0:
            return self.medusa_base
        elif idx == 1:
            return self.medusa_linear_layers
        else:
            raise IndexError(f"MedusaHeads only supports indexing 0 (base layers) and 1 (linear layers), got {idx}")
    
    def __len__(self):
        """Return number of heads."""
        return len(self.medusa_base)

class MedusaTransformerDecoder(TransformerDecoder):
    def __init__(self, *, medusa_num_layers: int = 3, medusa_num_heads: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers

        # Get the output dimension from the main output layer
        if hasattr(self.output, 'out_features'):
            self.vocab_size = self.output.out_features
        elif hasattr(self.output, 'weight'):
            self.vocab_size = self.output.weight.shape[0]
        else:
            raise ValueError("Unable to determine vocab size from output layer params")

        # Get the hidden dimension 
        self.hidden_size = self.tok_embeddings.embedding_dim

        # Initialize the medusa_heads
        self.medusa_heads = MedusaHeads(self.hidden_size, 
                                        self.vocab_size, 
                                        self.medusa_num_layers, 
                                        self.medusa_num_heads)
        #freezing the model
        self._freeze_base_model()

    def _freeze_base_model(self):
        """
        Freeze all parameters in the model except for the medusa prediction heads.
        This ensures only the medusa heads are trained while keeping the base model frozen.
        
        Args:
            model (MedusaTransformerDecoder): The medusa model to freeze
        """
        for params in self.parameters():
            params.requires_grad = False
        for params in self.medusa_heads.parameters():
            params.requires_grad = True

    def unembed(self, h):
        
        """
        Override to add Medusa outputs to base output.
        The output returned by unembed will be ultimately returned by the parent TransformerDecoder as [base_output, medusa_output].
        This is ultimately returned as output = [*hidden, base_output, medusa_output]
        """
        
        # Get base output from parent
        base_output = super().unembed(h)
        # Handle both tensor and list cases for batch_size
        if isinstance(base_output, list):
            batch_size = base_output[0].shape[0]
        else:
            batch_size = base_output.shape[0]
            
        # In medusa's original implementation, the normed_last_hidden_state is passed to the heads.
        h = h.detach()
        h = self.norm(h)
        # Pass base LM hidden state to medusa heads and return the final hidden state before applying the medusa linear layer 
        medusa_hidden_state = self.medusa_heads(h)
        
        # Follow the same embed logic as the base LM output layer
        if self.skip_output_layer:
            medusa_output = medusa_hidden_state
        elif self.num_output_chunks > 0:
            # Calculate the chunked_outputs with shape [num_heads, num_output_chunks, batch_size, seq_len, vocab_dim/num_output_chunks]
            medusa_output = self.medusa_heads.get_medusa_output(medusa_hidden_state, chunked = True, num_output_chunks = self.num_output_chunks)
        else:
            # Calculate the outputs without chunking with shape: [num_heads, batch_size, seq_len, vocab_dim]
            medusa_output = []
            # Perform the linear layer pass for each hidden_state and put them in an array[num_heads, batch_size, seq_len, vocab_dim]
            medusa_output = self.medusa_heads.get_medusa_output(medusa_hidden_state, chunked = False)
            
        # Return combined outputs
        if isinstance(base_output, list) and isinstance(base_output, list):
            return base_output + medusa_output
        else:
            return [base_output] + medusa_output

    def forward(self, tokens: torch.Tensor, *, mask: Optional[_MaskType] = None, encoder_input: Optional[torch.Tensor] = None, encoder_mask: Optional[torch.Tensor] = None, input_pos: Optional[torch.Tensor] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass that returns both main LM head output and Medusa head outputs.

        The combined output is formatted as follows:
        [*hidden, output,num_medusa_heads*medusa_logits]
        """
        # Get the base output from parent
        combined_output = super().forward(tokens, mask=mask, encoder_input=encoder_input, 
                                    encoder_mask=encoder_mask, input_pos=input_pos)
        return combined_output

"""
Implementation strategy: the Medusa heads need the (last_hidden_state,  hidden_dim, vocab_size) from the last transformer layer. The hidden_dim, vocab_size are known during the component_building phase. Only the last_hidden_state needs to be obtained. The last TransformerDecoder block can return this separately but will have to be initialized separately to expose this. The hidden_state will have to be extracted and so on. An alternative implementation would be to add that as the last layer of the TransformerDecoder to create the MedusaTransformerDecoder from which the [base_output, stacked_medusa_logits] can be extracted. This was the original implementation chosen during the hackathon. The second implementation will be tried right now.
"""