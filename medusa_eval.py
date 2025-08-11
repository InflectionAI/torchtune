import torch
import torchtune
from torchtune.models.llama3_1 import llama3_1_8b_medusa
from accelerate import init_empty_weights
import os
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import json
from transformers import LlamaTokenizer, AutoTokenizer
from functools import partial
from torchtune.models.llama3._tokenizer import Llama3Tokenizer
from torchtune.datasets._chat import chat_dataset
import gc
from torchtune.modules.common_utils import disable_kv_cache
from torchtune.modules import TransformerSelfAttentionLayer
import torch.nn.functional as F
import logging
import numpy as np
import random
import torch.nn as nn
from contextlib import nullcontext
# Set up logging globally here
logging.basicConfig(
    level=logging.DEBUG,  # or INFO in production
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

import sys, pdb, traceback; sys.excepthook = lambda t, v, tb: (traceback.print_exception(t, v, tb), pdb.post_mortem(tb))

# Force SDPA math kernel to reduce numeric drift across paths
def sdp_math_context():
    try:
        from torch.nn.attention import sdpa_kernel
        return sdpa_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    except Exception:
        return nullcontext()

global tokenizer
def load_data(dataset_dir, tokenizer_dir, bs = 1):
    global tokenizer
    tokenizer = Llama3Tokenizer(tokenizer_dir)

    dataset = chat_dataset(
        tokenizer=tokenizer,
        source="json",
        data_files=dataset_dir,
        conversation_column="messages",
        conversation_style="openai",
        split = 'train[80%:90%]'
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        collate_fn=torchtune.data.padded_collate_sft
    )
    return dataloader
    
def load_model(checkpoint_dir):
    if checkpoint_dir == None:
        return
    with init_empty_weights():
        model = llama3_1_8b_medusa()
        if debug == True: 
            no_kv_model = llama3_1_8b_medusa()
    checkpoint_dir += 'model-00001-of-00001.bin'
    checkpoint = torch.load(checkpoint_dir, map_location = device)

    model.load_state_dict(checkpoint, assign = True)
    model = model.to(device)
    model.eval()
    # Use the same dtype as the model parameters
    model_dtype = next(model.parameters()).dtype
    # Set up caches on the same device as the model
    with device:
        model.setup_caches(batch_size=1, dtype=model_dtype, decoder_max_seq_len=max_cache_size)
    if debug == True: 
        no_kv_model.load_state_dict(checkpoint, assign = True)
        no_kv_model = no_kv_model.to(device)
        no_kv_model.eval()
    torch.set_grad_enabled(False)
    # Assuming model.output is your final projection layer (nn.Linear)
    # model.output.weight = nn.Parameter(model.output.weight.float())
    # if model.output.bias is not None:
    #     model.output.bias = nn.Parameter(model.output.bias.float())
    # no_kv_model.output.weight = nn.Parameter(no_kv_model.output.weight.float())
    # if no_kv_model.output.bias is not None:
    #     no_kv_model.output.bias = nn.Parameter(no_kv_model.output.bias.float())
    # model = model.float()
    # no_kv_model = no_kv_model.float()
    # with torch.no_grad():
    #     convert_module_to_float32(model)
    #     convert_module_to_float32(no_kv_model)
    # model_cpu = model.cpu()
    # model_cpu = model_cpu.float()
    # model = model_cpu.to(device)
    # torch.cuda.empty_cache()
    # no_kv_model_cpu = no_kv_model.cpu()
    # torch.cuda.empty_cache()
    # no_kv_model_cpu = no_kv_model_cpu.float()
    # no_kv_model = no_kv_model_cpu.to(device)
    # torch.cuda.empty_cache()
    if debug == True:
        return model, no_kv_model
    
    return model

def convert_module_to_float32(module):
    for param in module.parameters(recurse=False):
        param.data = param.data.float()
        if param._grad is not None:
            param._grad.data = param._grad.data.float()

    for buffer_name, buffer in module.named_buffers(recurse=False):
        module.register_buffer(buffer_name, buffer.float())

    for child in module.children():
        convert_module_to_float32(child)
    torch.cuda.empty_cache()

def format_input(input_tokens):
    assistant_token = 78191
    assistant_append_tokens =  torch.tensor([128007], device = device)
    tensor_2d = input_tokens  # shape: [1, seq_len]
    tensor_1d = tensor_2d[0]  # shape: [seq_len]
    for i in range(len(tensor_1d)):
        if int(tensor_1d[i]) == assistant_token:
            break
    part1 = torch.cat((tensor_1d[:i+1], assistant_append_tokens))
    # Optional: keep them 2D
    part1 = part1.unsqueeze(0)
    return part1

def decode(x, skip_special_tokens = False):
    return tokenizer.decode(x.flatten().tolist(), skip_special_tokens = skip_special_tokens)

def create_causal_mask2(
    batch_size: int,
    current_seq_len: int,  # This is (1+n) where n is number of medusa heads
    cached_seq_len: int,   # Current KV cache length
    max_cache_size,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Create a causal mask for Medusa evaluation with KV cache.
    
    Args:
        batch_size: Number of sequences in batch (typically 1 for Medusa)
        current_seq_len: Length of new tokens being processed (1 + medusa_heads)
        cached_seq_len: Length of tokens already in KV cache
        device: Device to create tensor on
        dtype: Data type for the mask
    
    Returns:
        Causal mask of shape [batch_size, current_seq_len, total_seq_len]
    """
    total_seq_len = cached_seq_len + current_seq_len
    
    # Create a lower triangular mask for the full sequence
    mask = torch.tril(
        torch.ones(
            current_seq_len, 
            total_seq_len, 
            device=device, 
            dtype=dtype
        )
    )
    suffix_mask_dim = max_cache_size - mask.shape[-1]
    
    mask_suffix = torch.zeros((current_seq_len , suffix_mask_dim), device = device, dtype = dtype)
    full_mask = torch.cat((mask, mask_suffix), dim = -1) 
    # Expand to batch dimension
    full_mask = full_mask.unsqueeze(0)  # [1, current_seq_len, total_seq_len]
    # full_mask = full_mask.expand(batch_size, current_seq_len, max_cache_size)
    return full_mask


def evaluate(dataloader, model, tokens_to_generate = 100):
    predictions = []
    # initialize kv cache
    for batch in dataloader:
        model.reset_caches()
        model_dtype = next(model.parameters()).dtype
        # empty kv cache
        input_tokens = batch['tokens'].to(device)
        input_prompt = format_input(input_tokens) # bs, seq
        # DEBUG
        input_prompt = input_prompt#[:, :4] 

        print('input_prompt:', decode(input_prompt))

        bs = input_prompt.shape[0]; curr_seq_len = input_prompt.shape[1] 
        curr_kv_len = 0

        # Use full decoder cache size for mask (matches KV cache setup)
        causal_mask = create_causal_mask(bs, curr_seq_len, 0, max_cache_size, device, model_dtype)
        input_pos = torch.arange(curr_seq_len, device = device).unsqueeze(0)
        print("model input: ", input_prompt)

        with sdp_math_context():
            output = model(input_prompt, mask = causal_mask, input_pos = input_pos) # shape: [(1+n), bs, seq, vocab_dim]

        base_logits = output[0][:, -1] # shape: [bs, vocab_dim]
        pred = base_logits.argmax(dim = -1) # shape: [bs, 1]

        medusa_logits = torch.stack(output[1:])[:, :, -1] # shape: [n, bs, vocab_dim]
        medusa_out = medusa_logits.argmax(dim = -1) # shape: [n, bs]
        medusa_out = medusa_out.permute((1,0)) # shape: [bs, n]
        tokens_generated = 1
        preds = torch.cat((pred.unsqueeze(-1), medusa_out), dim = -1) # shape: [bs, 1+n]
        accept_lengths = []
        pass_idx = 0
        curr_kv_len = curr_seq_len

        while(tokens_generated<tokens_to_generate):
            pass_idx += 1
            #now take all of the previous outputs and put them into the model as a batch
            curr_seq_len = preds.shape[1] 
            # Use full decoder cache size for mask (matches KV cache setup)
            causal_mask = create_causal_mask(bs, curr_seq_len, curr_kv_len, max_cache_size, device, model_dtype)
            # shape: [bs, curr_seq, self.encoder_max_cache_seq_len], boolean mask with True representing queries to attend
            input_pos = torch.arange(curr_kv_len, curr_kv_len + curr_seq_len, device=device).unsqueeze(0)
            # All True rect mask of new_tokens x tokens_generated | upper_triangular mask of new_tokens x new_tokens + False rect mask of new_tokens x (encoder_max_cache_seq_len - (tokens_generated + new_tokens))
            with sdp_math_context():
                pred = model(preds, mask = causal_mask, input_pos = input_pos) # shape: [(1+n), bs, (1+n), vocab_dim]

            
            base_logits = pred[0] # shape: [bs, (1+n), vocab_dim]
            medusa_logits = torch.stack(pred[1:]) # shape: [n, bs, (1+n), vocab_dim]
            base_out = base_logits.argmax(dim = -1) # shape: [bs, (1+n)]
            medusa_out = medusa_logits.argmax(dim = -1) # shape: [n, bs, (1+n)]
            
            # compare base_out with preds to see which medusa_heads in the prev inference were correct:
            mask = (base_out[:, :-1] == preds[:, 1:])
            correct_pred_mask = mask.cumprod(dim = -1)
            last_accepted_head = correct_pred_mask.sum().item()

            # accept_len denotes the last head that was correct. If the last head was correct then when it is inputted back into the model, the output will also be relevant (with the base_out also being correct). Therefore the base_out is taken as an accepted token and the medusa_out is taken as the input for the next pass.
            curr_kv_len += (last_accepted_head+1)

            # reset kv cache to curr_kv_len
            model.revert_cache_to_valid_length(curr_kv_len)
            tokens_generated += (last_accepted_head+1)
            

            # what should be the input for the next pass? The last medusa pred that was correct. Take it's output as the input for the next pass.
            accepted_head_medusa_pred = medusa_out[:, :, last_accepted_head] # shape: [n, bs]
            accepted_head_medusa_pred = accepted_head_medusa_pred.transpose(0, 1)
            
            preds = torch.cat((base_out[:, last_accepted_head: last_accepted_head + 1], accepted_head_medusa_pred), dim = -1)
            accept_lengths.append((last_accepted_head+1))
            
            # Extract the accepted tokens for decoding
            accepted_tokens = base_out[0, :last_accepted_head+1]  # shape: [last_accepted_head+1]
            decoded_prediction = decode(accepted_tokens) 
            predictions.extend(decoded_prediction)
            # tokenizer.decode(accepted_tokens.flatten().tolist(), skip_special_tokens=False)
            print(f"Prediction {pass_idx}: ", decoded_prediction)

            # preds is the new input for the next pass
            
        print("accept_lengths: ", accept_lengths)
        print("Prediction: ", ''.join(predictions))
        return

def create_causal_mask(
    batch_size: int,
    current_seq_len: int,  # This is (1+n) where n is number of medusa heads
    cached_seq_len: int,   # Current KV cache length
    max_cache_size,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Create a causal mask for Medusa evaluation with KV cache.
    
    Args:
        batch_size: Number of sequences in batch (typically 1 for Medusa)
        current_seq_len: Length of new tokens being processed (1 + medusa_heads)
        cached_seq_len: Length of tokens already in KV cache
        device: Device to create tensor on
        dtype: Data type for the mask
    
    Returns:
        Causal mask of shape [batch_size, current_seq_len, total_seq_len]
    """
    total_seq_len = cached_seq_len + current_seq_len
    old_tokens_mask = torch.ones(current_seq_len, cached_seq_len, device=device, dtype=dtype)
    # Create a lower triangular mask for the full sequence
    new_tokens_triangular_mask = torch.tril(
        torch.ones(
            current_seq_len, 
            current_seq_len, 
            device=device, 
            dtype=dtype
        )
    )
    suffix_mask_dim = max_cache_size - total_seq_len
    mask_suffix = torch.zeros((current_seq_len , suffix_mask_dim), device = device, dtype = dtype)
    # mask = torch.cat((mask, mask_suffix), dim = -1) 
    mask = torch.cat((old_tokens_mask, new_tokens_triangular_mask, mask_suffix), dim = -1)

    # Expand to batch dimension
    mask = mask.unsqueeze(0)  # [1, current_seq_len, total_seq_len]
    # full_mask = full_mask.expand(batch_size, current_seq_len, max_cache_size)
    # print("mask:\n", mask)
    # print("bool mask:\n", mask.bool())
    mask = mask.bool()

    curr_mask = torch.cat((old_tokens_mask, new_tokens_triangular_mask), dim = -1)
    # default_mask = torch.ones(current_seq_len, total_seq_len, dtype=torch.bool, device = device).tril(diagonal=0)
    # print("full_mask:\n", curr_mask, curr_mask.shape)
    # print("temp_mask:\n", default_mask, default_mask.shape)
    # print("Mask Equality:", (curr_mask == default_mask).all().item())
    # assert False
    # return default_mask.unsqueeze(0)
    return mask

def no_kv_evaluate(dataloader, model, batch, no_kv_predictions, tokens_to_generate = 5, accepted_preds = None):
    # predictions = []
    # global no_kv_predictions
    accepted_tokens_list = []
    # breakpoint()
    if accepted_preds is None:

        input_tokens = batch['tokens'].to(device)
        input_prompt = format_input(input_tokens)
        # input_prompt = batch
        print('input_prompt:', decode(input_prompt))
        
        bs = input_prompt.shape[0]
        accepted_preds = input_prompt  # Initialize with full prompt
        # tokens_generated = 0
        # Always pass a boolean causal mask so SDPA takes the same code path
        seq_len = accepted_preds.shape[1]
        mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        ).unsqueeze(0)
        input_pos = torch.arange(seq_len, device=device).unsqueeze(0)
        if model.caches_are_enabled():
            with disable_kv_cache(model):
                with sdp_math_context():
                    output = model(accepted_preds, mask=mask, input_pos=input_pos)  # pred[0] = base logits, shape: [bs, seq, vocab]
        else:
            with sdp_math_context():
                output = model(accepted_preds, mask=mask, input_pos=input_pos)
        base_logits = output[0][:, -1, :]  # last token logits
        # tokens_generated += 1
        # if tokens_generated == tokens_to_generate:
        #     return base_logits
        next_token = base_logits.argmax(dim=-1, keepdim=True)  # [bs, 1]
        accepted_preds = torch.cat([accepted_preds, next_token], dim=-1)
    else:
        # while tokens_generated < tokens_to_generate:
        # torch.cuda.empty_cache() 
        # Always pass a boolean causal mask so SDPA takes the same code path
        bs = accepted_preds.shape[0]
        seq_len = accepted_preds.shape[1]
        mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        ).unsqueeze(0)
        input_pos = torch.arange(seq_len, device=device).unsqueeze(0)
        if model.caches_are_enabled():
            with disable_kv_cache(model):
                with sdp_math_context():
                    output = model(accepted_preds, mask=mask, input_pos=input_pos)  # pred[0] = base logits, shape: [bs, seq, vocab]
        else:
            with sdp_math_context():
                output = model(accepted_preds, mask=mask, input_pos=input_pos)
        base_logits = output[0][:, -1, :]  # last token logits
        # tokens_generated += 1
        # if tokens_generated == tokens_to_generate:
        #     return base_logits
        next_token = base_logits.argmax(dim=-1, keepdim=True)  # [bs, 1]

        # Append base prediction to input
        accepted_preds = torch.cat([accepted_preds, next_token], dim=-1)
        # accepted_tokens_list.append(next_token.item())

        # Decode and store
        decoded = decode(next_token)
        no_kv_predictions.append(decoded)
    return base_logits, next_token, accepted_preds
    print("-----------------------------------------------------")
    print("Prediction: ", ''.join(predictions))
    print("Accepted token IDs:", accepted_tokens_list)
    return accepted_tokens_list

def get_attn_layer(model):
    for m in model.modules():
            if isinstance(m, TransformerSelfAttentionLayer):
                layer = m
                break
            else:
                layer = None
    return layer

def get_model_dtype(model):
    return next(model.parameters()).dtype

def get_medusa_preds(output, first_decode = False):
    bs = output[0].shape[0]
    seq_len = output[0].shape[1]
    vocab_dim = output[0].shape[-1]
    n = len(output[1:])
     
    if first_decode:
        medusa_logits = [m[:, -1:] for m in output[1:]] # shape: list of [bs, 1, vocab_dim]
        medusa_logits = torch.cat(medusa_logits, dim = -2)
    else:
        medusa_logits = output[1:]
        medusa_logits = torch.stack(medusa_logits, dim = -2) # shape: list of [bs, seq_len, n, vocab_dim]
    
    medusa_preds = medusa_logits.argmax(dim = -1) # shape: list of [bs, seq_len, n]
    return medusa_preds, medusa_logits
    
def kv_evaluate(dataloader, model, batch, kv_predictions, tokens_to_generate = 5, input_token = None):
    '''Eval using kv caching'''
    layer = get_attn_layer(model)
    model_dtype = get_model_dtype(model)
    accepted_tokens_list = []
    # breakpoint()
    if input_token is None:
        model.reset_caches()
        input_tokens = batch['tokens'].to(device)
        input_prompt = format_input(input_tokens)
        
        print("input_prompt:", input_prompt)
        print('input_prompt:', decode(input_prompt))
        
        bs = input_prompt.shape[0]
        curr_seq_len = input_prompt.shape[1]
        curr_kv_len = 0

        # Initial forward pass with full prompt
        # Create mask and input_pos for kv caching
        causal_mask = create_causal_mask(bs, curr_seq_len, curr_kv_len, max_cache_size, device, model_dtype)
        input_pos = torch.arange(curr_seq_len, device=device).unsqueeze(0).expand(bs, -1)

        # Perform model forward pass
        with sdp_math_context():
            output = model(input_prompt, mask=causal_mask, input_pos=input_pos)

        base_logits = output[0][:, -1:]  # shape: [bs, vocab_dim]
        vocab_dim = base_logits.shape[-1]
        next_token = base_logits.argmax(dim=-1)  # shape: [bs, 1]
        
        n = len(output[1:]) # num medusa heads
        # get last medusa logits
        medusa_preds, medusa_logits = get_medusa_preds(output, first_decode = True)
    
        # accepted_tokens_list.append(next_token.item())
        decoded_prediction = decode(next_token)
        kv_predictions.append(decoded_prediction)
        
        combined_preds = torch.cat((next_token, medusa_preds), dim = 1) # shape: [bs, 1+n]
        combined_logits = torch.cat((base_logits, medusa_logits), dim = 1)

        # update kv cache
        curr_kv_len = curr_seq_len
        model_kv_len = layer.attn.kv_cache.size
        assert (curr_kv_len == int(model_kv_len))
        # curr_seq_len = 1
        print("-----------------------------------------------------")
        print('cache len now:', layer.attn.kv_cache.size)
        return combined_logits, combined_preds, None
    else:
        # breakpoint()
        bs = input_token.shape[0]
        next_token = input_token
        n = model.medusa_num_heads
        # Verify kv cache size
        curr_kv_len = layer.attn.kv_cache.size
        curr_seq_len = 1 + n # seq contains base_model pred with all medusa preds
        # Create mask and positions for single new token
        causal_mask = create_causal_mask(bs, curr_seq_len, curr_kv_len, max_cache_size, device, model_dtype)
        # breakpoint()
        input_pos = torch.arange(curr_kv_len, curr_kv_len + (1+n), device=device).unsqueeze(0)
        # input_pos = torch.full((bs, 1), curr_kv_len, device=device)

        assert(input_pos.shape == (bs, 1+n))
        
        # Forward pass with just the new token
        with sdp_math_context():
            output = model(next_token, mask=causal_mask, input_pos=input_pos) # shape: list [bs, 1+n, vocab_dim]
        # assert(output[0].shape == (bs, 1+n, vocab_dim))
        
        base_logits = output[0]#[:, -1:]  # shape: [bs, seq_len, vocab_dim] where seq_len = 1+n
        base_tokens = base_logits.argmax(dim=-1)  # shape: [bs, 1+n]
        medusa_preds, medusa_logits = get_medusa_preds(output) # shape:  # shape: list of [bs, seq_len, n] where seq_len = 1+n
        # Verify preds
        
        mask = (next_token[:, 1:] == base_tokens[:, :-1])
        mask = mask.cumprod(dim = 1)
        nz = mask.nonzero(as_tuple=True)[1]
        # breakpoint()
        

        accepted_head_idx = (1 + nz[-1].item()) if nz.numel()>0 else 0
        accepted_head_idx = min(accepted_head_idx, n)
        # accepted_head_idx += 1
        # if accepted_head_idx = 0 then no head is accepted. So take the last_accepted_base, the head_predictions corresponding to that base for the next iteration.
        print("accepted_head_idx:", accepted_head_idx)
        if accepted_head_idx>0:
            print("next_token:", next_token, decode(next_token))
            print("base_tokens:", base_tokens, decode(base_tokens))

        # breakpoint()
        base_token = base_tokens[:, accepted_head_idx].unsqueeze(-1) # shape: [bs, 1]
        medusa_tokens = medusa_preds[:, accepted_head_idx] # shape: [bs, n]
        

        # breakpoint()
        next_tokens = torch.cat((base_token, medusa_tokens), dim = -1)
        # assert(next_tokens.shape[1] == (1 + n))
        next_logits = torch.cat((base_logits[:, accepted_head_idx].unsqueeze(1), medusa_logits[:, accepted_head_idx]), dim = -2)
        
        curr_kv_len += 1
        # reset cache
        print("-----------------------------------------------------")
        print('cache len before:', layer.attn.kv_cache.size)
        curr_len = layer.attn.kv_cache.size
        invalid_len = n - accepted_head_idx
        valid_len = curr_len - invalid_len
        model.revert_cache_to_valid_length(valid_len)
        print('cache len after:', layer.attn.kv_cache.size)

        # accepted_tokens_list.append(next_token.item())
        print("-----------------------------------------------------")
        decoded_prediction = decode(base_token)
        kv_predictions.append(decoded_prediction)
        # print(decoded_prediction)
        # predictions.append(decoded_prediction)
        return next_logits, next_tokens, accepted_head_idx
        
    print("-----------------------------------------------------")
    print("Prediction: ", ''.join(predictions))
    print("-----------------------------------------------------")

    print("Accepted token IDs:", accepted_tokens_list)
    return accepted_tokens_list

def first_diff_index(s1, s2):
    for i, (a, b) in enumerate(zip(s1, s2)):
        if a != b:
            return i
    if len(s1) != len(s2):
        return min(len(s1), len(s2))  # one is a prefix of the other
    return "They're exactly same!"  # strings are identical


def no_kv_evaluate2(dataloader, model, batch, tokens_to_generate=10):
    predictions = []

    input_tokens = batch['tokens'].to(device)
    input_prompt = format_input(input_tokens)  # shape: [bs, seq]
    print('input_prompt:', decode(input_prompt))

    bs = input_prompt.shape[0]
    accepted_preds = input_prompt  # Initialize input sequence
    accepted_preds_list = []
    
    tokens_generated = 0

    while tokens_generated < tokens_to_generate:
        output = model(accepted_preds)  # pred[0] = base logits, shape: [bs, seq, vocab]
        base_logits = output[0][:, -1, :]  # last token logits
        next_token = base_logits.argmax(dim=-1, keepdim=True)  # [bs, 1]

        # Append base prediction to input
        accepted_preds = torch.cat([accepted_preds, next_token], dim=-1)
        accepted_preds_list.append(next_token.item())

        # Decode and store
        decoded = decode(next_token)
        predictions.extend(decoded)
        print(f"Generated token {tokens_generated + 1}: {decoded}")

        tokens_generated += 1

    print("Final prediction:", ''.join(predictions))
    print("Accepted token IDs:", accepted_preds_list)
    return ''.join(predictions)

def run():
    kv_model, no_kv_model = model
    diff_list = []
    # kv_model = model
    model_dtype = next(kv_model.parameters()).dtype
    i = 0
    prediction_consistencies = []
    for batch in dataloader:
    #     batch = torch.tensor([[128000, 128006,   9125, 128007,    271,    791,   3823,    374,   2663,
    #   21077,    323,    279,    892,    374,   7418,     11,   6250,    220,
    #    1114,    220,   2366,     18,     11,    220,   2839,     25,   1114,
    #    8971,  42084, 128009, 128006,    882, 128007,    271,   3923,    656,
    #     499,   1781,    922,  36142,   2191,     30, 128009, 128006,  78191,
    #  128007]], device='cuda:1')
        # batch = batch[:, :1]
        tokens_generated = 0
        accepted_preds = None
        next_token_kv = None
        no_kv_predictions = []; kv_predictions = []
        while(tokens_generated<tokens_to_generate):
            # print("-------no_kv_evaluate--------:")
            # pred_no_kv = no_kv_evaluate(dataloader, no_kv_model, batch, tokens_to_generate)
            base_logits_no_kv, next_token_no_kv, accepted_preds = no_kv_evaluate(dataloader, no_kv_model, batch, no_kv_predictions, tokens_to_generate, accepted_preds)
            # print("------kv_evaluate--------:")
            
            # pred_kv = kv_evaluate(dataloader, kv_model, batch, tokens_to_generate)
            # breakpoint()
            next_logits, next_token_kv, accepted_head_idx = kv_evaluate(dataloader, kv_model, batch, kv_predictions, tokens_to_generate, next_token_kv)
            tokens_generated += 1
            continue
        
            if next_token_no_kv != next_token_kv:
                p_log = F.log_softmax(base_logits_no_kv, dim=-1)
                q_log = F.log_softmax(base_logits_kv, dim=-1)
                kl = torch.sum(torch.exp(p_log) * (p_log - q_log))  # KL(p || q)
                # print("pred_kv, pred_no_kv, kl norm:", kl)
                kval, kindices = torch.topk(base_logits_kv, k = 4)
                nokval, nokindices = torch.topk(base_logits_no_kv, k = 4)
                print("next_token_kv, next_token_no_kv", next_token_kv.item(), next_token_no_kv.item())
                print("kindices, nokindices:", kindices, nokindices)
                print("kval, nokval:", kval, nokval)
                prediction_consistencies.append(tokens_generated)
                # if kval[0, 0] != kval[0, 1] and nokval[0, 0] != nokval[0, 1]:
                #     # breakpoint()
                #     prediction_consistencies.append(tokens_generated)
                break
                # else:
                #     break
            # else:
            #     print("Same!")

        print('------------------------------')
        print("kv_predictions:", ''.join(kv_predictions))
        print('------------------------------')
        print("no_kv_predictions:", ''.join(no_kv_predictions))
        print('------------------------------')
        break
        # diff = first_diff_index(pred_no_kv, pred_kv)
        # if isinstance(diff, int): 
        #     diff_list.append(diff)
        # print("Difference starts from:", diff)

        # p_log = F.log_softmax(pred_kv, dim=-1)
        # q_log = F.log_softmax(pred_no_kv, dim=-1)
        # kl = torch.sum(torch.exp(p_log) * (p_log - q_log))  # KL(p || q)
        # print("pred_kv, pred_no_kv, kl norm:", kl)
        tokens_generated = 0
        accepted_preds = None
        next_token_kv = None
        i+=1
        if i>=20:
            print('------------------------------')
            print("prediction_consistencies: ", prediction_consistencies)
            break
    print(diff_list)

if __name__ == "__main__":
    # Set environment variable for deterministic CuBLAS operations
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_printoptions(precision=9)  
    debug = True
    tokenizer_dir = '/home/ubuntu/.llama/checkpoints/Llama3.1-8B-Instruct/tokenizer.model'
    checkpoint_dir = '/home/ubuntu/vanshaj/inf2-training/3rdparty/torchtune/medusa_checkpoints/epoch_0/'
    dataset_dir = '/home/ubuntu/vanshaj/justpi.jsonl'
    torch.cuda.empty_cache()
    device = torch.device('cuda:1')
    max_cache_size = 500
    seed = 42
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    tokens_to_generate = 100

    dataloader = load_data(dataset_dir, tokenizer_dir, bs = 1)
    # checkpoint_dir = None

    model = load_model(checkpoint_dir)
    # no_kv_predictions = []
    # kv_predictions = []
    # evaluate(dataloader, model)
    run()