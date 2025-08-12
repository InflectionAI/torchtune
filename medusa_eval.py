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
from datetime import datetime
import uuid
# Set up logging globally here
logging.basicConfig(
    level=logging.DEBUG,  # or INFO in production
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# import sys, pdb, traceback; sys.excepthook = lambda t, v, tb: (traceback.print_exception(t, v, tb), pdb.post_mortem(tb))

# Force SDPA math kernel to reduce numeric drift across paths
def sdp_math_context():
    try:
        from torch.nn.attention import sdpa_kernel
        return sdpa_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    except Exception:
        return nullcontext()

global tokenizer

def check_stop_tokens(token_id, stop_tokens=None):
    """Check if a token is a stop token
    
    This function is used by individual generation functions to detect stop tokens
    and immediately halt their generation process. When a stop token is detected,
    the function returns True, signaling that generation should stop.
    """
    if stop_tokens is None:
        # Common stop tokens for Llama models
        stop_tokens = [
            128009,  # eot_id token
            128006,  # start_header_id token
            128007,  # end_header_id token
            2,       # </s> token
            1,       # <s> token
            78191,   # assistant token (common in chat models)
        ]
    return token_id in stop_tokens

def load_data(dataset_dir, tokenizer_dir, bs = 1):
    global tokenizer
    tokenizer = Llama3Tokenizer(tokenizer_dir)

    dataset = chat_dataset(
        tokenizer=tokenizer,
        source="json",
        data_files=dataset_dir,
        conversation_column="messages",
        conversation_style="openai",
        split = 'train[90%:]'
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
        # Create model with 5 Medusa heads to match the checkpoint
        model = llama3_1_8b_medusa(medusa_num_heads=5)
        if debug == True: 
            no_kv_model = llama3_1_8b_medusa(medusa_num_heads=5)
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

# def kl_div():
#     if next_token_no_kv != next_token_kv:
#         p_log = F.log_softmax(base_logits_no_kv, dim=-1)
#         q_log = F.log_softmax(base_logits_kv, dim=-1)
#         kl = torch.sum(torch.exp(p_log) * (p_log - q_log))  # KL(p || q)
#         # print("pred_kv, pred_no_kv, kl norm:", kl)
#         kval, kindices = torch.topk(base_logits_kv, k = 4)
#         nokval, nokindices = torch.topk(base_logits_no_kv, k = 4)
#         print("next_token_kv, next_token_no_kv", next_token_kv.item(), next_token_no_kv.item())
#         print("kindices, nokindices:", kindices, nokindices)
#         print("kval, nokval:", kval, nokval)
#         prediction_consistencies.append(tokens_generated)
        # if kval[0, 0] != kval[0, 1] and nokval[0, 0] != nokval[0, 1]:
        #     
        #     prediction_consistencies.append(tokens_generated)
        # break
        # else:
        #     break
    # else:
    #     print("Same!")

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

def no_kv_evaluate(dataloader, model, batch, no_kv_predictions, no_kv_tokens, tokens_to_generate = 5, accepted_preds = None):
    # predictions = []
    # global no_kv_predictions
    accepted_tokens_list = []
    
    
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
        # Appending next token to list of predicted tokens
        decoded = decode(next_token)
        no_kv_predictions.append(decoded)
        no_kv_tokens.append(next_token)
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
        no_kv_tokens.append(next_token)
        
        # Check for stop tokens and stop generating immediately
        if check_stop_tokens(next_token.item()):
            print(f"🚫 Stop token {next_token.item()} detected in no_kv_evaluate, stopping generation")
            print(f"Decoded stop token: {decode(torch.tensor([[next_token.item()]], device=device))}")
            return base_logits, next_token, accepted_preds, True  # True indicates stop token found
            
        # Continue generating more tokens if no stop token
        # This would be where you'd add a loop to generate more tokens
        # For now, we just return after the first token
        
    return base_logits, next_token, accepted_preds, False  # False indicates no stop token
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
        # Fix: properly stack medusa logits from all heads
        medusa_logits = torch.stack(output[1:], dim=0) # shape: [n, bs, seq_len, vocab_dim]
        # Transpose to get [bs, seq_len, n, vocab_dim] for easier processing
        medusa_logits = medusa_logits.transpose(0, 1).transpose(1, 2) # shape: [bs, seq_len, n, vocab_dim]
    
    medusa_preds = medusa_logits.argmax(dim = -1) # shape: [bs, seq_len, n] or [bs, 1, n] for first_decode
    return medusa_preds, medusa_logits
    
def kv_evaluate(dataloader, model, batch, kv_predictions, kv_tokens, accepted_heads, accelerated_predictions, tokens_to_generate = 5, input_token = None):
    '''Eval using kv caching'''
    layer = get_attn_layer(model)
    model_dtype = get_model_dtype(model)
    accepted_tokens_list = []
    
    
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

        base_logits = output[0][:, -1:]  # shape: [bs, 1, vocab_dim]
        vocab_dim = base_logits.shape[-1]
        next_token = base_logits.argmax(dim=-1)  # shape: [bs, 1]
        
        n = len(output[1:]) # num medusa heads
        # get last medusa logits
        medusa_preds, medusa_logits = get_medusa_preds(output, first_decode = True)
    
        # accepted_tokens_list.append(next_token.item())
        decoded_prediction = decode(next_token)
        kv_predictions.append(decoded_prediction)
        kv_tokens.append(next_token)
        # Check for stop tokens in the first generated token
        if check_stop_tokens(next_token.item()):
            print(f"🚫 Stop token {next_token.item()} detected in kv_evaluate (first token), stopping generation")
            print(f"Decoded stop token: {decode(torch.tensor([[next_token.item()]], device=device))}")
            return combined_logits, combined_preds, None, True  # True indicates stop token found
        
        # Continue with normal processing if no stop token
        combined_preds = torch.cat((next_token, medusa_preds), dim = 1) # shape: [bs, 1+n]
        combined_logits = torch.cat((base_logits, medusa_logits), dim = 1) # shape: [bs, 1+n, vocab_dim]

        # update kv cache
        curr_kv_len = curr_seq_len
        model_kv_len = layer.attn.kv_cache.size
        assert (curr_kv_len == int(model_kv_len))
        # curr_seq_len = 1
        print("-----------------------------------------------------")
        print('cache len now:', layer.attn.kv_cache.size)
        return combined_logits, combined_preds, None, False  # False indicates no stop token
    else:
        
        bs = input_token.shape[0]
        prev_preds = input_token  # This contains [base_token, medusa_tokens] from previous iteration
        n = model.medusa_num_heads
        # Verify kv cache size
        curr_kv_len = layer.attn.kv_cache.size
        curr_seq_len = 1 + n # seq contains base_model pred with all medusa preds
        
        # Create mask and positions for the new tokens
        causal_mask = create_causal_mask(bs, curr_seq_len, curr_kv_len, max_cache_size, device, model_dtype)
        input_pos = torch.arange(curr_kv_len, curr_kv_len + curr_seq_len, device=device).unsqueeze(0)
        
        assert(input_pos.shape == (bs, 1+n))
        
        # Forward pass with the new tokens
        with sdp_math_context():
            output = model(prev_preds, mask=causal_mask, input_pos=input_pos) # shape: list [bs, 1+n, vocab_dim]
        
        base_logits = output[0]  # shape: [bs, 1+n, vocab_dim] where seq_len = 1+n
        base_tokens = base_logits.argmax(dim=-1)  # shape: [bs, 1+n]
        medusa_preds, medusa_logits = get_medusa_preds(output) # shape: [bs, 1+n, n]
        
        # CORRECTED: Compare base model predictions with previous medusa predictions
        # The base model should predict the same tokens that were previously predicted by medusa heads
        # prev_preds[:, 1:] contains the previous medusa predictions
        # base_tokens[:, :-1] contains what the base model predicts for those positions
        mask = (prev_preds[:, 1:] == base_tokens[:, :-1])
        correct_pred_mask = mask.cumprod(dim=1)
        last_accepted_head = correct_pred_mask.sum().item()
        
        accepted_heads.append(last_accepted_head)
        # debug
        # last_accepted_head = 0

        print("prev_preds:", prev_preds, decode(prev_preds))
        print("base_tokens:", base_tokens, decode(base_tokens))
        print("mask:", mask)
        print("correct_pred_mask:", correct_pred_mask)
        print("last_accepted_head:", last_accepted_head)
        
        # The number of tokens to accept is last_accepted_head + 1
        # (the base token + the accepted medusa tokens)
        accepted_tokens_count = last_accepted_head + 1
        

        # Update KV cache length
        curr_kv_len += accepted_tokens_count
        
        # Reset cache to valid length (remove invalid medusa predictions)
        print("-----------------------------------------------------")
        print('cache len before:', layer.attn.kv_cache.size)
        model.revert_cache_to_valid_length(curr_kv_len)
        print('cache len after:', layer.attn.kv_cache.size)
        assert(layer.attn.kv_cache.size == curr_kv_len)
        
        accepted_preds = base_tokens[:, :last_accepted_head+1].unsqueeze(-1)  # shape: [bs, 1]
        # Prepare next input: base token + medusa predictions for next iteration
        base_token = base_tokens[:, last_accepted_head].unsqueeze(-1)  # shape: [bs, 1]
        medusa_tokens = medusa_preds[:, last_accepted_head]  # shape: [bs, n]
        
        # Analyze the accepted tokens
        if last_accepted_head>0:
            accelerated_tokens = base_tokens[:, 1:last_accepted_head+1].unsqueeze(-1)
            accelerated_prediction = decode(accelerated_tokens)
            accelerated_predictions.append(accelerated_prediction)
        next_tokens = torch.cat((base_token, medusa_tokens), dim=-1)  # shape: [bs, 1+n]
        next_logits = torch.cat((
            base_logits[:, last_accepted_head].unsqueeze(1), 
            medusa_logits[:, last_accepted_head]
        ), dim=-2)
        
        # Decode and store the accepted base token
        decoded_prediction = decode(accepted_preds)
        kv_predictions.append(decoded_prediction)
        kv_tokens.append(accepted_preds)
        # Check for stop tokens in the accepted base token
        if check_stop_tokens(base_tokens[:, last_accepted_head].item()):
            print(f"🚫 Stop token {base_tokens[:, last_accepted_head].item()} detected in kv_evaluate, stopping generation")
            print(f"Decoded stop token: {decode(torch.tensor([[base_tokens[:, last_accepted_head].item()]], device=device))}")
            return next_logits, next_tokens, last_accepted_head, True  # True indicates stop token found
        
        # Continue with normal processing if no stop token
        print("-----------------------------------------------------")
        return next_logits, next_tokens, last_accepted_head, False  # False indicates no stop token
        
    print("-----------------------------------------------------")
    print("Prediction: ", ''.join(predictions))
    print("-----------------------------------------------------")

    print("Accepted token IDs:", accepted_tokens_list)
    return accepted_tokens_list

def find_index_in_list(target_index, predlist):
    index_sum = 0
    for i, elem in enumerate(predlist):
        index_sum += len(elem)
        if index_sum>target_index:
            return i-1
        if index_sum == target_index:
            return i
    return None

def first_diff_index(kv_pred, no_kv_pred):
    kv_str = ''.join(kv_pred)
    no_kv_str = ''.join(no_kv_pred)
    for i, (a, b) in enumerate(zip(kv_str, no_kv_str)):
        # if they diverge
        if a != b:
            break
    kv_index = find_index_in_list(i, kv_pred)
    no_kv_index = find_index_in_list(i, no_kv_pred)
    return i, kv_index, no_kv_index
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

def log_evaluation_results(prompt_num, prompt_text, kv_predictions_str, no_kv_predictions_str, 
                          kv_tokens, no_kv_tokens, avg_acceleration, value_avg_acceleration, 
                          accelerated_predictions, diff_list, accepted_heads, kv_index, no_kv_index,
                          avg_head_acceptance, tokens_generated, kv_stop, no_kv_stop, 
                          kv_stop_token, no_kv_stop_token, log_file="medusa_eval_results.jsonl"):
    """
    Log evaluation results for each prompt to a JSONL file for later analysis.
    JSONL format allows appending results line by line without loading entire file.
    """
    
    # Helper function to convert tensors to serializable types
    def tensor_to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.item()
            else:
                return obj.tolist()
        elif isinstance(obj, list):
            return [tensor_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: tensor_to_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    # Create log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt_id": str(uuid.uuid4()),
        "prompt_number": prompt_num,
        "prompt_text": prompt_text,
        "results": {
            "kv_predictions": kv_predictions_str,
            "no_kv_predictions": no_kv_predictions_str,
            "kv_tokens": tensor_to_serializable(kv_tokens),
            "no_kv_tokens": tensor_to_serializable(no_kv_tokens),
            "tokens_generated": tokens_generated,
            "divergence": {
                "kv_index": kv_index,
                "no_kv_index": no_kv_index,
                "difference_starts_at": no_kv_index,
                "total_no_kv_tokens": len(no_kv_tokens) if no_kv_tokens else 0
            },
            "head_acceptance": {
                "accepted_heads": tensor_to_serializable(accepted_heads),
                "accepted_heads_till_divergence": tensor_to_serializable(accepted_heads[:kv_index-1] if kv_index and kv_index > 1 else []),
                "avg_head_acceptance": avg_head_acceptance
            },
            "acceleration_metrics": {
                "current_acceleration": 1 + avg_head_acceptance if avg_head_acceptance is not None else 0,
                "cumulative_avg_acceleration": value_avg_acceleration,
                "all_accelerations": tensor_to_serializable(avg_acceleration),
                "accelerated_predictions": tensor_to_serializable(accelerated_predictions)
            },
            "stop_token_analysis": {
                "kv_stopped": kv_stop,
                "no_kv_stopped": no_kv_stop,
                "kv_stop_token_id": kv_stop_token,
                "no_kv_stop_token_id": no_kv_stop_token,
                "kv_stop_token_decoded": decode(torch.tensor([[kv_stop_token]], device=device)) if kv_stop_token is not None and 'decode' in globals() else None,
                "no_kv_stop_token_decoded": decode(torch.tensor([[no_kv_stop_token]], device=device)) if no_kv_stop_token is not None and 'decode' in globals() else None
            }
        },
        "metadata": {
            "model_checkpoint": checkpoint_dir if 'checkpoint_dir' in globals() else "unknown",
            "dataset": dataset_dir if 'dataset_dir' in globals() else "unknown",
            "tokens_to_generate": tokens_to_generate if 'tokens_to_generate' in globals() else "unknown"
        }
    }
    
    # Append to JSONL file (one JSON object per line)
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    print(f"📝 Results logged to {log_file} for prompt {prompt_num}")
    return log_entry

def analyze_jsonl_results(log_file="medusa_eval_results.jsonl"):
    """
    Utility function to analyze the logged JSONL results.
    Returns a summary of all evaluations.
    """
    results = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    except FileNotFoundError:
        print(f"Log file {log_file} not found.")
        return None
    
    if not results:
        print("No results found in log file.")
        return None
    
    # Calculate summary statistics
    total_prompts = len(results)
    avg_accelerations = [r['results']['acceleration_metrics']['current_acceleration'] for r in results]
    avg_acceleration = sum(avg_accelerations) / len(avg_accelerations) if avg_accelerations else 0
    
    # Find divergence points
    divergence_points = [r['results']['divergence']['difference_starts_at'] for r in results if r['results']['divergence']['difference_starts_at'] is not None]
    avg_divergence = sum(divergence_points) / len(divergence_points) if divergence_points else 0
    
    # Stop token statistics
    kv_stops = sum(1 for r in results if r['results']['stop_token_analysis']['kv_stopped'])
    no_kv_stops = sum(1 for r in results if r['results']['stop_token_analysis']['no_kv_stopped'])
    
    # Most common stop tokens
    kv_stop_tokens = [r['results']['stop_token_analysis']['kv_stop_token_id'] for r in results if r['results']['stop_token_analysis']['kv_stop_token_id'] is not None]
    no_kv_stop_tokens = [r['results']['stop_token_analysis']['no_kv_stop_token_id'] for r in results if r['results']['stop_token_analysis']['no_kv_stop_token_id'] is not None]
    
    summary = {
        "total_prompts": total_prompts,
        "average_acceleration": avg_acceleration,
        "average_divergence_point": avg_divergence,
        "total_tokens_generated": sum(r['results']['tokens_generated'] for r in results),
        "stop_token_stats": {
            "kv_stops": kv_stops,
            "no_kv_stops": no_kv_stops,
            "kv_stop_tokens": kv_stop_tokens,
            "no_kv_stop_tokens": no_kv_stop_tokens
        },
        "results": results
    }
    
    print(f"📊 Analysis Summary:")
    print(f"   Total prompts evaluated: {total_prompts}")
    print(f"   Average acceleration: {avg_acceleration:.3f}")
    print(f"   Average divergence point: {avg_divergence:.1f} tokens")
    print(f"   Total tokens generated: {summary['total_tokens_generated']}")
    print(f"   KV model stopped: {kv_stops}/{total_prompts} prompts")
    print(f"   No-KV model stopped: {no_kv_stops}/{total_prompts} prompts")
    
    return summary

def run():
    kv_model, no_kv_model = model
    diff_list = []
    # kv_model = model
    model_dtype = next(kv_model.parameters()).dtype
    prompt_num = 0
    prediction_consistencies = []
    avg_acceleration = []; outputs = []; accepted_heads = []; accelerated_predictions = []
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
        no_kv_predictions = []; kv_predictions = []; no_kv_tokens = []; kv_tokens = []
        no_kv_stop = kv_stop = False
        prompt_num += 1
        
        # Decode the prompt for logging
        try:
            prompt_text = tokenizer.decode(batch[0], skip_special_tokens=True)
        except:
            prompt_text = f"Prompt {prompt_num} (could not decode)"
            
        while(tokens_generated<tokens_to_generate):
            print(f"--- Generation step {tokens_generated + 1} ---")
            # print("-------no_kv_evaluate--------:")
            if not no_kv_stop:
                base_logits_no_kv, next_token_no_kv, accepted_preds, no_kv_stop = no_kv_evaluate(dataloader, no_kv_model, batch, no_kv_predictions, no_kv_tokens, tokens_to_generate, accepted_preds)
                if no_kv_stop:
                    print(f"🚫 no_kv stopped at step {tokens_generated + 1}")

            # print("------kv_evaluate--------:")
            if not kv_stop:
                next_logits, next_token_kv, accepted_head_idx, kv_stop = kv_evaluate(dataloader, kv_model, batch, kv_predictions, kv_tokens, accepted_heads, accelerated_predictions, tokens_to_generate, next_token_kv)
                if kv_stop:
                    print(f"🚫 kv stopped at step {tokens_generated + 1}")

            tokens_generated += 1
            
            # Break the loop if both models have stopped
            if kv_stop and no_kv_stop:
                print(f"Both models have stopped, ending generation at token {tokens_generated}")
                break
        kv_predictions_str = ''.join(kv_predictions)
        no_kv_predictions_str = ''.join(no_kv_predictions)
        print('------------------------------')
        print("kv_predictions:", kv_predictions_str)
        print('------------------------------')
        print("no_kv_predictions:", no_kv_predictions_str)
        print('------------------------------')

        diff, kv_index, no_kv_index = first_diff_index(kv_predictions, no_kv_predictions)
        if isinstance(no_kv_index, int): 
            diff_list.append(no_kv_index)
        print("Difference starts from:", no_kv_index, " out of ", len(no_kv_predictions))
        print('accepted_heads till divergence:', accepted_heads[:kv_index-1])
        avg_head_acceptance = (sum(accepted_heads[:kv_index-1])/len(accepted_heads[:kv_index-1])) if len(accepted_heads[:kv_index-1])!=0 else 0

        print('avg accepted_heads upon divergence:', )
        avg_acceleration.append(1+avg_head_acceptance)
        preds = {"medusa:", kv_predictions_str, "no medusa:", no_kv_predictions_str}
        outputs.append(preds)
        tokens_generated = 0
        accepted_preds = None
        next_token_kv = None
        prompt_num+=1
        value_avg_acceleration = sum(avg_acceleration)/len(avg_acceleration)

        # logging
        kv_predictions_str, kv_tokens, no_kv_tokens, no_kv_predictions_str
        avg_acceleration, value_avg_acceleration, accelerated_predictions
        print("Difference starts from:", no_kv_index, " out of ", len(no_kv_predictions))
        
        # Get stop token information
        kv_stop_token = None
        no_kv_stop_token = None
        
        if kv_stop and kv_tokens:
            # Get the last token that caused KV model to stop
            last_kv_token = kv_tokens[-1]
            if isinstance(last_kv_token, torch.Tensor):
                kv_stop_token = last_kv_token.item() if last_kv_token.numel() == 1 else last_kv_token.flatten()[-1].item()
            else:
                kv_stop_token = last_kv_token
                
        if no_kv_stop and no_kv_tokens:
            # Get the last token that caused no-KV model to stop
            last_no_kv_token = no_kv_tokens[-1]
            if isinstance(last_no_kv_token, torch.Tensor):
                no_kv_stop_token = last_no_kv_token.item() if last_no_kv_token.numel() == 1 else last_no_kv_token.flatten()[-1].item()
            else:
                no_kv_stop_token = last_no_kv_token
        
        # Log results for this prompt
        log_evaluation_results(
            prompt_num=prompt_num,
            prompt_text=prompt_text,
            kv_predictions_str=kv_predictions_str,
            no_kv_predictions_str=no_kv_predictions_str,
            kv_tokens=kv_tokens,
            no_kv_tokens=no_kv_tokens,
            avg_acceleration=avg_acceleration,
            value_avg_acceleration=value_avg_acceleration,
            accelerated_predictions=accelerated_predictions,
            diff_list=diff_list,
            accepted_heads=accepted_heads,
            kv_index=kv_index,
            no_kv_index=no_kv_index,
            avg_head_acceptance=avg_head_acceptance,
            tokens_generated=tokens_generated,
            kv_stop=kv_stop,
            no_kv_stop=no_kv_stop,
            kv_stop_token=kv_stop_token,
            no_kv_stop_token=no_kv_stop_token
        )
        
        if prompt_num>=num_prompts:
        
            print('------------------------------')
            value_avg_acceleration = sum(avg_acceleration)/len(avg_acceleration)
            print('avg_acceleration:', avg_acceleration)
            print('value_avg_acceleration:', value_avg_acceleration)
            print("accelerated_predictions:", accelerated_predictions)

            breakpoint()

    # print('------------------------------')
    # value_avg_acceleration = sum(avg_acceleration)/len(avg_acceleration)
    # print('avg_acceleration:', avg_acceleration)
    # print('value_avg_acceleration:', value_avg_acceleration)
    # print("accelerated_predictions:", accelerated_predictions)

        

if __name__ == "__main__":
    # Set environment variable for deterministic CuBLAS operations
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_printoptions(precision=9)  
    debug = True
    tokenizer_dir = '/home/ubuntu/.llama/checkpoints/Llama3.1-8B-Instruct/tokenizer.model'
    # checkpoint_dir = '/home/ubuntu/vanshaj/inf2-training/3rdparty/torchtune/medusa_checkpoints/epoch_0/'
    checkpoint_dir = '/home/ubuntu/vanshaj/torchtune/medusa_sharegpt_val_checkpoints/epoch_2000046/'
    # dataset_dir = '/home/ubuntu/vanshaj/justpi.jsonl'
    dataset_dir = 'sharegpt_60k_full.jsonl'
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

    tokens_to_generate = 200

    dataloader = load_data(dataset_dir, tokenizer_dir, bs = 1)
    # checkpoint_dir = None
    num_prompts = 20
    model = load_model(checkpoint_dir)
    # no_kv_predictions = []
    # kv_predictions = []
    # evaluate(dataloader, model)
    run()