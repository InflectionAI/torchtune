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

import sys, pdb, traceback; sys.excepthook = lambda t, v, tb: (traceback.print_exception(t, v, tb), pdb.post_mortem(tb))

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

    torch.set_grad_enabled(False)

    return model

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

def decode(x):
    return tokenizer.decode(x.flatten().tolist(), skip_special_tokens = False )

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
    # breakpoint()
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

        causal_mask = create_causal_mask(bs, curr_seq_len, curr_kv_len, max_cache_size, device, model_dtype)
        input_pos = torch.arange(curr_seq_len, device = device).unsqueeze(0)
        print("model input: ", input_prompt)

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
            causal_mask = create_causal_mask(bs, curr_seq_len, curr_kv_len, max_cache_size, device, model_dtype)
            # shape: [bs, curr_seq, self.encoder_max_cache_seq_len], boolean mask with True representing queries to attend
            input_pos = torch.arange(curr_kv_len, curr_kv_len + curr_seq_len, device=device).unsqueeze(0)
            # All True rect mask of new_tokens x tokens_generated | upper_triangular mask of new_tokens x new_tokens + False rect mask of new_tokens x (encoder_max_cache_seq_len - (tokens_generated + new_tokens))
            print("model input: ", preds)
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
            # breakpoint()
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

if __name__ == "__main__":
    tokenizer_dir = '/home/ubuntu/.llama/checkpoints/Llama3.1-8B-Instruct/tokenizer.model'
    checkpoint_dir = '/home/ubuntu/vanshaj/inf2-training/3rdparty/torchtune/medusa_checkpoints/epoch_0/'
    dataset_dir = '/home/ubuntu/vanshaj/justpi.jsonl'
    torch.cuda.empty_cache()
    device = torch.device('cuda:1')
    max_cache_size = 256
    dataloader = load_data(dataset_dir, tokenizer_dir, bs = 1)
    # checkpoint_dir = None
    model = load_model(checkpoint_dir)

    evaluate(dataloader, model)