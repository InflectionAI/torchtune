import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4" # define GPU id, remove if you want to use all GPUs available
os.chdir('..')
import torch
from contextlib import contextmanager
import numpy as np
from medusa.model.medusa_model import MedusaModel
from medusa.model.kv_cache import *
from medusa.model.utils import *
from medusa.model.medusa_choices import *
from copy import deepcopy
import matplotlib.pyplot as plt


model_name = 'FasterDecoding/medusa-vicuna-7b-v1.3'
model = MedusaModel.from_pretrained(
    model_name,
    # medusa_num_heads = 4,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
tokenizer = model.get_tokenizer()

medusa_choices = mc_sim_7b_63


past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.base_model)
model.past_key_values = past_key_values
model.past_key_values_data = past_key_values_data
model.current_length_data = current_length_data

model.current_length_data.zero_() # this is for rerun
prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hi, could you share a tale about a charming llama that grows Medusa-like hair and starts its own coffee shop? ASSISTANT:"
input_ids = tokenizer([prompt]).input_ids
input_len = len(input_ids[0])
print('Input token length:', len(input_ids[0]))
print('Init KV cache shape for attention modules:', model.past_key_values[0][0].shape, model.past_key_values[0][1].shape)

free, total = torch.cuda.mem_get_info()
print(f"Free memory: {free / 1024**2:.2f} MB")
print(f"Total memory: {total / 1024**2:.2f} MB")
print(f"Used memory: {(total - free) / 1024**2:.2f} MB")


print("start")
inference_count = 0
accept_lengths = []
with torch.inference_mode():
    breakpoint()
    input_ids = tokenizer([prompt]).input_ids
    input_len = len(input_ids[0])
    input_ids = torch.as_tensor(input_ids).cuda()
    model.current_length_data.zero_() # this is for rerun
    medusa_logits, outputs, logits = model(input_ids, output_orig = True, past_key_values=model.past_key_values, medusa_forward=True)
    inference_count += 1

    medusa_pred = torch.argmax(medusa_logits[..., -1, :], dim = -1)
    pred = torch.argmax(logits[..., -1, :], dim = -1)
    preds = torch.cat([pred, medusa_pred[:, 0 ]], dim = -1)
    print(f'Prediction @ {inference_count}: {tokenizer.batch_decode(pred)}')
    cur_length = input_len
    accept_lengths.append(1)
    for _ in range(1024):
        medusa_logits, outputs, logits = model(preds.cuda().unsqueeze(0), output_orig = True, past_key_values = model.past_key_values, medusa_forward=True)
        inference_count += 1

        medusa_pred = torch.argmax(medusa_logits[..., -5:, :], dim = -1)
        pred = torch.argmax(logits[..., :, :], dim = -1)
        posterior_mask = (
                    preds[1:] == pred[0, :-1]
                ).int()
        accept_length = torch.cumprod(posterior_mask, dim = -1).sum().item()
        cur_length = cur_length + accept_length + 1
        # update kv cache
        model.current_length_data.fill_(cur_length)
        # create new input
        preds = torch.cat([pred[:, accept_length], medusa_pred[:,0,accept_length]], dim = -1)
        print(f'Prediction @ {inference_count}: {tokenizer.batch_decode(pred[0, :accept_length + 1])}')
        accept_lengths.append(accept_length + 1)
        if tokenizer.eos_token_id in pred[0, :accept_length + 1]:
            break