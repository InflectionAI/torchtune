import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # define GPU id, remove if you want to use all GPUs available
# os.chdir('../')
print(os.getcwd())
import torch
from contextlib import contextmanager
import numpy as np
# from medusa.model.medusa_model import MedusaModel
# from medusa.model.kv_cache import *
from medusa_utils import *
from medusa_choices import *
from copy import deepcopy
import torchtune_tree_utils

''' Their model:
model_name = 'FasterDecoding/medusa-vicuna-7b-v1.3'
model = MedusaModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    # medusa_forward = True
)
tokenizer = model.get_tokenizer()
'''
medusa_choices = mc_sim_7b_63
prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hi, could you share a tale about a charming llama that grows Medusa-like hair and starts its own coffee shop? ASSISTANT:"
print(prompt)
accept_lengths_tree = []
model, dataloader, tokenizer = torchtune_tree_utils.setup()
with torch.inference_mode():
    new_token = 0
    input_ids = tokenizer([prompt]).input_ids
    input_len = len(input_ids[0])

    input_ids = torch.as_tensor(input_ids).cuda()
    # breakpoint()
    model.current_length_data.zero_() # this is for rerun
    reset_medusa_mode(model)
    medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=model.base_model.device
            )
    medusa_logits, logits = initialize_medusa(
            input_ids, model, medusa_buffers["medusa_attn_mask"]
        )
    cur_length = input_len + 1
    accept_lengths_tree.append(1)
    valid_kv_len = 0
    ''' There exists a loop invariant such that every time the loop runs, the kv cache only contains valid entries '''
    for i in range(1024):

        layer, valid_kv_len = get_attn_layer(model)
        # no fwd pass occurs here. The logits are sampled and mapped into a tree structure of multiple candidate [medusa_predictions]
        candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
            )
        
        # model fwd_pass to generate + verify tree_candidates occurs here
        
        medusa_logits, logits, outputs = tree_decoding(
                model,
                tree_candidates,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
            )
        best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature = 0, posterior_threshold = 0, posterior_alpha = 0
            )
        
        # model fwd_pass to generate + verify tree_candidates occurs here
        input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits,
                new_token,
                model,
                valid_kv_len
            )
        
        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_lengths_tree.append(accept_length_tree)
        if model.tokenizer.eos_token_id in input_ids[0, input_len:]:
            break
print('Decode:', tokenizer.batch_decode(input_ids[:,input_len:]))