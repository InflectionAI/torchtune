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
    return dataloader, tokenizer
    
def load_model(checkpoint_dir, device, debug = False):
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
    if debug == True:
        return model, no_kv_model
    
    return model

def setup():
    # Set environment variable for deterministic CuBLAS operations
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_printoptions(precision=9)  
    debug = False
    tokenizer_dir = '/home/ubuntu/.llama/checkpoints/Llama3.1-8B-Instruct/tokenizer.model'
    # checkpoint_dir = '/home/ubuntu/vanshaj/inf2-training/3rdparty/torchtune/medusa_checkpoints/epoch_0/'
    # checkpoint_dir = '/home/ubuntu/vanshaj/torchtune/medusa_sharegpt_val_checkpoints/epoch_2000046/'
    checkpoint_dir = '/home/ubuntu/vanshaj/torchtune/checkpoints_backup/val16_5/medusa_justpi_val_checkpoints/epoch_10420/'
    dataset_dir = '/home/ubuntu/vanshaj/justpi.jsonl'
    # dataset_dir = 'sharegpt_60k_full.jsonl'
    torch.cuda.empty_cache()
    device = torch.device('cuda:1')
    max_cache_size = 500
    seed = 42

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    tokens_to_generate = 200

    dataloader, tokenizer = load_data(dataset_dir, tokenizer_dir, bs = 1)
    num_prompts = 20
    model = load_model(checkpoint_dir, device, debug)
    return model, dataloader, tokenizer