medusa_eval.pyimport torch
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



def load_data(dataset_dir, tokenizer_dir, bs = 4):

    tokenizer = Llama3Tokenizer(tokenizer_dir)    

    dataset = chat_dataset(
        tokenizer=tokenizer,
        source="json",
        data_files=dataset_dir,
        conversation_column="messages",
        conversation_style="openai"
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        collate_fn=torchtune.data.padded_collate_sft
    )
    for batch in dataloader:
        
        breakpoint()

        print("batch.shape:", batch.shape)
    return dataloader
    
def load_model(checkpoint_dir):
    
    with init_empty_weights():
        model = llama3_1_8b_medusa()
    checkpoint_dir += '/model-00001-of-00001.bin'
    checkpoint = torch.load(checkpoint_dir, map_location = 'cuda')

    model.load_state_dict(checkpoint, assign = True)
    model = model.cuda()
    model.eval()
    return model
def evaluate(dataloader, model):
    for batch in dataloader:
        # breakpoint()

        input_tokens = batch['tokens'].to('cuda')
        output = model(input_tokens)
        breakpoint()
        break

if __name__ == "__main__":
    tokenizer_dir = '/mnt/vast/share/inf2-training/models/open_source/Meta-Llama-3.1-8B-Instruct/original/tokenizer.model'
    checkpoint_dir = 'medusa_checkpoints/epoch_4'
    dataset_dir = '/mnt/vast/home/vanshaj/justpi_sft_identity_fixed_nosearch_nosys.jsonl'

    dataloader = load_data(dataset_dir, tokenizer_dir, 2)
    model = load_model(checkpoint_dir)
    evaluate(dataloader, model)