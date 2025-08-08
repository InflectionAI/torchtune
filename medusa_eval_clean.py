"""
Medusa evaluation script for comparing KV cache and non-KV cache model inference.

This script evaluates a Medusa model with and without KV caching to compare
prediction consistency and performance differences.
"""

import logging
import os
import random
from contextlib import nullcontext
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaTokenizer

import torchtune
from torchtune.datasets._chat import chat_dataset
from torchtune.models.llama3._tokenizer import Llama3Tokenizer
from torchtune.models.llama3_1 import llama3_1_8b_medusa
from torchtune.modules import TransformerSelfAttentionLayer
from torchtune.modules.common_utils import disable_kv_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MedusaEvaluator:
    """Evaluator for Medusa model with KV cache comparison."""
    
    def __init__(self, config):
        """Initialize the evaluator with configuration."""
        self.config = config
        self.device = torch.device(config.get('device', 'cuda:0'))
        self.tokenizer = None
        self.model = None
        self.no_kv_model = None
        
        # Set deterministic behavior
        self._setup_deterministic()
        
    def _setup_deterministic(self):
        """Setup deterministic behavior for reproducible results."""
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        
        # Set seeds
        seed = self.config.get('seed', 42)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    def _sdp_math_context(self):
        """Force SDPA math kernel to reduce numeric drift across paths."""
        try:
            from torch.backends.cuda import sdp_kernel
            return sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except Exception:
            return nullcontext()
    
    def load_data(self, dataset_dir, tokenizer_dir, batch_size=1):
        """Load and prepare the dataset."""
        self.tokenizer = Llama3Tokenizer(tokenizer_dir)
        
        dataset = chat_dataset(
            tokenizer=self.tokenizer,
            source="json",
            data_files=dataset_dir,
            conversation_column="messages",
            conversation_style="openai",
            split='train[80%:90%]'
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=torchtune.data.padded_collate_sft
        )
        return dataloader
    
    def load_model(self, checkpoint_dir):
        """Load the Medusa model from checkpoint."""
        if checkpoint_dir is None:
            return None
            
        with init_empty_weights():
            model = llama3_1_8b_medusa()
            if self.config.get('debug', False):
                no_kv_model = llama3_1_8b_medusa()
        
        checkpoint_path = os.path.join(checkpoint_dir, 'model-00001-of-00001.bin')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load main model
        model.load_state_dict(checkpoint, assign=True)
        model = model.to(self.device)
        model.eval()
        
        # Setup caches
        model_dtype = next(model.parameters()).dtype
        max_cache_size = self.config.get('max_cache_size', 256)
        with self.device:
            model.setup_caches(batch_size=1, dtype=model_dtype, decoder_max_seq_len=max_cache_size)
        
        # Load no-KV model if in debug mode
        if self.config.get('debug', False):
            no_kv_model.load_state_dict(checkpoint, assign=True)
            no_kv_model = no_kv_model.to(self.device)
            no_kv_model.eval()
            self.no_kv_model = no_kv_model
        
        torch.set_grad_enabled(False)
        self.model = model
        return model
    
    def format_input(self, input_tokens):
        """Format input tokens for the model."""
        assistant_token = 78191
        assistant_append_tokens = torch.tensor([128007], device=self.device)
        
        tensor_2d = input_tokens  # shape: [1, seq_len]
        tensor_1d = tensor_2d[0]  # shape: [seq_len]
        
        for i, token in enumerate(tensor_1d):
            if int(token) == assistant_token:
                break
        
        part1 = torch.cat((tensor_1d[:i+1], assistant_append_tokens))
        return part1.unsqueeze(0)
    
    def decode(self, tokens):
        """Decode tokens to text."""
        return self.tokenizer.decode(tokens.flatten().tolist(), skip_special_tokens=False)
    
    def create_causal_mask(self, batch_size, current_seq_len, cached_seq_len, max_cache_size):
        """Create a causal mask for Medusa evaluation with KV cache."""
        total_seq_len = cached_seq_len + current_seq_len
        model_dtype = next(self.model.parameters()).dtype
        
        # Create masks for old tokens and new tokens
        old_tokens_mask = torch.ones(
            current_seq_len, cached_seq_len, 
            device=self.device, dtype=model_dtype
        )
        new_tokens_triangular_mask = torch.tril(
            torch.ones(current_seq_len, current_seq_len, device=self.device, dtype=model_dtype)
        )
        
        # Create suffix mask
        suffix_mask_dim = max_cache_size - total_seq_len
        mask_suffix = torch.zeros(
            (current_seq_len, suffix_mask_dim), 
            device=self.device, dtype=model_dtype
        )
        
        # Combine masks
        mask = torch.cat((old_tokens_mask, new_tokens_triangular_mask, mask_suffix), dim=-1)
        mask = mask.unsqueeze(0).bool()  # [1, current_seq_len, total_seq_len]
        
        return mask
    
    def evaluate_medusa(self, dataloader, tokens_to_generate=100):
        """Evaluate the Medusa model with KV caching."""
        for batch in dataloader:
            self.model.reset_caches()
            model_dtype = next(self.model.parameters()).dtype
            
            input_tokens = batch['tokens'].to(self.device)
            input_prompt = self.format_input(input_tokens)
            
            logger.info(f'Input prompt: {self.decode(input_prompt)}')
            
            bs = input_prompt.shape[0]
            curr_seq_len = input_prompt.shape[1]
            curr_kv_len = 0
            
            # Initial forward pass
            causal_mask = self.create_causal_mask(
                bs, curr_seq_len, 0, self.config.get('max_cache_size', 256)
            )
            input_pos = torch.arange(curr_seq_len, device=self.device).unsqueeze(0)
            
            with self._sdp_math_context():
                output = self.model(input_prompt, mask=causal_mask, input_pos=input_pos)
            
            # Process base and medusa predictions
            base_logits = output[0][:, -1]
            pred = base_logits.argmax(dim=-1)
            
            medusa_logits = torch.stack(output[1:])[:, :, -1]
            medusa_out = medusa_logits.argmax(dim=-1).permute((1, 0))
            
            tokens_generated = 1
            preds = torch.cat((pred.unsqueeze(-1), medusa_out), dim=-1)
            accept_lengths = []
            curr_kv_len = curr_seq_len
            
            # Generate tokens
            while tokens_generated < tokens_to_generate:
                curr_seq_len = preds.shape[1]
                causal_mask = self.create_causal_mask(
                    bs, curr_seq_len, curr_kv_len, self.config.get('max_cache_size', 256)
                )
                input_pos = torch.arange(
                    curr_kv_len, curr_kv_len + curr_seq_len, device=self.device
                ).unsqueeze(0)
                
                with self._sdp_math_context():
                    pred = self.model(preds, mask=causal_mask, input_pos=input_pos)
                
                base_logits = pred[0]
                medusa_logits = torch.stack(pred[1:])
                base_out = base_logits.argmax(dim=-1)
                medusa_out = medusa_logits.argmax(dim=-1)
                
                # Check which medusa heads were correct
                mask = (base_out[:, :-1] == preds[:, 1:])
                correct_pred_mask = mask.cumprod(dim=-1)
                last_accepted_head = correct_pred_mask.sum().item()
                
                curr_kv_len += (last_accepted_head + 1)
                self.model.revert_cache_to_valid_length(curr_kv_len)
                tokens_generated += (last_accepted_head + 1)
                
                # Prepare next input
                accepted_head_medusa_pred = medusa_out[:, :, last_accepted_head].transpose(0, 1)
                preds = torch.cat((
                    base_out[:, last_accepted_head:last_accepted_head + 1], 
                    accepted_head_medusa_pred
                ), dim=-1)
                
                accept_lengths.append(last_accepted_head + 1)
                
                # Decode accepted tokens
                accepted_tokens = base_out[0, :last_accepted_head + 1]
                decoded_prediction = self.decode(accepted_tokens)
                logger.info(f"Generated tokens: {decoded_prediction}")
            
            logger.info(f"Accept lengths: {accept_lengths}")
            return
    
    def no_kv_evaluate(self, batch, tokens_to_generate=5, accepted_preds=None):
        """Evaluate without KV caching."""
        if accepted_preds is None:
            input_tokens = batch['tokens'].to(self.device)
            input_prompt = self.format_input(input_tokens)
            logger.info(f'Input prompt: {self.decode(input_prompt)}')
            
            accepted_preds = input_prompt
            seq_len = accepted_preds.shape[1]
            mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device)
            ).unsqueeze(0)
            input_pos = torch.arange(seq_len, device=self.device).unsqueeze(0)
            
            if self.no_kv_model.caches_are_enabled():
                with disable_kv_cache(self.no_kv_model):
                    with self._sdp_math_context():
                        output = self.no_kv_model(accepted_preds, mask=mask, input_pos=input_pos)
            else:
                with self._sdp_math_context():
                    output = self.no_kv_model(accepted_preds, mask=mask, input_pos=input_pos)
            
            base_logits = output[0][:, -1, :]
            next_token = base_logits.argmax(dim=-1, keepdim=True)
            accepted_preds = torch.cat([accepted_preds, next_token], dim=-1)
        else:
            bs = accepted_preds.shape[0]
            seq_len = accepted_preds.shape[1]
            mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device)
            ).unsqueeze(0)
            input_pos = torch.arange(seq_len, device=self.device).unsqueeze(0)
            
            if self.no_kv_model.caches_are_enabled():
                with disable_kv_cache(self.no_kv_model):
                    with self._sdp_math_context():
                        output = self.no_kv_model(accepted_preds, mask=mask, input_pos=input_pos)
            else:
                with self._sdp_math_context():
                    output = self.no_kv_model(accepted_preds, mask=mask, input_pos=input_pos)
            
            base_logits = output[0][:, -1, :]
            next_token = base_logits.argmax(dim=-1, keepdim=True)
            accepted_preds = torch.cat([accepted_preds, next_token], dim=-1)
        
        return base_logits, next_token, accepted_preds
    
    def kv_evaluate(self, batch, tokens_to_generate=5, input_token=None):
        """Evaluate with KV caching."""
        # Find the attention layer
        for module in self.model.modules():
            if isinstance(module, TransformerSelfAttentionLayer):
                layer = module
                break
        else:
            layer = None
        
        model_dtype = next(self.model.parameters()).dtype
        
        if input_token is None:
            self.model.reset_caches()
            
            input_tokens = batch['tokens'].to(self.device)
            input_prompt = self.format_input(input_tokens)
            logger.info(f'Input prompt: {self.decode(input_prompt)}')
            
            bs = input_prompt.shape[0]
            curr_seq_len = input_prompt.shape[1]
            curr_kv_len = 0
            
            # Initial forward pass
            causal_mask = self.create_causal_mask(
                bs, curr_seq_len, curr_kv_len, self.config.get('max_cache_size', 256)
            )
            input_pos = torch.arange(curr_seq_len, device=self.device).unsqueeze(0).expand(bs, -1)
            
            with self._sdp_math_context():
                output = self.model(input_prompt, mask=causal_mask, input_pos=input_pos)
            
            base_logits = output[0][:, -1]
            next_token = base_logits.argmax(dim=-1, keepdim=True)
            curr_kv_len = curr_seq_len
        else:
            bs = input_token.shape[0]
            next_token = input_token
            model_kv_len = layer.attn.kv_cache.size
            curr_kv_len = int(model_kv_len)
            curr_seq_len = 1
            
            causal_mask = self.create_causal_mask(
                bs, curr_seq_len, curr_kv_len, self.config.get('max_cache_size', 256)
            )
            input_pos = torch.full((bs, 1), curr_kv_len, device=self.device)
            
            with self._sdp_math_context():
                output = self.model(next_token, mask=causal_mask, input_pos=input_pos)
            
            base_logits = output[0][:, -1]
            next_token = base_logits.argmax(dim=-1, keepdim=True)
            curr_kv_len += 1
        
        return base_logits, next_token
    
    def compare_predictions(self, dataloader, tokens_to_generate=100):
        """Compare predictions between KV and non-KV models."""
        if not self.config.get('debug', False):
            logger.error("Debug mode required for comparison")
            return
        
        prediction_consistencies = []
        i = 0
        
        for batch in dataloader:
            tokens_generated = 0
            accepted_preds = None
            next_token_kv = None
            
            while tokens_generated < tokens_to_generate:
                base_logits_no_kv, next_token_no_kv, accepted_preds = self.no_kv_evaluate(
                    batch, tokens_to_generate, accepted_preds
                )
                base_logits_kv, next_token_kv = self.kv_evaluate(
                    batch, tokens_to_generate, next_token_kv
                )
                tokens_generated += 1
                
                if next_token_no_kv != next_token_kv:
                    # Calculate KL divergence
                    p_log = F.log_softmax(base_logits_no_kv, dim=-1)
                    q_log = F.log_softmax(base_logits_kv, dim=-1)
                    kl = torch.sum(torch.exp(p_log) * (p_log - q_log))
                    
                    # Get top-k predictions
                    kval, kindices = torch.topk(base_logits_kv, k=4)
                    nokval, nokindices = torch.topk(base_logits_no_kv, k=4)
                    
                    logger.info(f"Prediction divergence at token {tokens_generated}")
                    logger.info(f"KV token: {next_token_kv.item()}, No-KV token: {next_token_no_kv.item()}")
                    logger.info(f"KL divergence: {kl.item()}")
                    
                    prediction_consistencies.append(tokens_generated)
                    break
            
            i += 1
            if i >= 20:
                logger.info(f"Prediction consistencies: {prediction_consistencies}")
                break


def main():
    """Main execution function."""
    config = {
        'device': 'cuda:1',
        'max_cache_size': 256,
        'seed': 42,
        'debug': True,
        'tokens_to_generate': 100,
        'tokenizer_dir': '/home/ubuntu/.llama/checkpoints/Llama3.1-8B-Instruct/tokenizer.model',
        'checkpoint_dir': '/home/ubuntu/vanshaj/inf2-training/3rdparty/torchtune/medusa_checkpoints/epoch_0/',
        'dataset_dir': '/home/ubuntu/vanshaj/justpi.jsonl',
        'batch_size': 1
    }
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Initialize evaluator
    evaluator = MedusaEvaluator(config)
    
    # Load data and model
    dataloader = evaluator.load_data(
        config['dataset_dir'], 
        config['tokenizer_dir'], 
        config['batch_size']
    )
    model = evaluator.load_model(config['checkpoint_dir'])
    
    if config.get('debug', False):
        evaluator.compare_predictions(dataloader, config['tokens_to_generate'])
    else:
        evaluator.evaluate_medusa(dataloader, config['tokens_to_generate'])


if __name__ == "__main__":
    main()
