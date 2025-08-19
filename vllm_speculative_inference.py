#!/usr/bin/env python3
"""
vLLM High-Level API with Speculative Decoding (Medusa)

This script uses vLLM's LLM wrapper with a properly constructed
speculative_config to integrate a Medusa checkpoint with Llama 3.1 8B Instruct.
"""

import argparse
import json
import os
from typing import List, Dict, Any
from vllm import LLM, SamplingParams


def load_vllm_config(config_path: str) -> Dict[str, Any]:
    """Load the vLLM Medusa configuration JSON."""
    print(f"📂 Loading vLLM config from: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"✅ Config loaded:\n{json.dumps(config, indent=2)}")
    return config


def create_speculative_config(medusa_checkpoint: str, medusa_config: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Create the speculative configuration for vLLM (matching current SpeculativeConfig schema)."""
    n_predict = int(config.get("n_predict", 5))
    
    # Use the directory containing config.json and checkpoint
    medusa_model_dir = "./vllm_medusa_model"  # Use our created directory with config.json

    speculative_config = {
        "model": medusa_model_dir,  # This should point to a directory with config.json
        "num_speculative_tokens": n_predict,
    }

    print("\n⚙️ Final speculative config passed to LLM:")
    print(json.dumps(speculative_config, indent=2))
    return speculative_config




def run_speculative_inference(llm: LLM, prompts: List[str], sampling_params: SamplingParams) -> List[Dict[str, Any]]:
    """Run inference with speculative decoding."""
    print(f"\n🚀 Running speculative inference on {len(prompts)} prompts...")
    results = []

    try:
        # Try to access engine stats before generation
        engine_stats_before = None
        if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'get_stats'):
            try:
                engine_stats_before = llm.llm_engine.get_stats()
            except:
                pass

        outputs = llm.generate(prompts, sampling_params)
        
        # Try to access engine stats after generation
        engine_stats_after = None
        if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'get_stats'):
            try:
                engine_stats_after = llm.llm_engine.get_stats()
            except:
                pass
        for i, output in enumerate(outputs):
            # Debug: print all available attributes of the output object
            print(f"\n🔍 Debug: Available attributes on output object:")
            print(f"   dir(output): {[attr for attr in dir(output) if not attr.startswith('_')]}")
            
            # Count tokens more accurately
            completion_tokens = 0
            if output.outputs[0].logprobs:
                completion_tokens = len(output.outputs[0].logprobs)
            elif output.outputs[0].text:
                # Fallback: rough estimate based on text length
                completion_tokens = len(output.outputs[0].text.split())
            
            result = {
                "prompt": prompts[i],
                "generated_text": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason,
                "usage": {
                    "completion_tokens": completion_tokens
                }
            }
            
            # Check for speculative stats in various possible locations
            speculative_stats = None
            if hasattr(output, "speculative_stats"):
                speculative_stats = output.speculative_stats
                print(f"   Found speculative_stats: {speculative_stats}")
            elif hasattr(output, "speculative"):
                speculative_stats = output.speculative
                print(f"   Found speculative: {speculative_stats}")
            elif hasattr(output, "stats"):
                speculative_stats = output.stats
                print(f"   Found stats: {speculative_stats}")
            
            if speculative_stats:
                result["speculative_stats"] = speculative_stats

            results.append(result)

            print(f"\n📝 Prompt {i+1}:")
            print(f"   Input: {prompts[i]}")
            print(f"   Output: {output.outputs[0].text}")
            print(f"   Finish reason: {output.outputs[0].finish_reason}")
            print(f"   Tokens generated: {result['usage']['completion_tokens']}")
            
            # Print speculative decoding statistics if available
            if hasattr(output, "speculative_stats") and output.speculative_stats:
                print(f"\n🔮 Speculative Decoding Stats:")
                stats = output.speculative_stats
                
                # Access specific speculative stats
                num_accepted = stats.get("num_accepted_tokens", 0)
                num_proposed = stats.get("num_proposed_tokens", 0)
                
                print(f"   num_accepted_tokens: {num_accepted}")
                print(f"   num_proposed_tokens: {num_proposed}")
                
                if num_proposed > 0:
                    accepted_ratio = num_accepted / num_proposed
                    print(f"   Acceptance ratio: {accepted_ratio:.2%}")
                
                # Print any other stats
                for key, value in stats.items():
                    if key not in ["num_accepted_tokens", "num_proposed_tokens"]:
                        print(f"   {key}: {value}")
                        
            elif "speculative_stats" in result:
                print(f"\n🔮 Speculative Decoding Stats:")
                stats = result["speculative_stats"]
                
                # Access specific speculative stats
                num_accepted = stats.get("num_accepted_tokens", 0)
                num_proposed = stats.get("num_proposed_tokens", 0)
                
                print(f"   num_accepted_tokens: {num_accepted}")
                print(f"   num_proposed_tokens: {num_proposed}")
                
                if num_proposed > 0:
                    accepted_ratio = num_accepted / num_proposed
                    print(f"   Acceptance ratio: {accepted_ratio:.2%}")
                
                # Print any other stats
                for key, value in stats.items():
                    if key not in ["num_accepted_tokens", "num_proposed_tokens"]:
                        print(f"   {key}: {value}")
        
        # Print engine-level stats if available
        if engine_stats_after:
            print(f"\n🔧 Engine Statistics:")
            for key, value in engine_stats_after.items():
                if "spec" in key.lower() or "draft" in key.lower() or "accept" in key.lower():
                    print(f"   {key}: {value}")
    except Exception as e:
        import traceback
        print(f"❌ Error during speculative inference: {e}")
        traceback.print_exc()

    return results


def print_speculative_summary(results: List[Dict[str, Any]]):
    """Print a summary of speculative decoding performance."""
    print(f"\n" + "="*60)
    print("🔮 SPECULATIVE DECODING PERFORMANCE SUMMARY")
    print("="*60)
    
    total_tokens = 0
    total_accepted = 0
    total_proposed = 0
    
    for i, result in enumerate(results):
        if "speculative_stats" in result and result["speculative_stats"]:
            stats = result["speculative_stats"]
            print(f"\n📊 Prompt {i+1} Speculative Stats:")
            
            # Extract specific speculative stats
            num_accepted = stats.get("num_accepted_tokens", 0)
            num_proposed = stats.get("num_proposed_tokens", 0)
            
            print(f"   num_accepted_tokens: {num_accepted}")
            print(f"   num_proposed_tokens: {num_proposed}")
            
            if num_proposed > 0:
                accepted_ratio = num_accepted / num_proposed
                print(f"   Acceptance ratio: {accepted_ratio:.2%}")
            
            # Print any other stats
            for key, value in stats.items():
                if key not in ["num_accepted_tokens", "num_proposed_tokens"]:
                    print(f"   {key}: {value}")
            
            # Accumulate totals
            total_accepted += num_accepted
            total_proposed += num_proposed
            total_tokens += result["usage"]["completion_tokens"]
    
    print(f"\n📈 Overall Performance:")
    print(f"   Total tokens generated: {total_tokens}")
    if total_proposed > 0:
        acceptance_rate = (total_accepted / total_proposed) * 100
        print(f"   Total tokens proposed: {total_proposed}")
        print(f"   Total tokens accepted: {total_accepted}")
        print(f"   Acceptance rate: {acceptance_rate:.1f}%")
        print(f"   Speedup factor: {total_proposed / total_tokens:.2f}x")
    else:
        print("   No detailed speculative stats available")


def compare_with_base_model(base_model_path: str, prompts: List[str], sampling_params: SamplingParams) -> List[Dict[str, Any]]:
    """Run inference with just the base model for comparison."""
    print(f"\n🔍 Running base model inference for comparison...")
    results = []

    try:
        base_llm = LLM(model=base_model_path)
        outputs = base_llm.generate(prompts, sampling_params)

        for i, output in enumerate(outputs):
            # Count tokens more accurately
            completion_tokens = 0
            if output.outputs[0].logprobs:
                completion_tokens = len(output.outputs[0].logprobs)
            elif output.outputs[0].text:
                # Fallback: rough estimate based on text length
                completion_tokens = len(output.outputs[0].text.split())
            
            result = {
                "prompt": prompts[i],
                "generated_text": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason,
                "usage": {
                    "completion_tokens": completion_tokens
                }
            }
            results.append(result)

            print(f"\n📝 Base Model Prompt {i+1}:")
            print(f"   Input: {prompts[i]}")
            print(f"   Output: {output.outputs[0].text}")
            print(f"   Finish reason: {output.outputs[0].finish_reason}")
            print(f"   Tokens generated: {result['usage']['completion_tokens']}")
    except Exception as e:
        import traceback
        print(f"❌ Error during base model inference: {e}")
        traceback.print_exc()

    return results


def main():
    parser = argparse.ArgumentParser(description="vLLM High-Level API with Speculative Decoding (Medusa)")
    parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                       help="Base model path (HuggingFace model or local path)")
    parser.add_argument("--medusa_checkpoint", default="./vllm_medusa_heads_final.pt",
                       help="Path to converted vLLM Medusa checkpoint")
    parser.add_argument("--medusa_config", default="./vllm_medusa_config_final.json",
                       help="Path to vLLM Medusa configuration file")
    parser.add_argument("--prompts", nargs="+", 
                       default=["The quick brown fox jumps over the lazy dog. The weather today is"],
                       help="Prompts to test")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--compare_base", action="store_true", help="Also run base model inference for comparison")

    args = parser.parse_args()

    try:
        print("="*80)
        print("🌟 vLLM High-Level API with Speculative Decoding (Medusa)")
        print("="*80)

        # Check required files
        if not os.path.exists(args.medusa_checkpoint):
            print(f"❌ Medusa checkpoint not found: {args.medusa_checkpoint}")
            return 1
        if not os.path.exists(args.medusa_config):
            print(f"❌ Medusa config not found: {args.medusa_config}")
            return 1

        # Load Medusa config
        medusa_config = load_vllm_config(args.medusa_config)

        # Build speculative config
        speculative_config = create_speculative_config(args.medusa_checkpoint, args.medusa_config, medusa_config)

                 # Sampling params
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop=["\n", ".", "!", "?"],
            logprobs=1,  # Request logprobs to get accurate token counts
            include_stop_str_in_output=True  # Include stop strings to see full generation
        )
        print(f"\n🎯 Sampling parameters: {sampling_params}")

        # Initialize vLLM
        print(f"\n🚀 Initializing vLLM with speculative decoding...")
        llm = LLM(
                model=args.base_model,
                speculative_config=speculative_config,
                trust_remote_code=True,
                max_model_len=4096,
                disable_log_stats=False  # Enable statistics logging
                )

        print("✅ vLLM initialized successfully!")

        # Run speculative inference
        speculative_results = run_speculative_inference(llm, args.prompts, sampling_params)
        if not speculative_results:
            print("❌ No results from speculative inference")
            return 1
        
        # Print speculative decoding performance summary
        print_speculative_summary(speculative_results)

        # Compare with base model if requested
        if args.compare_base:
            print("\n" + "="*60)
            print("COMPARISON WITH BASE MODEL")
            print("="*60)
            base_results = compare_with_base_model(args.base_model, args.prompts, sampling_params)

            if base_results:
                print(f"\n📊 Comparison Summary:")
                for i, prompt in enumerate(args.prompts):
                    print(f"\n   Prompt {i+1}: {prompt[:50]}...")
                    print(f"   Speculative: {speculative_results[i]['generated_text'][:100]}...")
                    print(f"   Base Model:  {base_results[i]['generated_text'][:100]}...")

        print(f"\n🎉 Speculative decoding completed successfully!")
        print(f"   Total prompts processed: {len(args.prompts)}")
        print(f"   Medusa checkpoint: {args.medusa_checkpoint}")
        print(f"   Base model: {args.base_model}")
        return 0
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
