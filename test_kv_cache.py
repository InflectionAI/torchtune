#!/usr/bin/env python3
"""
Test script for KVCache class
"""

import torch
import sys
import os

# Add the torchtune directory to the path so we can import the module
sys.path.append(os.path.join(os.path.dirname(__file__), 'torchtune'))

from torchtune.modules.kv_cache import KVCache

def test_kv_cache_basic():
    """Basic functionality test"""
    print("=== Basic KVCache Test ===")
    
    # Create a cache
    batch_size = 2
    max_seq_len = 10
    num_kv_heads = 4
    head_dim = 32
    dtype = torch.float32
    
    cache = KVCache(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype
    )
    
    print(f"Initial cache size: {cache.size}")
    print(f"Cache shapes - k: {cache.k_cache.shape}, v: {cache.v_cache.shape}")
    print(f"Cache position tensor: {cache.cache_pos}")
    
    # Test first update
    seq_len = 3
    k_val = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype)
    v_val = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype)
    print("$$ before update cache.cache_pos:", cache.cache_pos)
    print(f"\nUpdating with {seq_len} tokens...")
    k_out, v_out = cache.update(k_val, v_val)
    print("$$ after update cache.cache_pos:", cache.cache_pos)
    
    print(f"Cache size after update: {cache.size}")
    print(f"Cache position tensor: {cache.cache_pos}")
    
    # Test second update
    seq_len2 = 2
    k_val2 = torch.randn(batch_size, num_kv_heads, seq_len2, head_dim, dtype=dtype)
    v_val2 = torch.randn(batch_size, num_kv_heads, seq_len2, head_dim, dtype=dtype)
    
    print(f"\nUpdating with {seq_len2} more tokens...")
    k_out2, v_out2 = cache.update(k_val2, v_val2)
    
    print(f"Cache size after second update: {cache.size}")
    print(f"Cache position tensor: {cache.cache_pos}")
    
    # Test reset
    print(f"\nResetting cache...")
    cache.reset()
    print(f"Cache size after reset: {cache.size}")
    print(f"Cache position tensor: {cache.cache_pos}")

def test_kv_cache_edge_cases():
    """Test edge cases and error conditions"""
    print("\n=== Edge Cases Test ===")
    
    cache = KVCache(
        batch_size=1,
        max_seq_len=5,
        num_kv_heads=2,
        head_dim=16,
        dtype=torch.float32
    )
    
    # Test filling the cache completely
    print("Filling cache completely...")
    for i in range(5):
        k_val = torch.ones(1, 2, 1, 16, dtype=torch.float32) * (i + 1)
        v_val = torch.ones(1, 2, 1, 16, dtype=torch.float32) * (i + 1)
        cache.update(k_val, v_val)
        print(f"Step {i+1}: cache size = {cache.size}")
    
    # Test that we can't exceed max_seq_len
    print("\nTrying to exceed max_seq_len...")
    try:
        k_val = torch.ones(1, 2, 1, 16, dtype=torch.float32)
        v_val = torch.ones(1, 2, 1, 16, dtype=torch.float32)
        cache.update(k_val, v_val)
        print("ERROR: Should have failed!")
    except AssertionError:
        print("✓ Correctly caught assertion error for exceeding max_seq_len")
    
    # Test batch size mismatch
    print("\nTesting batch size mismatch...")
    cache = KVCache(batch_size=1, max_seq_len=5, num_kv_heads=2, head_dim=16, dtype=torch.float32)
    try:
        k_val = torch.ones(2, 2, 1, 16, dtype=torch.float32)  # batch_size=2
        v_val = torch.ones(2, 2, 1, 16, dtype=torch.float32)
        cache.update(k_val, v_val)
        print("ERROR: Should have failed!")
    except ValueError as e:
        print(f"✓ Correctly caught ValueError: {e}")

def test_kv_cache_state_dict():
    """Test state dict functionality"""
    print("\n=== State Dict Test ===")
    
    cache = KVCache(
        batch_size=2,
        max_seq_len=8,
        num_kv_heads=3,
        head_dim=24,
        dtype=torch.float32
    )
    
    # Update cache
    k_val = torch.randn(2, 3, 2, 24, dtype=torch.float32)
    v_val = torch.randn(2, 3, 2, 24, dtype=torch.float32)
    cache.update(k_val, v_val)
    
    print(f"Cache size before save: {cache.size}")
    
    # Save state dict
    state_dict = cache.state_dict()
    print(f"State dict keys: {list(state_dict.keys())}")
    
    # Create new cache and load state
    new_cache = KVCache(
        batch_size=2,
        max_seq_len=8,
        num_kv_heads=3,
        head_dim=24,
        dtype=torch.float32
    )
    
    new_cache.load_state_dict(state_dict)
    print(f"New cache size after load: {new_cache.size}")
    
    # Verify they're the same
    print(f"Original cache pos: {cache.cache_pos}")
    print(f"New cache pos: {new_cache.cache_pos}")
    print(f"Caches are identical: {torch.allclose(cache.k_cache, new_cache.k_cache)}")

if __name__ == "__main__":
    test_kv_cache_basic()
    # test_kv_cache_edge_cases()
    # test_kv_cache_state_dict()
    print("\n=== All tests completed ===") 