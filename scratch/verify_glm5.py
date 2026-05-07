import sys
import os

# Add python dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from sgl_jax.srt.models.glm5_moe import Glm5Attention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.mem_cache.memory_pool import KVCache

def verify_attention():
    print("Verifying Glm5Attention...")
    
    # GLM-5.1 Config values
    hidden_size = 6144
    num_heads = 64
    num_kv_heads = 64
    max_position_embeddings = 202752
    rope_theta = 1000000
    head_dim = 64
    rms_norm_eps = 1e-5
    
    # Create a mesh for testing (dummy mesh)
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices).reshape(1, -1), axis_names=("data", "tensor"))
    
    # Instantiate Attention
    try:
        with jax.set_mesh(mesh):
            attn = Glm5Attention(

                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                max_position_embeddings=max_position_embeddings,
                mesh=mesh,
                rope_theta=rope_theta,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                layer_id=0,
                dtype=jnp.bfloat16,
            )
        print("Successfully instantiated Glm5Attention!")

    except Exception as e:
        print(f"Failed to instantiate Glm5Attention: {e}")
        return

    # Generate random inputs
    batch_size = 2
    seq_len = 10
    hidden_states = jnp.ones((batch_size * seq_len, hidden_size), dtype=jnp.bfloat16)
    positions = jnp.arange(seq_len, dtype=jnp.int32)
    positions = jnp.tile(positions, batch_size)
    
    # Create a dummy ForwardBatch and KVCache
    # These might need real objects depending on how RadixAttention uses them.
    # For now, let's pass None or dummy if allowed.
    # In sglang-jax, forward_batch and token_to_kv_pool are usually complex.
    
    print("Running forward pass...")
    try:
        # We might need to mock or create real ForwardBatch/KVCache objects.
        # Let's try to run it and see what happens.
        # output, kv_fused = attn(positions, hidden_states, forward_batch=None, token_to_kv_pool=None)
        print("Forward pass requires valid forward_batch and token_to_kv_pool. Skipping run until we can mock them.")
        # TODO: Add mocking for ForwardBatch and KVCache
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    verify_attention()
