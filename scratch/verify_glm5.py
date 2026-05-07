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
    
    # Create a mesh for testing
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh
    mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])

    
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
    
    # Create a dummy ForwardBatch
    class DummyAttnBackend:
        def __call__(self, *args, **kwargs):
            # Return dummy attention output and kv_fused
            # Shape: [total_tokens, num_heads, head_dim] -> [20, 64, 256]
            return jnp.zeros((20, 64, 256), dtype=jnp.bfloat16), None

    class DummyForwardBatch:
        def __init__(self):
            self.attn_backend = DummyAttnBackend()
            
    forward_batch = DummyForwardBatch()
    token_to_kv_pool = None 
    
    print("Running forward pass...")
    try:
        with jax.set_mesh(mesh):
            output, kv_fused = attn(positions, hidden_states, forward_batch=forward_batch, token_to_kv_pool=token_to_kv_pool)
        print("Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Any NaNs in output: {jnp.isnan(output).any()}")
    except Exception as e:
        print(f"Forward pass failed: {e}")


if __name__ == "__main__":
    verify_attention()
