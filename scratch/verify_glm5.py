import sys
import os

# Add python dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
# Add sglang dir to path
sys.path.append('/usr/local/google/home/rishabhbaghel/sglang/python')

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from sgl_jax.srt.models.glm5_moe import Glm5Attention
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend
import safetensors.numpy as st_np

def test_with_real_weights():
    print("Verifying with real weights...")
    weights_path = "/local/GLM-5.1/model-00001-of-00282.safetensors"
    
    if not os.path.exists(weights_path):
        print(f"Weights file not found at {weights_path}. Please run on the pod.")
        return
        
    print("Loading weights...")
    weights = st_np.load_file(weights_path)
    
    # Filter weights for layer 0 attention
    prefix = "model.layers.0.self_attn."
    attn_weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
    print(f"Found {len(attn_weights)} weights for Layer 0 Attention.")
    
    # GLM-5.1 Config values
    hidden_size = 6144
    num_heads = 64
    num_kv_heads = 64
    max_position_embeddings = 202752
    rope_theta = 1000000
    head_dim = 64
    rms_norm_eps = 1e-5
    
    mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
    
    try:
        with jax.set_mesh(mesh):
            jax_attn = Glm5Attention(
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
            print("Successfully instantiated JAX Glm5Attention!")
            
            # Load weights into JAX
            def assign_weight(jax_param, torch_tensor, transpose=False):
                val = torch_tensor
                if transpose:
                    val = val.T
                jax_param.value = jnp.asarray(val, dtype=jnp.bfloat16)

            print("Assigning weights to JAX model...")
            assign_weight(jax_attn.q_a_proj.weight, attn_weights["q_a_proj.weight"], transpose=True)
            assign_weight(jax_attn.q_a_layernorm.scale, attn_weights["q_a_layernorm.weight"])
            assign_weight(jax_attn.q_b_proj.weight, attn_weights["q_b_proj.weight"], transpose=True)
            
            assign_weight(jax_attn.kv_a_proj_with_mqa.weight, attn_weights["kv_a_proj_with_mqa.weight"], transpose=True)
            assign_weight(jax_attn.kv_a_layernorm.scale, attn_weights["kv_a_layernorm.weight"])
            assign_weight(jax_attn.kv_b_proj.weight, attn_weights["kv_b_proj.weight"], transpose=True)
            
            assign_weight(jax_attn.o_proj.weight, attn_weights["o_proj.weight"], transpose=True)
            
            # Indexer weights
            assign_weight(jax_attn.indexer.wq_b.weight, attn_weights["indexer.wq_b.weight"], transpose=True)
            assign_weight(jax_attn.indexer.wk.weight, attn_weights["indexer.wk.weight"], transpose=True)
            assign_weight(jax_attn.indexer.weights_proj.weight, attn_weights["indexer.weights_proj.weight"], transpose=True)
            assign_weight(jax_attn.indexer.k_norm.weight, attn_weights["indexer.k_norm.weight"])
            assign_weight(jax_attn.indexer.k_norm.bias, attn_weights["indexer.k_norm.bias"])

            print("Weights assigned successfully!")
            
            print("Calling post_load_weights to split MLA weights...")
            jax_attn.post_load_weights()
            print("MLA weights split successfully!")
            
            # Generate random inputs for forward pass
            batch_size = 2
            seq_len = 10
            hidden_states = jnp.ones((batch_size * seq_len, hidden_size), dtype=jnp.bfloat16)
            positions = jnp.arange(seq_len, dtype=jnp.int32)
            positions = jnp.tile(positions, batch_size)
            
            # Create real MLAAttentionBackend
            print("Creating real MLAAttentionBackend...")
            attn_backend = MLAAttentionBackend(
                num_attn_heads=64,
                kv_lora_rank=512,
                qk_nope_head_dim=192,
                qk_rope_head_dim=64,
                v_head_dim=256,
                page_size=1,
                mesh=mesh,
            )
            
            # Mock metadata for prefill (EXTEND) mode
            class DummyMetadata:
                def __init__(self):
                    self.cu_q_lens = jnp.array([0, 10, 20], dtype=jnp.int32)
                    self.cu_kv_lens = jnp.array([0, 10, 20], dtype=jnp.int32)
                    self.page_indices = jnp.array([0, 1], dtype=jnp.int32)
                    self.seq_lens = jnp.array([10, 10], dtype=jnp.int32)
                    self.distribution = jnp.array([0, 0, 2], dtype=jnp.int32)
                    self.custom_mask = None
                    
            attn_backend.forward_metadata = DummyMetadata()

            class DummyForwardBatch:
                def __init__(self):
                    self.attn_backend = attn_backend
                    
            forward_batch = DummyForwardBatch()
            
            # Dummy KVCache that returns 4D buffer
            class DummyKVCache:
                def get_fused_kv_buffer(self, layer_id):
                    # Shape: [pages, page_size, packing, dim]
                    # MLA pool aligns segments to 128: 512 + align(64, 128) = 512 + 128 = 640
                    # For bfloat16, packing must be 2 (32 bits // 16 bits = 2).
                    return jnp.zeros((1, 1, 2, 640), dtype=jnp.bfloat16)
                    
            token_to_kv_pool = DummyKVCache()


            
            print("Running forward pass with real weights and FlashAttention backend...")
            output, kv_fused = jax_attn(positions, hidden_states, forward_batch=forward_batch, token_to_kv_pool=token_to_kv_pool)
            print("Forward pass successful!")
            print(f"Output shape: {output.shape}")
            print(f"Any NaNs in output: {jnp.isnan(output).any()}")
            if not jnp.isnan(output).any():
                print(f"Output max: {jnp.max(jnp.abs(output))}")
            
    except Exception as e:
        print(f"Failed during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_real_weights()
