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
from sgl_jax.srt.models.glm5_moe import Glm5DecoderLayer
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend
import safetensors.numpy as st_np

class MockConfig:
    def __init__(self):
        self.hidden_size = 6144
        self.rms_norm_eps = 1e-5
        self.num_attention_heads = 64
        self.num_key_value_heads = 64
        self.first_k_dense_replace = 3
        self.n_routed_experts = 256
        self.num_experts_per_tok = 8
        self.norm_topk_prob = True
        self.moe_intermediate_size = 2048
        self.scoring_func = "sigmoid"
        self.topk_group = 1
        self.n_group = 1
        self.routed_scaling_factor = 2.5 # Use the real scaling factor from config.json!
        self.moe_backend = "epmoe"
        self.quantization_config = None
        self.ep_size = 1

def test_routing_with_real_weights():
    print("Verifying Routing with real weights...")
    
    # Load expert 0 weights and gate weights
    file75_path = "/local/GLM-5.1/model-00075-of-00282.safetensors"
    file79_path = "/local/GLM-5.1/model-00079-of-00282.safetensors"
    
    if not os.path.exists(file75_path) or not os.path.exists(file79_path):
        print("Required weights files not found. Please run on the pod.")
        return
        
    print("Loading weights...")
    weights75 = st_np.load_file(file75_path)
    weights79 = st_np.load_file(file79_path)
    
    # Filter weights for layer 3 expert 0
    prefix_exp = "model.layers.3.mlp.experts.0."
    expert_weights = {k[len(prefix_exp):]: v for k, v in weights75.items() if k.startswith(prefix_exp)}
    print(f"Found {len(expert_weights)} weights for Expert 0.")
    
    # Filter weights for layer 3 gate
    prefix_gate = "model.layers.3.mlp.gate."
    gate_weights = {k[len(prefix_gate):]: v for k, v in weights79.items() if k.startswith(prefix_gate)}
    print(f"Found {len(gate_weights)} weights for Gate.")
    
    config = MockConfig()
    mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
    
    try:
        with jax.set_mesh(mesh):
            layer = Glm5DecoderLayer(
                config=config,
                mesh=mesh,
                layer_id=3,
                dtype=jnp.bfloat16,
            )
            print("Successfully instantiated Glm5DecoderLayer (MoE)!")
            
            # Assign Expert 0 weights
            print("Assigning Expert 0 weights...")
            def assign_expert_weight(param, torch_tensor, expert_idx, transpose=True):
                val = torch_tensor
                if transpose:
                    val = val.T
                from jax.sharding import PartitionSpec as P
                if param.shape[1] == 6144:
                    out_sharding = P("expert", None, "tensor")
                else:
                    out_sharding = P("expert", "tensor", None)
                    
                with jax.set_mesh(layer.mlp.moe_mesh):
                    param.value = param.value.at[expert_idx].set(
                        jnp.asarray(val, dtype=jnp.bfloat16),
                        out_sharding=out_sharding
                    )

            assign_expert_weight(layer.mlp.wi_0, expert_weights["gate_proj.weight"], 0, transpose=True)
            assign_expert_weight(layer.mlp.wi_1, expert_weights["up_proj.weight"], 0, transpose=True)
            assign_expert_weight(layer.mlp.wo, expert_weights["down_proj.weight"], 0, transpose=True)
            
            # Assign Gate weights
            print("Assigning Gate weights...")
            # gate.weight is [256, 6144] -> transpose to [6144, 256]
            layer.moe_gate.kernel.value = jnp.asarray(gate_weights["weight"].T, dtype=jnp.float32)
            layer.moe_gate.bias.value = jnp.asarray(gate_weights["e_score_correction_bias"], dtype=jnp.bfloat16)
            
            print("Weights assigned successfully!")

        # Generate random inputs
        batch_size = 2
        seq_len = 10
        hidden_size = 6144
        hidden_states = jnp.ones((batch_size * seq_len, hidden_size), dtype=jnp.bfloat16)
        positions = jnp.arange(seq_len, dtype=jnp.int32)
        positions = jnp.tile(positions, batch_size)
        
        # Create real MLAAttentionBackend
        attn_backend = MLAAttentionBackend(
            num_attn_heads=64,
            kv_lora_rank=512,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=256,
            page_size=1,
            mesh=mesh,
        )
        
        class DummyMetadata:
            def __init__(self):
                self.cu_q_lens = jnp.array([0, 10, 20], dtype=jnp.int32)
                self.cu_kv_lens = jnp.array([0, 10, 20], dtype=jnp.int32)
                self.page_indices = jnp.arange(20, dtype=jnp.int32)
                self.seq_lens = jnp.array([10, 10], dtype=jnp.int32)
                self.distribution = jnp.array([0, 0, 2], dtype=jnp.int32)
                self.custom_mask = None
                
        attn_backend.forward_metadata = DummyMetadata()

        class DummyForwardBatch:
            def __init__(self):
                self.attn_backend = attn_backend
                self.expert_location_metadata = None
                
            def get_token_valid_mask(self, num_tokens):
                return jnp.ones((num_tokens,), dtype=jnp.bool_)
                
        forward_batch = DummyForwardBatch()
        
        class DummyKVCache:
            def get_fused_kv_buffer(self, layer_id):
                return jnp.zeros((20, 1, 2, 640), dtype=jnp.bfloat16)
                
        token_to_kv_pool = DummyKVCache()
        
        print("Running forward pass on full layer to check routing...")
        with jax.set_mesh(mesh):
            output, residual, kv_fused, topk_ids = layer(
                positions,
                hidden_states,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
            )
            
        print("Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Any NaNs in output: {jnp.isnan(output).any()}")
        if topk_ids is not None:
            print(f"Selected Topk IDs for first 5 tokens:\n{topk_ids[:5]}")
            print(f"Unique experts selected: {np.unique(np.array(topk_ids))}")
            
    except Exception as e:
        print(f"Failed during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_routing_with_real_weights()
