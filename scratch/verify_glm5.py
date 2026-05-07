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
        self.routed_scaling_factor = 1.0
        self.moe_backend = "epmoe"
        self.quantization_config = None
        self.ep_size = 1

def test_moe_with_real_weights():
    print("Verifying MoE block with real weights for Expert 0...")
    
    weights_path = "/local/GLM-5.1/model-00075-of-00282.safetensors"
    if not os.path.exists(weights_path):
        print(f"Weights file not found at {weights_path}. Please run on the pod.")
        return
        
    print("Loading weights from file 75...")
    weights = st_np.load_file(weights_path)
    
    # Filter weights for layer 3 expert 0
    prefix = "model.layers.3.mlp.experts.0."
    expert_weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
    print(f"Found {len(expert_weights)} weights for Expert 0.")
    
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
            
            # Assign weights to Expert 0 in layer.mlp (which is EPMoE)
            print("Assigning Expert 0 weights...")
            
            def assign_expert_weight(param, torch_tensor, expert_idx, transpose=True):
                val = torch_tensor
                if transpose:
                    val = val.T
                with jax.set_mesh(layer.mlp.moe_mesh):
                    param.value = param.value.at[expert_idx].set(jnp.asarray(val, dtype=jnp.bfloat16))


            assign_expert_weight(layer.mlp.wi_0, expert_weights["gate_proj.weight"], 0, transpose=True)
            assign_expert_weight(layer.mlp.wi_1, expert_weights["up_proj.weight"], 0, transpose=True)
            assign_expert_weight(layer.mlp.wo, expert_weights["down_proj.weight"], 0, transpose=True)
            
            print("Expert 0 weights assigned successfully!")

        # Generate random inputs
        batch_size = 2
        seq_len = 10
        hidden_size = 6144
        hidden_states = jnp.ones((batch_size * seq_len, hidden_size), dtype=jnp.bfloat16)
        
        # Force expert 0
        topk_ids = jnp.zeros((20, 8), dtype=jnp.int32)
        topk_weights = jnp.ones((20, 8), dtype=jnp.bfloat16) / 8.0
        
        print("Running forward pass on MoE block (Expert 0 only)...")
        with jax.set_mesh(mesh):
            output = layer.mlp(hidden_states, topk_weights, topk_ids)
            
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
    test_moe_with_real_weights()
