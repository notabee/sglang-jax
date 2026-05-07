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
    
    print("Keys in checkpoint:")
    for k in sorted(attn_weights.keys()):
        print(f"  {k}: {attn_weights[k].shape}")

    # GLM-5.1 Config values
    hidden_size = 6144
    num_heads = 64
    num_kv_heads = 64
    max_position_embeddings = 202752
    rope_theta = 1000000
    head_dim = 64
    rms_norm_eps = 1e-5
    
    mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
    
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
        
    print("\nSuccessfully instantiated JAX Glm5Attention!")
    
    # TODO: Once we see the keys, we will add the manual assignment here
    # e.g., jax_attn.q_a_proj.weight.value = ...
    
    print("\nPlease run this script on the pod to see the exact keys and shapes in the checkpoint.")

if __name__ == "__main__":
    test_with_real_weights()
