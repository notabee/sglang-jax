import os
import sys

# Add python dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))
# Add sglang dir to path
sys.path.append("/usr/local/google/home/rishabhbaghel/sglang/python")

import jax
import jax.numpy as jnp
import numpy as np
import safetensors.numpy as st_np
from flax import nnx

from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend
from sgl_jax.srt.models.glm5_moe import Glm5DecoderLayer
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


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


def test_moe_with_random_weights():
    print("Verifying MoE block with random weights...")

    config = MockConfig()
    mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])

    try:
        with jax.set_mesh(mesh):
            # Instantiate layer 3 which should be an MoE layer!
            layer = Glm5DecoderLayer(
                config=config,
                mesh=mesh,
                layer_id=3,
                dtype=jnp.bfloat16,
            )
            print("Successfully instantiated Glm5DecoderLayer (MoE)!")

        # Generate random inputs
        batch_size = 2
        seq_len = 10
        hidden_size = 6144
        hidden_states = jnp.ones((batch_size * seq_len, hidden_size), dtype=jnp.bfloat16)
        positions = jnp.arange(seq_len, dtype=jnp.int32)
        positions = jnp.tile(positions, batch_size)

        # Create real MLAAttentionBackend for the attention part of the layer
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

        print("Running forward pass on MoE layer...")
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
            print(f"Topk IDs shape: {topk_ids.shape}")

    except Exception as e:
        print(f"Failed during verification: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_moe_with_random_weights()
