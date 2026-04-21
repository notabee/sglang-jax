import jax
import jax.numpy as jnp
import sys
import os

# Add the project root to python path to find sgl_jax
# File is at python/sgl_jax/srt/kernels/gmm/reproduce_gmm.py
# Project root is 4 levels up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 import gmm_v2


def main():
    print("Starting reproducer...")

    hidden_size = 5120
    intermediate_dim = 1536
    num_experts = 2
    total_tokens = 16

    # lhs: [m, k] -> [total_tokens, hidden_size]
    lhs = jax.random.normal(
        jax.random.PRNGKey(0), (total_tokens, hidden_size), dtype=jnp.bfloat16
    )

    # rhs: [num_groups, k, n] -> [num_experts, hidden_size, intermediate_dim]
    rhs = jax.random.normal(
        jax.random.PRNGKey(1),
        (num_experts, hidden_size, intermediate_dim),
        dtype=jnp.bfloat16,
    )

    # group_sizes: [num_groups]
    group_sizes = jnp.array([8, 8], dtype=jnp.int32)

    # rhs_scale: [num_groups, 1, 1, n]
    rhs_scale = jax.random.normal(
        jax.random.PRNGKey(2),
        (num_experts, 1, 1, intermediate_dim),
        dtype=jnp.float32,
    )

    # group_offset: [1]
    group_offset = jnp.array([0], dtype=jnp.int32)

    print("Running gmm_v2 in interpreter mode...")
    try:
        out = gmm_v2(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            rhs_scale=rhs_scale,
            group_offset=group_offset,
            maybe_quantize_lhs=True,  # Test the quantized path
            zero_initialize=True,
        )
        print("Output shape:", out.shape)
        print("Output has NaN:", jnp.any(jnp.isnan(out)))
    except Exception as e:
        print("Error running kernel:", e)


if __name__ == "__main__":
    main()
