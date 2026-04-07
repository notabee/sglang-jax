import logging
from typing import Any

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from transformers import PretrainedConfig
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.model_config import ModelConfig, MoEBackend
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import (
    EPMoE,
    FusedEPMoE,
    GateLogit,
    TopK,
    create_moe_weights_mapping,
)
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

class GlmNorm(nnx.Module):
    def __init__(self, dim: int, dtype: jnp.dtype = jnp.bfloat16):
        self.weight = nnx.Param(jnp.ones((dim,), dtype=dtype))
        self.bias = nnx.Param(jnp.zeros((dim,), dtype=dtype))

class GlmDsaIndexer(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: int,
        index_head_dim: int,
        index_n_heads: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        scope_name: str = "indexer",
    ):
        self.head_dim = index_head_dim
        self.n_head = index_n_heads
        
        self.wq_b = LinearBase(
            input_size=q_lora_rank,
            output_size=index_head_dim * index_n_heads,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="wq_b",
        )
        self.wk = LinearBase(
            input_size=hidden_size,
            output_size=index_head_dim,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="wk",
        )
        self.k_norm = GlmNorm(index_head_dim, dtype)
        
        self.weights_proj = LinearBase(
            input_size=hidden_size,
            output_size=index_n_heads,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="weights_proj",
        )

    def __call__(self, hidden_states: jax.Array, qr: jax.Array) -> jax.Array:
        # Dummy implementation for now to allow compilation
        # TODO: Implement full DSA indexing logic
        return jnp.zeros((hidden_states.shape[0], self.n_head), dtype=jnp.int32)

class Glm5Attention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        mesh: jax.sharding.Mesh,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        rms_norm_eps: float = None,
        use_qk_norm: bool = True,
        rotary_dim: int = 0,
        layer_id: int = 0,
        attention_bias: bool = False,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.mesh = mesh
        assert num_heads % num_kv_heads == 0

        self.head_dim = head_dim or hidden_size // num_heads
        self.q_head_num = num_heads
        self.kv_head_num = num_kv_heads

        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.use_qk_norm = use_qk_norm

        if use_qk_norm:
            self.q_norm = RMSNorm(
                self.head_dim, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="q_norm"
            )
            self.k_norm = RMSNorm(
                self.head_dim, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="k_norm"
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.q_a_proj = LinearBase(
            input_size=hidden_size,
            output_size=2048, # q_lora_rank
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_a_proj",
        )
        self.q_a_layernorm = RMSNorm(
            2048, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="q_a_layernorm"
        )
        self.q_b_proj = LinearBase(
            input_size=2048,
            output_size=num_heads * 256, # num_heads * qk_head_dim
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_b_proj",
        )
        self.kv_a_proj_with_mqa = LinearBase(
            input_size=hidden_size,
            output_size=512 + 64, # kv_lora_rank + qk_rope_head_dim
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(
            512, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="kv_a_layernorm"
        )
        self.kv_b_proj = LinearBase(
            input_size=512,
            output_size=num_heads * (192 + 256), # num_heads * (qk_nope_head_dim + v_head_dim)
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="kv_b_proj",
        )
        self.o_proj = LinearBase(
            input_size=num_heads * 256, # num_heads * v_head_dim
            output_size=hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="o_proj",
        )
        self.indexer = GlmDsaIndexer(
            hidden_size=hidden_size,
            q_lora_rank=2048,
            index_head_dim=128,
            index_n_heads=32,
            mesh=mesh,
            dtype=dtype,
            scope_name="indexer",
        )
        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            dtype=dtype,
        )
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> jax.Array:
        # 1. Q projection
        q, _ = self.q_a_proj(hidden_states)
        q = self.q_a_layernorm(q)
        q, _ = self.q_b_proj(q)
        q = q.reshape(-1, self.q_head_num, 256) # 256 is qk_head_dim
        
        # 2. KV projection
        latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
        kv_a, k_pe = jnp.split(latent_cache, [512], axis=-1) # 512 is kv_lora_rank
        kv_a = self.kv_a_layernorm(kv_a)
        kv, _ = self.kv_b_proj(kv_a)
        kv = kv.reshape(-1, self.q_head_num, 192 + 256) # 192 qk_nope, 256 v_head_dim
        k_nope, v = jnp.split(kv, [192], axis=-1)
        
        # 3. Apply RoPE
        q_nope, q_pe = jnp.split(q, [192], axis=-1)
        k_pe = k_pe.reshape(-1, 1, 64)
        
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        
        # Combine Q and K
        q = jnp.concatenate([q_nope, q_pe], axis=-1)
        k_pe_repeated = k_pe.repeat(self.q_head_num, axis=1)
        k_pe_repeated = jax.sharding.reshard(
            k_pe_repeated, NamedSharding(self.mesh, P(None, "tensor", None))
        )
        k = jnp.concatenate([k_nope, k_pe_repeated], axis=-1)
        
        # 4. Attention
        attn_output, kv_fused = self.attn(
            q, k, v, forward_batch=forward_batch, token_to_kv_pool=token_to_kv_pool
        )
        
        # 5. Output projection
        attn_output = attn_output.reshape(-1, self.q_head_num * 256)
        output, _ = self.o_proj(attn_output)
        
        return output, kv_fused

class Glm5MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.layer_id = layer_id

        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="gate_proj",
        )

        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="up_proj",
        )

        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="down_proj",
        )

        self.act_fn = jax.nn.silu

    def __call__(self, hidden_states: jax.Array):
        a1, _ = self.gate_proj(hidden_states)
        a2, _ = self.up_proj(hidden_states)
        intermediate_parallel = a2 * self.act_fn(a1)
        output, _ = self.down_proj(intermediate_parallel)
        return output

class Glm5DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 131072)
        self.head_dim = getattr(config, "head_dim", None) or 128
        use_qk_norm = getattr(config, "use_qk_norm", True)
        
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)
        rotary_dim = int(self.head_dim * partial_rotary_factor)

        self.self_attn = Glm5Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=self.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=use_qk_norm,
            rotary_dim=rotary_dim,
            layer_id=layer_id,
            attention_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            mesh=mesh,
        )

        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)

        if layer_id < first_k_dense_replace:
            self.mlp = Glm5MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_id=layer_id,
                dtype=dtype,
                mesh=mesh,
            )
            self.is_moe_layer = False
            self.moe_gate = None
        else:
            router_dtype = jnp.float32
            self.moe_gate = GateLogit(
                input_size=config.hidden_size,
                num_experts=config.n_routed_experts,
                enable_expert_bias=True,
                weight_dtype=router_dtype,
                score_func=getattr(config, "scoring_func", "sigmoid"),
            )

            self.moe_backend = getattr(config, "moe_backend", MoEBackend.EPMOE)
            self.use_fused = self.moe_backend == MoEBackend.FUSED

            self.topk = TopK(
                topk=config.num_experts_per_tok,
                renormalize=config.norm_topk_prob,
                num_expert_group=getattr(config, "n_group", 1),
                topk_group=getattr(config, "topk_group", 1),
                routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
                layer_id=layer_id,
            )

            if self.use_fused:
                self.mlp = FusedEPMoE(
                    hidden_size=config.hidden_size,
                    num_experts=config.n_routed_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    intermediate_dim=config.moe_intermediate_size,
                    mesh=mesh,
                    ep_size=1, # Default to 1 for now
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_id,
                    renormalize_topk_logits=config.norm_topk_prob,
                    routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
                    use_grouped_topk=getattr(config, "n_group", 1) > 1,
                    num_groups=getattr(config, "n_group", 1),
                    top_k_groups=getattr(config, "topk_group", 1),
                    num_shared_experts=getattr(config, "n_shared_experts", 0),
                    moe_shared_expert_intermediate_size=config.moe_intermediate_size,
                    quantization_config=getattr(config, "quantization_config", None),
                )
            else:
                self.mlp = EPMoE(
                    hidden_size=config.hidden_size,
                    num_experts=config.n_routed_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    intermediate_dim=config.moe_intermediate_size,
                    mesh=mesh,
                    ep_size=1, # Default to 1 for now
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_id,
                    quantization_config=getattr(config, "quantization_config", None),
                )

            num_shared_experts = getattr(config, "n_shared_experts", 0)
            if num_shared_experts > 0 and not self.use_fused:
                self.shared_experts = Glm5MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size * num_shared_experts,
                    layer_id=layer_id,
                    dtype=dtype,
                    mesh=mesh,
                )
            else:
                self.shared_experts = None
            self.is_moe_layer = True

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="input_layernorm",
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="post_attention_layernorm",
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
        dispatch_info: ExpertLocationMetadata | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_moe_layer:
            if self.shared_experts is not None:
                shared_output = self.shared_experts(hidden_states)
            else:
                shared_output = None
            router_logits = self.moe_gate(hidden_states)

            correction_bias = self.moe_gate.bias.value if self.moe_gate.bias is not None else None
            topk_weights, topk_ids = self.topk(
                router_logits,
                correction_bias,
                dispatch_info=dispatch_info,
            )

            hidden_states = self.mlp(hidden_states, topk_weights, topk_ids)

            if shared_output is not None:
                hidden_states = hidden_states + shared_output
        else:
            hidden_states = self.mlp(hidden_states)
            topk_ids = None

        return hidden_states, residual, kv_fused, topk_ids

class Glm5Model(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                Glm5DecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                )
                for i in range(5) # For dummy testing
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype, scope_name="norm"
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> jax.Array:
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        residual = None
        layers_kv_fused = []
        layers_topk_ids = []
        for layer in self.layers:
            hidden_states, residual, kv_fused, topk_ids = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
                dispatch_info=forward_batch.expert_location_metadata,
            )
            layers_kv_fused.append(kv_fused)
            layers_topk_ids.append(topk_ids)

        if residual is not None:
            hidden_states += residual

        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_fused, layers_topk_ids

class Glm5ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        self.model = Glm5Model(config, dtype=self.dtype, mesh=mesh)
        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused, layers_topk_ids = self.model(
            forward_batch,
            token_to_kv_pool,
        )
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, layers_kv_fused, True, layers_topk_ids

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_glm5_weight_mappings(model_config)
        loader.load_weights_from_safetensors(weight_mappings)
        
        # Invert scales because checkpoint provides weight_scale_inv
        logger.info("Inverting weight scales...")
        for layer in self.model.layers:
            attn = layer.self_attn
            if hasattr(attn, "q_a_proj") and hasattr(attn.q_a_proj, "weight_scale"):
                if attn.q_a_proj.weight_scale is not None:
                    attn.q_a_proj.weight_scale.value = 1.0 / attn.q_a_proj.weight_scale.value
                if attn.q_b_proj.weight_scale is not None:
                    attn.q_b_proj.weight_scale.value = 1.0 / attn.q_b_proj.weight_scale.value
                if attn.kv_a_proj_with_mqa.weight_scale is not None:
                    attn.kv_a_proj_with_mqa.weight_scale.value = 1.0 / attn.kv_a_proj_with_mqa.weight_scale.value
                if attn.kv_b_proj.weight_scale is not None:
                    attn.kv_b_proj.weight_scale.value = 1.0 / attn.kv_b_proj.weight_scale.value
                if attn.o_proj.weight_scale is not None:
                    attn.o_proj.weight_scale.value = 1.0 / attn.o_proj.weight_scale.value
                if hasattr(attn.indexer, "wk") and attn.indexer.wk.weight_scale is not None:
                    attn.indexer.wk.weight_scale.value = 1.0 / attn.indexer.wk.weight_scale.value
                if hasattr(attn.indexer, "wq_b") and attn.indexer.wq_b.weight_scale is not None:
                    attn.indexer.wq_b.weight_scale.value = 1.0 / attn.indexer.wq_b.weight_scale.value
            
            mlp = layer.mlp
            if hasattr(mlp, "gate_proj") and hasattr(mlp.gate_proj, "weight_scale"):
                if mlp.gate_proj.weight_scale is not None:
                    mlp.gate_proj.weight_scale.value = 1.0 / mlp.gate_proj.weight_scale.value
                if mlp.up_proj.weight_scale is not None:
                    mlp.up_proj.weight_scale.value = 1.0 / mlp.up_proj.weight_scale.value
                if mlp.down_proj.weight_scale is not None:
                    mlp.down_proj.weight_scale.value = 1.0 / mlp.down_proj.weight_scale.value
            elif hasattr(mlp, "experts") and hasattr(mlp.experts, "wi_0_scale"):
                if mlp.experts.wi_0_scale is not None:
                     mlp.experts.wi_0_scale.value = 1.0 / mlp.experts.wi_0_scale.value
                if mlp.experts.wi_1_scale is not None:
                     mlp.experts.wi_1_scale.value = 1.0 / mlp.experts.wi_1_scale.value
                if mlp.experts.wo_scale is not None:
                     mlp.experts.wo_scale.value = 1.0 / mlp.experts.wo_scale.value
            
            if hasattr(layer, "shared_experts") and layer.shared_experts is not None:
                shared = layer.shared_experts
                if hasattr(shared.gate_proj, "weight_scale") and shared.gate_proj.weight_scale is not None:
                    shared.gate_proj.weight_scale.value = 1.0 / shared.gate_proj.weight_scale.value
                if hasattr(shared.up_proj, "weight_scale") and shared.up_proj.weight_scale is not None:
                    shared.up_proj.weight_scale.value = 1.0 / shared.up_proj.weight_scale.value
                if hasattr(shared.down_proj, "weight_scale") and shared.down_proj.weight_scale is not None:
                    shared.down_proj.weight_scale.value = 1.0 / shared.down_proj.weight_scale.value
        
        logger.info("Weights loaded and scales inverted successfully!")

    def _create_glm5_weight_mappings(self, model_config: ModelConfig) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }

        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            )

        num_layers = 5 # For dummy testing
        first_k_dense_replace = getattr(self.config, "first_k_dense_replace", 0)

        quant_config = getattr(model_config, "quantization_config", None)
        is_static_quant = quant_config is not None and quant_config.is_static_checkpoint

        for layer_idx in range(num_layers):
            layer_mappings = self._create_moe_layer_mappings(
                layer_idx, layer_idx < first_k_dense_replace, is_static_quant=is_static_quant
            )
            mappings.update(layer_mappings)

        return mappings

    def _create_moe_layer_mappings(
        self, layer_idx: int, is_mlp_layer: bool, is_static_quant: bool = False
    ) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"

        mappings = {
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        w_name = "weight_q" if is_static_quant else "weight"

        # Attention mappings (separate Q, K, V in checkpoint)
        # Attention mappings (MLA)
        mappings[f"{prefix}.self_attn.q_a_proj.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.q_a_proj.{w_name}",
            sharding=(None, None),
            transpose=False,
        )
        mappings[f"{prefix}.self_attn.q_a_layernorm.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.q_a_layernorm.scale",
            sharding=(None,),
        )
        mappings[f"{prefix}.self_attn.q_b_proj.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.q_b_proj.{w_name}",
            sharding=(None, "tensor"),
            transpose=False,
        )
        mappings[f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.kv_a_proj_with_mqa.{w_name}",
            sharding=(None, None),
            transpose=False,
        )
        mappings[f"{prefix}.self_attn.kv_a_layernorm.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.kv_a_layernorm.scale",
            sharding=(None,),
        )
        mappings[f"{prefix}.self_attn.kv_b_proj.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.kv_b_proj.{w_name}",
            sharding=(None, "tensor"),
            transpose=False,
        )
        mappings[f"{prefix}.self_attn.o_proj.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.o_proj.{w_name}",
            sharding=("tensor", None),
            transpose=False,
        )

        # Indexer mappings
        mappings[f"{prefix}.self_attn.indexer.wq_b.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.indexer.wq_b.{w_name}",
            sharding=(None, None),
            transpose=False,
        )
        mappings[f"{prefix}.self_attn.indexer.wk.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.indexer.wk.{w_name}",
            sharding=(None, None),
            transpose=False,
        )
        mappings[f"{prefix}.self_attn.indexer.weights_proj.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.indexer.weights_proj.{w_name}",
            sharding=(None, None),
            transpose=False,
        )
        mappings[f"{prefix}.self_attn.indexer.k_norm.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.indexer.k_norm.weight",
            sharding=(None,),
        )
        mappings[f"{prefix}.self_attn.indexer.k_norm.bias"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.indexer.k_norm.bias",
            sharding=(None,),
        )

        if is_static_quant:
            mappings[f"{prefix}.self_attn.q_a_proj.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_a_proj.weight_scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{prefix}.self_attn.q_b_proj.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_b_proj.weight_scale",
                sharding=("tensor",),
                transpose=False,
            )
            mappings[f"{prefix}.self_attn.kv_a_proj_with_mqa.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.kv_a_proj_with_mqa.weight_scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{prefix}.self_attn.kv_b_proj.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.kv_b_proj.weight_scale",
                sharding=("tensor",),
                transpose=False,
            )
            mappings[f"{prefix}.self_attn.o_proj.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight_scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{prefix}.self_attn.indexer.wk.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.indexer.wk.weight_scale",
                sharding=(None,),
                transpose=False,
            )
            mappings[f"{prefix}.self_attn.indexer.wq_b.weight_scale_inv"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.indexer.wq_b.weight_scale",
                sharding=(None,),
                transpose=False,
            )

        # DSA Indexer Norm
        mappings[f"{prefix}.self_attn.indexer.k_norm.weight"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.indexer.k_norm.weight", sharding=(None,)
        )
        mappings[f"{prefix}.self_attn.indexer.k_norm.bias"] = WeightMapping(
            target_path=f"{target_prefix}.self_attn.indexer.k_norm.bias", sharding=(None,)
        )

        if is_mlp_layer:
            mappings[f"{prefix}.mlp.gate_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.{w_name}",
                sharding=(None, "tensor"),
                transpose=False,
            )
            mappings[f"{prefix}.mlp.up_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.{w_name}",
                sharding=(None, "tensor"),
                transpose=False,
            )
            mappings[f"{prefix}.mlp.down_proj.weight"] = WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.{w_name}",
                sharding=("tensor", None),
                transpose=False,
            )
            if is_static_quant:
                mappings[f"{prefix}.mlp.gate_proj.weight_scale_inv"] = WeightMapping(
                    target_path=f"{target_prefix}.mlp.gate_proj.weight_scale",
                    sharding=(None, None),
                    transpose=False,
                )
                mappings[f"{prefix}.mlp.up_proj.weight_scale_inv"] = WeightMapping(
                    target_path=f"{target_prefix}.mlp.up_proj.weight_scale",
                    sharding=(None, None),
                    transpose=False,
                )
                mappings[f"{prefix}.mlp.down_proj.weight_scale_inv"] = WeightMapping(
                    target_path=f"{target_prefix}.mlp.down_proj.weight_scale",
                    sharding=(None, None),
                    transpose=False,
                )
        else:
            mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.kernel",
                sharding=(None, None),
                transpose=True,
            )
            # GLM-4 uses e_score_correction_bias
            mappings[f"{prefix}.mlp.gate.e_score_correction_bias"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.bias", sharding=(None,)
            )

            num_logical_experts = self.config.n_routed_experts
            moe_backend = getattr(self.config, "moe_backend", "epmoe")

            moe_mappings = create_moe_weights_mapping(
                prefix=prefix,
                target_prefix=target_prefix,
                num_experts=num_logical_experts,
                expert_type_names=("gate_proj", "up_proj", "down_proj"),
                moe_backend=moe_backend,
                physical_to_logical_map=None, # Handle physical mapping if needed later
            )

            if is_static_quant:
                new_moe_mappings = {}
                BLOCK_SIZE = 256
                hidden_size = self.config.hidden_size
                inter_size = self.config.moe_intermediate_size
                num_physical_experts = num_logical_experts # Assuming no redundant experts for now
                use_fused = moe_backend == "fused"

                for key, mapping in moe_mappings.items():
                    target_param = mapping.target_path[0]
                    src_paths = mapping.target_path[1:]

                    new_moe_mappings[key] = WeightMapping(
                        target_path=[target_param] + src_paths,
                        sharding=mapping.sharding,
                        transpose=True,
                        concat_axis=mapping.concat_axis,
                        physical_to_logical_map=mapping.physical_to_logical_map,
                    )

                    scale_key = key + "_scale"
                    target_scale_param = target_param + "_scale"
                    scale_src_paths = [p.replace(".weight", ".weight_scale_inv") for p in src_paths]

                    is_w2 = target_param.endswith("wo") or target_param.endswith("w2")
                    out_dim = hidden_size if is_w2 else inter_size

                    # For GLM-5 FP8, scales are stored as [num_experts, in_blocks, out_blocks]
                    # We need to transpose them to [num_experts, out_blocks, in_blocks] for moe.py
                    new_moe_mappings[scale_key] = WeightMapping(
                        target_path=[target_scale_param] + scale_src_paths,
                        sharding=None,
                        transpose=False,
                        transpose_axes=(0, 2, 1),
                        reshape=None,
                        repeat=None,
                        concat_axis=mapping.concat_axis,
                        physical_to_logical_map=mapping.physical_to_logical_map,
                    )
                moe_mappings = new_moe_mappings

            mappings.update(moe_mappings)

            num_shared = getattr(self.config, "n_shared_experts", 0)
            if num_shared > 0:
                mappings[f"{prefix}.mlp.shared_experts.gate_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.shared_experts.gate_proj.{w_name}",
                    sharding=(None, "tensor"),
                    transpose=False,
                )
                mappings[f"{prefix}.mlp.shared_experts.up_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.shared_experts.up_proj.{w_name}",
                    sharding=(None, "tensor"),
                    transpose=False,
                )
                mappings[f"{prefix}.mlp.shared_experts.down_proj.weight"] = WeightMapping(
                    target_path=f"{target_prefix}.shared_experts.down_proj.{w_name}",
                    sharding=("tensor", None),
                    transpose=False,
                )
                if is_static_quant:
                    mappings[f"{prefix}.mlp.shared_experts.gate_proj.weight_scale_inv"] = WeightMapping(
                        target_path=f"{target_prefix}.shared_experts.gate_proj.weight_scale",
                        sharding=(None,),
                        transpose=False,
                    )
                    mappings[f"{prefix}.mlp.shared_experts.up_proj.weight_scale_inv"] = WeightMapping(
                        target_path=f"{target_prefix}.shared_experts.up_proj.weight_scale",
                        sharding=(None,),
                        transpose=False,
                    )
                    mappings[f"{prefix}.mlp.shared_experts.down_proj.weight_scale_inv"] = WeightMapping(
                        target_path=f"{target_prefix}.shared_experts.down_proj.weight_scale",
                        sharding=(None,),
                        transpose=False,
                    )

        return mappings

class GlmMoeDsaForCausalLM(Glm5ForCausalLM):
    pass

EntryClass = [Glm5ForCausalLM, GlmMoeDsaForCausalLM]
