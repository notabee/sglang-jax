import os
import argparse
import json
import urllib.request
import urllib.error

# Hardcoded Hardware spec database
HARDWARE_PROFILES = {
    "ironwood": {
        "name": "Ironwood (TPU7x)",
        "peak_tflops_fp8": 2307.0,      # per core (4614 TFLOPS per chip / 2)
        "hbm_bandwidth_gb_s": 3690.0,   # per core (7380 GB/s per chip / 2)
        "hbm_capacity_gb": 96.0,        # per core (192 GB / 2)
        "c2c_bandwidth_gb_s": 600.0,    # uni-directional link bandwidth
        "c2c_latency_ms": 0.0013
    },
    "ghostfish_p7": {
        "name": "GhostFish (P7)",
        "peak_tflops_fp8": 1992.4,      # per core
        "hbm_bandwidth_gb_s": 3686.5,   # per core
        "hbm_capacity_gb": 96.0,        # per core
        "c2c_bandwidth_gb_s": 1600.0,
        "c2c_latency_ms": 0.001
    },
    "tpu_v5p": {
        "name": "TPU v5p",
        "peak_tflops_fp8": 229.5,       # per core (459 BF16 TFLOPS / 2)
        "hbm_bandwidth_gb_s": 1382.5,   # per core (2765 / 2)
        "hbm_capacity_gb": 47.5,        # per core
        "c2c_bandwidth_gb_s": 600.0,
        "c2c_latency_ms": 0.0015
    }
}

DTYPE_SIZES = {"bf16": 2.0, "fp8": 1.0}

def mib(elements, dtype="fp8"):
    return (elements * DTYPE_SIZES[dtype]) / (1024**2)

def gib(elements, dtype="fp8"):
    return (elements * DTYPE_SIZES[dtype]) / (1024**3)

def parse_args():
    parser = argparse.ArgumentParser(description="Generalized Model Roofline Calculator")
    parser.add_argument("--model-id", type=str, help="Hugging Face Model ID (e.g. google/gemma-4-31B)")
    parser.add_argument("--hardware", type=str, choices=list(HARDWARE_PROFILES.keys()), help="Target hardware platform")
    
    # Workload params
    parser.add_argument("--regime", type=str, choices=["prefill", "decode"], default="decode", help="Regime to analyze")
    parser.add_argument("--batch-size", type=int, default=64, help="Serving Batch Size")
    parser.add_argument("--seq-len-kv", type=int, default=1024, help="Context sequence length (S)")
    parser.add_argument("--gen-len", type=int, default=500, help="Decode token generation length")
    parser.add_argument("--token-step", type=str, choices=["FirstToken", "NextToken"], default="NextToken", help="Decode token step")
    
    # Parallelism params
    parser.add_argument("--tp", type=int, default=8, help="Tensor Parallelism size")
    parser.add_argument("--dp", type=int, default=1, help="Data Parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert Parallelism size")
    
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output MD and HTML reports")
    return parser.parse_args()

def fetch_hf_config(model_id):
    url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    print(f"Fetching config.json from Hugging Face: {url}...")
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode())
    except urllib.error.URLError as e:
        print(f"Error fetching config.json: {e}")
        raise SystemExit(f"Could not load Hugging Face config for repo '{model_id}'")

def extract_model_params(config_dict, model_id):
    if "text_config" in config_dict:
        cfg = config_dict["text_config"]
    else:
        cfg = config_dict

    params = {}
    params["name"] = config_dict.get("_name_or_path") or model_id
    params["hidden_size"] = cfg.get("hidden_size") or cfg.get("d_model")
    params["num_layers"] = cfg.get("num_hidden_layers") or cfg.get("num_layers") or cfg.get("n_layers")
    
    # MLP sizes
    params["dense_intermediate_size"] = cfg.get("intermediate_size") or cfg.get("ffn_hidden_size") or 0
    params["moe_intermediate_size"] = cfg.get("moe_intermediate_size") or cfg.get("moe_ffn_hidden_size") or params["dense_intermediate_size"]
    
    # MoE experts
    params["num_experts"] = cfg.get("n_routed_experts") or cfg.get("num_local_experts") or cfg.get("num_experts") or 0
    params["num_active_experts"] = cfg.get("num_experts_per_tok") or cfg.get("num_active_experts") or cfg.get("top_k_experts") or cfg.get("top_k") or 0
    params["num_shared_experts"] = cfg.get("n_shared_experts") or cfg.get("num_shared_experts") or 0
    
    # MLA / GQA heads
    params["num_query_heads"] = cfg.get("num_attention_heads") or cfg.get("n_heads") or 1
    params["num_kv_heads"] = cfg.get("num_key_value_heads") or cfg.get("n_kv_heads") or params["num_query_heads"]
    params["qk_head_dim"] = cfg.get("head_dim") or (params["hidden_size"] // params["num_query_heads"] if params["hidden_size"] and params["num_query_heads"] else 128)
    params["v_head_dim"] = params["qk_head_dim"]
    
    # MLA rankings (DeepSeek style)
    params["q_lora_rank"] = cfg.get("q_lora_rank")
    params["kv_lora_rank"] = cfg.get("kv_lora_rank")
    params["qk_nope_head_dim"] = cfg.get("qk_nope_head_dim")
    params["qk_rope_head_dim"] = cfg.get("qk_rope_head_dim")
    
    if params["q_lora_rank"] is not None:
        params["attention_type"] = "MLA"
    else:
        params["attention_type"] = "Standard"
        
    params["attention_k_eq_v"] = cfg.get("attention_k_eq_v", False)
    
    return params

def run_calculations(model, hw, B, T, S, tp_size, dp_size, ep_size, regime):
    D = model["hidden_size"]
    L = model["num_layers"]
    dense_L = 3 if model["num_experts"] > 0 else L
    moe_L = L - dense_L if model["num_experts"] > 0 else 0
    F_moe = model["moe_intermediate_size"]
    F_dense = model["dense_intermediate_size"]
    E = model["num_experts"]
    top_k = model["num_active_experts"]
    n_shared = model["num_shared_experts"]
    shared_F = F_moe * n_shared

    # Attention params
    d_h = model["qk_head_dim"]
    d_v = model["v_head_dim"]
    N_q = model["num_query_heads"]
    N_kv = model["num_kv_heads"]
    
    attention_type = model["attention_type"]
    k_eq_v = model["attention_k_eq_v"]

    num_devices = tp_size * dp_size * ep_size

    peak_tflops = hw["peak_tflops_fp8"]
    hbm_bw = hw["hbm_bandwidth_gb_s"]
    c2c_bw = hw["c2c_bandwidth_gb_s"]


    def t_hbm(elements, dtype="fp8"):
        sz = mib(elements, dtype)
        return (sz * 1024**2) / (hbm_bw * 1024**3) * 1000

    def t_comp(flops):
        return flops / (peak_tflops * 10**9)

    def t_ici_ar(size_in_mib):
        return 2 * (size_in_mib * 1024**2) / (2 * c2c_bw * 1024**3) * 1000

    def t_ici_a2a(size_in_mib):
        return (size_in_mib * 1024**2) / (2 * c2c_bw * 1024**3) * 1000

    metrics = {}

    if attention_type == "MLA":
        q_lora = model["q_lora_rank"]
        kv_lora = model["kv_lora_rank"]
        d_nope = model["qk_nope_head_dim"]
        d_rope = model["qk_rope_head_dim"]

        # MLA: QKV_a Proj
        flops_qkv_a = 2 * B * T * D * (q_lora + kv_lora + d_rope)
        mib_qkv_a = mib(B * T * D, "bf16") + mib(D * (q_lora + kv_lora + d_rope), "fp8") + mib(B * T * (q_lora + kv_lora + d_rope), "bf16")
        t_comp_qkv_a = t_comp(flops_qkv_a)
        t_hbm_qkv_a = t_hbm(B * T * D, "bf16") + t_hbm(D * (q_lora + kv_lora + d_rope), "fp8") + t_hbm(B * T * (q_lora + kv_lora + d_rope), "bf16")
        metrics["qkv_a"] = {"comp": t_comp_qkv_a, "hbm": t_hbm_qkv_a, "flops": flops_qkv_a, "size": mib_qkv_a}

        local_q_out = N_q * d_h // tp_size
        flops_qb = 2 * B * T * q_lora * local_q_out
        mib_qb = mib(B * T * q_lora, "bf16") + mib(q_lora * local_q_out, "fp8") + mib(B * T * local_q_out, "bf16")
        t_comp_qb = t_comp(flops_qb)
        t_hbm_qb = t_hbm(B * T * q_lora, "bf16") + t_hbm(q_lora * local_q_out, "fp8") + t_hbm(B * T * local_q_out, "bf16")
        metrics["q_b"] = {"comp": t_comp_qb, "hbm": t_hbm_qb, "flops": flops_qb, "size": mib_qb}

        local_heads = N_q // tp_size
        if regime == "prefill":
            flops_attn = (2 * B * local_heads * d_nope * kv_lora) + (2 * B * local_heads * (T * T / 2) * kv_lora) * 2 + (2 * B * local_heads * kv_lora * d_v)
            mib_attn = mib(kv_lora * local_heads * (d_nope + d_v), "fp8") + mib(B * T * kv_lora, "fp8") * 2 + mib(B * local_heads * (d_h + d_v), "bf16")
        else:
            flops_attn = (2 * B * local_heads * d_nope * kv_lora) + (2 * B * local_heads * S * kv_lora) * 2 + (2 * B * local_heads * kv_lora * d_v)
            mib_attn = mib(kv_lora * local_heads * (d_nope + d_v), "fp8") + mib(B * S * kv_lora, "fp8") * 2 + mib(B * local_heads * (d_h + d_v), "bf16")

        t_comp_attn = t_comp(flops_attn)
        t_hbm_attn = t_hbm(mib_attn * 1024**2, "fp8")
        metrics["attn_core"] = {"comp": t_comp_attn, "hbm": t_hbm_attn, "flops": flops_attn, "size": mib_attn}

        local_in_o = N_q * d_v // tp_size
        flops_o = 2 * B * T * local_in_o * D
        mib_o = mib(B * T * local_in_o, "bf16") + mib(local_in_o * D, "fp8") + mib(B * T * D, "bf16")
        t_comp_o = t_comp(flops_o)
        t_hbm_o = t_hbm(mib_o * 1024**2, "fp8")
        t_ici_o = t_ici_ar(mib(B * T * D, "bf16"))
        metrics["out_proj"] = {"comp": t_comp_o, "hbm": t_hbm_o, "ici": t_ici_o, "flops": flops_o, "size": mib_o}

        t_roof_attn_total = max(t_comp_qkv_a, t_hbm_qkv_a) + max(t_comp_qb, t_hbm_qb) + max(t_comp_attn, t_hbm_attn) + max(t_comp_o, t_hbm_o) + t_ici_o
    else:
        # Standard Attention
        local_q_out = N_q * d_h // tp_size
        flops_q = 2 * B * T * D * local_q_out
        mib_q = mib(B * T * D, "bf16") + mib(D * local_q_out, "fp8") + mib(B * T * local_q_out, "bf16")
        t_comp_q = t_comp(flops_q)
        t_hbm_q = t_hbm(B * T * D, "bf16") + t_hbm(D * local_q_out, "fp8") + t_hbm(B * T * local_q_out, "bf16")
        metrics["q_proj"] = {"comp": t_comp_q, "hbm": t_hbm_q, "flops": flops_q, "size": mib_q}

        local_kv_out = N_kv * d_h // tp_size
        flops_k = 2 * B * T * D * local_kv_out
        mib_k = mib(B * T * D, "bf16") + mib(D * local_kv_out, "fp8") + mib(B * T * local_kv_out, "bf16")
        t_comp_k = t_comp(flops_k)
        t_hbm_k = t_hbm(B * T * D, "bf16") + t_hbm(D * local_kv_out, "fp8") + t_hbm(B * T * local_kv_out, "bf16")
        metrics["k_proj"] = {"comp": t_comp_k, "hbm": t_hbm_k, "flops": flops_k, "size": mib_k}

        if k_eq_v:
            t_comp_v, t_hbm_v, flops_v, mib_v = 0, 0, 0, 0
        else:
            flops_v = 2 * B * T * D * local_kv_out
            mib_v = mib(B * T * D, "bf16") + mib(D * local_kv_out, "fp8") + mib(B * T * local_kv_out, "bf16")
            t_comp_v = t_comp(flops_v)
            t_hbm_v = t_hbm(B * T * D, "bf16") + t_hbm(D * local_kv_out, "fp8") + t_hbm(B * T * local_kv_out, "bf16")
        metrics["v_proj"] = {"comp": t_comp_v, "hbm": t_hbm_v, "flops": flops_v, "size": mib_v}

        local_heads = N_q // tp_size
        if regime == "prefill":
            flops_qk = 2 * B * local_heads * (T * T / 2) * d_h
            flops_av = 2 * B * local_heads * (T * T / 2) * d_v
            mib_attn = mib(B * T * local_heads * d_h, "bf16") + mib(B * T * local_kv_out, "fp8") * (1 if k_eq_v else 2)
        else:
            flops_qk = 2 * B * local_heads * T * S * d_h
            flops_av = 2 * B * local_heads * T * S * d_v
            mib_attn = mib(B * T * local_heads * d_h, "bf16") + mib(B * S * local_kv_out, "fp8") * (1 if k_eq_v else 2)

        flops_attn = flops_qk + flops_av
        t_comp_attn = t_comp(flops_attn)
        t_hbm_attn = t_hbm(mib_attn * 1024**2, "fp8")
        metrics["attn_core"] = {"comp": t_comp_attn, "hbm": t_hbm_attn, "flops": flops_attn, "size": mib_attn}

        local_in_o = N_q * d_h // tp_size
        flops_o = 2 * B * T * local_in_o * D
        mib_o = mib(B * T * local_in_o, "bf16") + mib(local_in_o * D, "fp8") + mib(B * T * D, "bf16")
        t_comp_o = t_comp(flops_o)
        t_hbm_o = t_hbm(mib_o * 1024**2, "fp8")
        t_ici_o = t_ici_ar(mib(B * T * D, "bf16"))
        metrics["out_proj"] = {"comp": t_comp_o, "hbm": t_hbm_o, "ici": t_ici_o, "flops": flops_o, "size": mib_o}

        t_roof_attn_total = max(t_comp_q, t_hbm_q) + max(t_comp_k, t_hbm_k) + max(t_comp_v, t_hbm_v) + max(t_comp_attn, t_hbm_attn) + max(t_comp_o, t_hbm_o) + t_ici_o

    # Dense MLP
    if F_dense > 0:
        local_f_dense = F_dense // tp_size
        flops_dense = 2 * B * T * D * local_f_dense * 2 + 2 * B * T * local_f_dense * D
        mib_dense = mib(B * T * D, "bf16") * 2 + mib(D * local_f_dense * 3, "fp8") + mib(B * T * local_f_dense, "bf16")
        t_comp_dense = t_comp(flops_dense)
        t_hbm_dense = t_hbm(mib_dense * 1024**2, "fp8")
        t_ici_dense = t_ici_ar(mib(B * T * D, "bf16"))
        t_roof_dense = max(t_comp_dense, t_hbm_dense) + t_ici_dense
    else:
        t_roof_dense, t_comp_dense, t_hbm_dense, t_ici_dense = 0, 0, 0, 0
    metrics["dense_mlp"] = {"comp": t_comp_dense, "hbm": t_hbm_dense, "ici": t_ici_dense, "flops": flops_dense, "size": mib_dense, "total": t_roof_dense}

    # MoE Layer
    if E > 0:
        local_f_shared = shared_F // tp_size
        flops_shared = 2 * B * T * D * local_f_shared * 2 + 2 * B * T * local_f_shared * D
        t_comp_shared = t_comp(flops_shared)
        t_mem_shared = t_hbm(B * T * D * 2, "bf16") + t_hbm(D * local_f_shared * 3, "fp8")
        mib_shared = mib(B * T * D * 2, "bf16") + mib(D * local_f_shared * 3, "fp8")

        local_tokens_per_core = B * T * top_k
        flops_routed = 2 * local_tokens_per_core * D * F_moe * 2 + 2 * local_tokens_per_core * F_moe * D
        t_comp_routed = t_comp(flops_routed)
        hbm_routed_weights_elements = E * 3 * D * F_moe
        t_mem_routed = t_hbm(local_tokens_per_core * D * 2, "bf16") + t_hbm(hbm_routed_weights_elements, "fp8")
        mib_routed = mib(local_tokens_per_core * D * 2, "bf16") + mib(hbm_routed_weights_elements, "fp8")

        t_ici_dispatch = t_ici_a2a(mib(B * T * top_k * D, "bf16")) * 2
        t_roof_moe = (max(t_comp_shared, t_mem_shared) + max(t_comp_routed, t_mem_routed)) + t_ici_dispatch
    else:
        t_roof_moe, t_comp_shared, t_mem_shared, t_comp_routed, t_mem_routed, t_ici_dispatch, mib_shared, mib_routed = 0, 0, 0, 0, 0, 0, 0, 0
    metrics["moe"] = {
        "shared_comp": t_comp_shared, "shared_hbm": t_mem_shared,
        "routed_comp": t_comp_routed, "routed_hbm": t_mem_routed,
        "ici": t_ici_dispatch, "total": t_roof_moe
    }

    # Totals
    metrics["attn_total"] = t_roof_attn_total
    metrics["dense_total"] = t_roof_dense
    metrics["moe_total"] = t_roof_moe
    
    e2e = L * t_roof_attn_total + dense_L * t_roof_dense + moe_L * t_roof_moe
    metrics["e2e_latency_ms"] = e2e
    metrics["throughput_tok_s"] = (B * T) / (e2e / 1000.0)
    
    # Weight sizes for memory breakdown
    if attention_type == "MLA":
        metrics["attn_w_global"] = gib(L * (D * (q_lora + kv_lora + d_rope) + q_lora * N_q * d_h + kv_lora * (N_q // tp_size) * (d_nope + d_v) + local_in_o * D), "fp8") * num_devices
    else:
        metrics["attn_w_global"] = gib(L * (D * N_q * d_h + D * N_kv * d_h * (1 if k_eq_v else 2) + N_q * d_h * D), "fp8") * num_devices

    metrics["moe_w_global"] = gib(moe_L * (E * 3 * D * F_moe + D * shared_F * 3), "fp8") * num_devices if E > 0 else 0
    
    if attention_type == "MLA":
        metrics["kv_cache_global"] = gib(B * S * kv_lora * 2 * L, "fp8") * num_devices
    else:
        metrics["kv_cache_global"] = gib(B * S * N_kv * d_h * (1 if k_eq_v else 2) * L, "fp8") * num_devices

    return metrics

def get_output_filenames(model_name, regime, seq_len_kv, gen_len, dtype, dp, tp, token_step):
    # Differentiate model size names
    model_size = "Model"
    if "31B" in model_name or "31b" in model_name:
        model_size = "31B"
    elif "26B" in model_name or "26b" in model_name:
        model_size = "26B"
    elif "9B" in model_name or "9b" in model_name:
        model_size = "9B"
    else:
        model_size = model_name.split("/")[-1]

    prompt_len_k = f"{seq_len_kv // 1024}k" if seq_len_kv >= 1024 else f"{seq_len_kv}"

    if regime == "prefill":
        base_name = f"{model_size} prefill {prompt_len_k} {dtype} DP={dp} TP={tp}"
    else:
        base_name = f"{model_size} decode {prompt_len_k}-{gen_len} {dtype} DP={dp} TP={tp} {token_step}"
        
    return f"{base_name}.md", f"{base_name}.html"

def main():
    args = parse_args()
    
    model_id = args.model_id
    if not model_id:
        model_id = input("Enter Hugging Face Model ID: ").strip()
    
    hardware = args.hardware
    if not hardware:
        print("\nAvailable Hardware Platforms:")
        for idx, key in enumerate(HARDWARE_PROFILES.keys()):
            print(f"  {idx + 1}. {HARDWARE_PROFILES[key]['name']} ({key})")
        hw_idx = int(input("Select target hardware number: ")) - 1
        hardware = list(HARDWARE_PROFILES.keys())[hw_idx]

    config_dict = fetch_hf_config(model_id)
    model = extract_model_params(config_dict, model_id)
    hw = HARDWARE_PROFILES[hardware]

    # Target Workload configurations
    B = args.batch_size
    S = args.seq_len_kv
    tp_size = args.tp
    dp_size = args.dp
    ep_size = args.ep
    regime = args.regime
    token_step = args.token_step
    gen_len = args.gen_len

    T = args.seq_len_kv if regime == "prefill" else 1

    # Run calculations
    res = run_calculations(model, hw, B, T, S, tp_size, dp_size, ep_size, regime)

    # Layout configurations
    N_q = model["num_query_heads"]
    N_kv = model["num_kv_heads"]
    d_h = model["qk_head_dim"]
    d_v = model["v_head_dim"]
    D = model["hidden_size"]
    L = model["num_layers"]
    dense_L = 3 if model["num_experts"] > 0 else L
    moe_L = L - dense_L if model["num_experts"] > 0 else 0
    F_moe = model["moe_intermediate_size"]
    F_dense = model["dense_intermediate_size"]
    E = model["num_experts"]
    top_k = model["num_active_experts"]
    
    attention_type = model["attention_type"]
    k_eq_v = model["attention_k_eq_v"]
    
    num_devices = tp_size * dp_size * ep_size
    local_q_out = N_q * d_h // tp_size
    local_kv_out = N_kv * d_h // tp_size
    local_f_dense = F_dense // tp_size

    # Build exact filename
    md_filename, html_filename = get_output_filenames(model["name"], regime, S, gen_len, dtype="fp8", dp=dp_size, tp=tp_size, token_step=token_step)

    lines = []
    lines.append(f"# {model['name']} Roofline Analysis ({regime.capitalize()} Path)\n")
    lines.append(f"> **Target Hardware Platform:** {hw['name']}\n")
    lines.append(f"> **Attention Type:** {attention_type} {'(K=V)' if k_eq_v else ''}\n")
    lines.append(f"> **Parallelism Layout:** TP={tp_size}, DP={dp_size}, EP={ep_size}\n")
    
    # 1. Model Details Table
    lines.append("## 1. Model Architecture details")
    lines.append("| Parameter | Symbol | Value | Notes |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(f"| Hidden Dimension | D | {D} | Channel size of main path |")
    lines.append(f"| # of Layers | L | {L} | Total block layers |")
    lines.append(f"| Dense MLP Layers | dense_L | {dense_L} | Dense block layer count |")
    lines.append(f"| MoE Layers | moe_L | {moe_L} | MoE block layer count |")
    lines.append(f"| Dense Intermediate Dimension | F_dense | {F_dense} | Dense FFN hidden size |")
    lines.append(f"| MoE Intermediate Dimension | F_moe | {F_moe} | MoE FFN hidden size |")
    lines.append(f"| # of Experts | E | {E} | Total routing choices |")
    lines.append(f"| # of Activated Experts | top_k | {top_k} | Top-K routing constraint |")
    lines.append(f"| # of Query Heads | N_q | {N_q} | Attn Q head count |")
    lines.append(f"| # of KV Heads | N_kv | {N_kv} | Attn KV head count |")
    lines.append(f"| Head Dimension | d_h | {d_h} | Projection element size |")

    # 2. Hardware Specs Table
    lines.append("\n## 2. Target Hardware Specifications")
    lines.append("| Parameter | Value | Notes |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| Per-Chip HBM-BW (GB/s) | {hw['hbm_bandwidth_gb_s']*2:.1f} | Global chip HBM buffer |")
    lines.append(f"| Per-Core HBM-BW (GB/s) | {hw['hbm_bandwidth_gb_s']:.1f} | Local core capacity |")
    lines.append(f"| Per-Chip TFLOPS (fp8) | {hw['peak_tflops_fp8']*2:.1f} | Double core peak compute |")
    lines.append(f"| Per-Core TFLOPS (fp8) | {hw['peak_tflops_fp8']:.1f} | Local core compute |")
    lines.append(f"| Per-Chip HBM Capacity (GB) | {hw['hbm_capacity_gb']*2:.1f} | Global buffer size |")
    lines.append(f"| Core-to-Core ICI (GB/s) | {hw['c2c_bandwidth_gb_s']:.1f} | Inter-core links |")

    # 3. Serving Workloads
    lines.append("\n## 3. Serving Workload Configuration")
    lines.append("| Configurable Parameter | Symbol | Value | Notes |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(f"| Batch Size | B | {B} | Workload batch size |")
    lines.append(f"| Query Length | T | {T} | Prompt / Query length |")
    lines.append(f"| KV Context Length | S | {S} | Context sequence tokens |")
    if regime == "decode":
        lines.append(f"| Generate Length | - | {gen_len} | Output generated tokens |")
        lines.append(f"| Token Step | - | {token_step} | Decode phase token |")

    # 4. Memory Occupancy Breakdown
    lines.append("\n## 4. Memory Occupancy Breakdown")
    lines.append("| Memory Object (Global) | Size (GiB) | Memory Object (Local) | Size (GiB) | Notes / Limit |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(f"| Attention Weights, Global | {res['attn_w_global']:.3f} | Attention Weights, Local | {res['attn_w_global']/num_devices:.3f} | Global Model size |")
    lines.append(f"| MoE Weights, Global | {res['moe_w_global']:.3f} | MoE Weights, Local | {res['moe_w_global']/num_devices:.3f} | Expert weight storage |")
    lines.append(f"| KV Cache Size, Global | {res['kv_cache_global']:.3f} | KV Cache Size, Local | {res['kv_cache_global']/num_devices:.3f} | Dynamic allocation |")
    lines.append(f"| **TOTAL (Global)** | **{res['attn_w_global'] + res['moe_w_global'] + res['kv_cache_global']:.3f}** | **{(res['attn_w_global'] + res['moe_w_global'] + res['kv_cache_global'])/num_devices + 2.0:.3f}** | **Limit: <= {hw['hbm_capacity_gb']*2} GiB** |")

    # 5. Top-Line Serving Metrics
    lines.append("\n## 5. Top-Line Serving Metrics")
    lines.append("| Metric Name | Value | Notes |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| # of Tokens | {B*T} | Batch size * Seq len |")
    lines.append(f"| E2E Latency | **{res['e2e_latency_ms']:.3f} ms** | Total layers latency |")
    lines.append(f"| Throughput | **{res['throughput_tok_s']:.2f} tok/s** | Generated tokens throughput |")
    lines.append(f"| Throughput/Chip | **{res['throughput_tok_s'] / (num_devices/2.0):.2f} tok/s/chip** | Normalized per accelerator |")

    # 6. Per-Layer Breakdown
    lines.append("\n## 6. Per-Layer Roofline Breakdown")
    lines.append("| Category | Op Name | Memory Time (ms) | Compute Time (ms) | ICI BW Time (ms) | Roofline Time (ms) | Boundness |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    
    if attention_type == "MLA":
        lines.append(f"| Attention | QKV_a Proj | {res['qkv_a']['hbm']:.4f} | {res['qkv_a']['comp']:.4f} | 0.0000 | {max(res['qkv_a']['hbm'], res['qkv_a']['comp']):.4f} | HBM |")
        lines.append(f"| Attention | Q_b Proj | {res['q_b']['hbm']:.4f} | {res['q_b']['comp']:.4f} | 0.0000 | {max(res['q_b']['hbm'], res['q_b']['comp']):.4f} | HBM |")
    else:
        lines.append(f"| Attention | Q Proj | {res['q_proj']['hbm']:.4f} | {res['q_proj']['comp']:.4f} | 0.0000 | {max(res['q_proj']['hbm'], res['q_proj']['comp']):.4f} | HBM |")
        lines.append(f"| Attention | K Proj | {res['k_proj']['hbm']:.4f} | {res['k_proj']['comp']:.4f} | 0.0000 | {max(res['k_proj']['hbm'], res['k_proj']['comp']):.4f} | HBM |")
        if not k_eq_v:
            lines.append(f"| Attention | V Proj | {res['v_proj']['hbm']:.4f} | {res['v_proj']['comp']:.4f} | 0.0000 | {max(res['v_proj']['hbm'], res['v_proj']['comp']):.4f} | HBM |")

    lines.append(f"| Attention | Core Attention | {res['attn_core']['hbm']:.4f} | {res['attn_core']['comp']:.4f} | 0.0000 | {max(res['attn_core']['hbm'], res['attn_core']['comp']):.4f} | HBM |")
    lines.append(f"| Attention | Out Proj | {res['out_proj']['hbm']:.4f} | {res['out_proj']['comp']:.4f} | {res['out_proj']['ici']:.4f} | {max(res['out_proj']['hbm'], res['out_proj']['comp']) + res['out_proj']['ici']:.4f} | HBM |")
    
    if F_dense > 0:
        lines.append(f"| MLP | Dense MLP | {res['dense_mlp']['hbm']:.4f} | {res['dense_mlp']['comp']:.4f} | {res['dense_mlp']['ici']:.4f} | {res['dense_mlp']['total']:.4f} | HBM |")

    if E > 0:
        lines.append(f"| MoE | Shared Expert | {res['moe']['shared_hbm']:.4f} | {res['moe']['shared_comp']:.4f} | 0.0000 | {max(res['moe']['shared_hbm'], res['moe']['shared_comp']):.4f} | HBM |")
        lines.append(f"| MoE | Routed Experts | {res['moe']['routed_hbm']:.4f} | {res['moe']['routed_comp']:.4f} | {res['moe']['ici']:.4f} | {res['moe']['total']:.4f} | HBM |")

    # 7. Detailed Op Analysis: Projections & Shapes
    lines.append(f"\n## 7. Detailed Op Analysis: {attention_type} Attention Block")
    if attention_type == "MLA":
        lines.append("\n### MLA Attention Flow Diagram")
        lines.append("```mermaid")
        lines.append("graph TD")
        lines.append("    X[\"Input Hidden States: [B, T, D]\"] --> QKVa[\"QKV_a Proj: [D -> L_q + L_kv + d_rope]\"]")
        lines.append("    QKVa --> Latent[\"Compressed Latents\"]")
        lines.append("    Latent --> Qb[\"Q_b Proj: [L_q -> N_q * d_head]\"]")
        lines.append("    Latent --> KVRope[\"KV compressed + RoPE key\"]")
        lines.append("    Qb --> Q[\"Queries: [B, T, N_q, d_h]\"]")
        lines.append("    KVRope --> KVCache[\"KV Cache (absorbed format)\"]")
        lines.append("    Q --> MLA_Kernel[\"MLA Attention Core: Q_nope @ W_uk, Q@K^T, Softmax, Attn@V, Out@W_uv\"]")
        lines.append("    KVCache --> MLA_Kernel")
        lines.append("    MLA_Kernel --> OutProj[\"Out Proj: [N_q * d_v -> D]\"]")
        lines.append("    OutProj --> AR[\"All-Reduce (Tensor Parallel)\"]")
        lines.append("    AR --> FinalAttn[\"Attention Output\"]")
        lines.append("```\n")
        
        lines.append("\n### 7.1 QKV_a Projection")
        lines.append("| Tensor Role | Global Shape | Local Shape | Data Type | Size (MiB) |")
        lines.append("| --- | --- | --- | --- | --- |")
        lines.append(f"| LHS Input (x) | ({B}, {T}, {D}) | ({B//tp_size if B>=tp_size else B}, {T}, {D}) | bf16 | {mib(B*T*D, 'bf16')/tp_size:.3f} |")
        lines.append(f"| RHS Weights | ({D}, {q_lora + kv_lora + d_rope}) | ({D}, {q_lora + kv_lora + d_rope}) | fp8 | {mib(D * (q_lora+kv_lora+d_rope), 'fp8'):.2f} |")
        lines.append(f"\n*   **Compute Block**: Local FLOPS = {res['qkv_a']['flops']:.1e} | Compute Time = {res['qkv_a']['comp']:.4f} ms")
        lines.append(f"*   **Memory Block**: HBM Transferred = {res['qkv_a']['size']:.2f} MiB | HBM Time = {res['qkv_a']['hbm']:.4f} ms")

        lines.append("\n### 7.2 Q_b Projection")
        lines.append("| Tensor Role | Global Shape | Local Shape | Data Type | Size (MiB) |")
        lines.append("| --- | --- | --- | --- | --- |")
        lines.append(f"| LHS Input | ({B}, {T}, {q_lora}) | ({B//tp_size if B>=tp_size else B}, {T}, {q_lora}) | bf16 | {mib(B*T*q_lora, 'bf16')/tp_size:.3f} |")
        lines.append(f"| RHS Weights | ({q_lora}, {local_q_out * tp_size}) | ({q_lora}, {local_q_out}) | fp8 | {mib(q_lora*local_q_out, 'fp8'):.2f} |")
        lines.append(f"\n*   **Compute Block**: Local FLOPS = {res['q_b']['flops']:.1e} | Compute Time = {res['q_b']['comp']:.4f} ms")
        lines.append(f"*   **Memory Block**: HBM Transferred = {res['q_b']['size']:.2f} MiB | HBM Time = {res['q_b']['hbm']:.4f} ms")

    else:
        lines.append("\n### Standard Attention Flow Diagram")
        lines.append("```mermaid")
        lines.append("graph TD")
        lines.append("    X[\"Input Hidden States: [B, T, D]\"] --> QProj[\"Q Proj: [D -> N_q * d_h]\"]")
        lines.append("    X --> KProj[\"K Proj: [D -> N_kv * d_h]\"]")
        if not k_eq_v:
            lines.append("    X --> VProj[\"V Proj: [D -> N_kv * d_h]\"]")
        lines.append("    QProj --> Q[\"Queries: [B, T, N_q, d_h]\"]")
        lines.append("    KProj --> K[\"Keys: [B, S, N_kv, d_h]\"]")
        if not k_eq_v:
            lines.append("    VProj --> V[\"Values: [B, S, N_kv, d_h]\"]")
        else:
            lines.append("    KProj --> V[\"Values (K=V): [B, S, N_kv, d_h]\"]")
        lines.append("    Q --> AttnCore[\"Core Attention: Q @ K^T, Softmax, Attn @ V\"]")
        lines.append("    K --> AttnCore")
        lines.append("    V --> AttnCore")
        lines.append("    AttnCore --> OutProj[\"Out Proj: [N_q * d_h -> D]\"]")
        lines.append("    OutProj --> AR[\"All-Reduce (Tensor Parallel)\"]")
        lines.append("    AR --> FinalAttn[\"Attention Output\"]")
        lines.append("```\n")

        lines.append("\n### 7.1 Q Projection")
        lines.append("| Tensor Role | Global Shape | Local Shape | Data Type | Size (MiB) |")
        lines.append("| --- | --- | --- | --- | --- |")
        lines.append(f"| LHS Input (x) | ({B}, {T}, {D}) | ({B//tp_size if B>=tp_size else B}, {T}, {D}) | bf16 | {mib(B*T*D, 'bf16')/tp_size:.3f} |")
        lines.append(f"| RHS Weights | ({D}, {N_q * d_h}) | ({D}, {local_q_out}) | fp8 | {mib(D * local_q_out, 'fp8'):.2f} |")
        lines.append(f"\n*   **Compute Block**: Local FLOPS = {res['q_proj']['flops']:.1e} | Compute Time = {res['q_proj']['comp']:.4f} ms")
        lines.append(f"*   **Memory Block**: HBM Transferred = {res['q_proj']['size']:.2f} MiB | HBM Time = {res['q_proj']['hbm']:.4f} ms")

        lines.append("\n### 7.2 K Projection")
        lines.append("| Tensor Role | Global Shape | Local Shape | Data Type | Size (MiB) |")
        lines.append("| --- | --- | --- | --- | --- |")
        lines.append(f"| LHS Input (x) | ({B}, {T}, {D}) | ({B//tp_size if B>=tp_size else B}, {T}, {D}) | bf16 | {mib(B*T*D, 'bf16')/tp_size:.3f} |")
        lines.append(f"| RHS Weights | ({D}, {N_kv * d_h}) | ({D}, {local_kv_out}) | fp8 | {mib(D * local_kv_out, 'fp8'):.2f} |")
        lines.append(f"\n*   **Compute Block**: Local FLOPS = {res['k_proj']['flops']:.1e} | Compute Time = {res['k_proj']['comp']:.4f} ms")
        lines.append(f"*   **Memory Block**: HBM Transferred = {res['k_proj']['size']:.2f} MiB | HBM Time = {res['k_proj']['hbm']:.4f} ms")

    # MLP / MoE blocks
    if F_dense > 0:
        lines.append("\n## 8. Detailed Op Analysis: Dense MLP Layer")
        lines.append("| Tensor Role | Global Shape | Local Shape | Data Type | Size (MiB) |")
        lines.append("| --- | --- | --- | --- | --- |")
        lines.append(f"| Input (x) | ({B}, {T}, {D}) | ({B//tp_size if B>=tp_size else B}, {T}, {D}) | bf16 | {mib(B*T*D, 'bf16')/tp_size:.3f} |")
        lines.append(f"| RHS Fused Weights | ({D}, {F_dense*2}) | ({D}, {local_f_dense*2}) | fp8 | {mib(D * local_f_dense * 2, 'fp8'):.2f} |")
        lines.append(f"| Down Weights | ({F_dense}, {D}) | ({local_f_dense}, {D}) | fp8 | {mib(local_f_dense * D, 'fp8'):.2f} |")
        lines.append(f"\n*   **Compute Block**: Local FLOPS = {res['dense_mlp']['flops']:.1e} | Compute Time = {res['dense_mlp']['comp']:.4f} ms")
        lines.append(f"*   **Memory Block**: HBM Transferred = {res['dense_mlp']['size']:.2f} MiB | HBM Time = {res['dense_mlp']['hbm']:.4f} ms")
        lines.append(f"*   **ICI Communication**: All-Reduce Time = {res['dense_mlp']['ici']:.4f} ms")

    if E > 0:
        lines.append("\n## 9. Detailed Op Analysis: MoE Layer (Routed Experts)")
        lines.append("| Tensor Role | Global Shape | Local Shape | Data Type | Size (MiB) |")
        lines.append("| --- | --- | --- | --- | --- |")
        lines.append(f"| LHS Input (x) | ({B}, {top_k}, {D}) | ({B//tp_size if B>=tp_size else B}, {top_k}, {D}) | fp8 | {mib(B * top_k * D, 'fp8')/tp_size:.3f} |")
        lines.append(f"| RHS Weights | ({E}, {D}, {F_moe*3}) | ({E}, {D}, {F_moe*3}) | fp8 | {mib(E * D * F_moe * 3, 'fp8'):.2f} |")
        lines.append(f"\n*   **Compute Block**: Local FLOPS = {res['moe']['routed_comp'] * 10**9 * peak_tflops:.1e} | Compute Time = {res['moe']['routed_comp']:.4f} ms")
        lines.append(f"*   **Memory Block**: HBM Transferred = {res['moe']['routed_hbm'] * hbm_bw / 1000 * 1024**3:.2f} Bytes | HBM Time = {res['moe']['routed_hbm']:.4f} ms")

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Write to target files using the exact filename formats
    md_filepath = os.path.join(args.output_dir, md_filename)
    with open(md_filepath, 'w') as f:
        f.write("\n".join(lines))

    # Compile HTML
    escaped_md = "\n".join(lines).replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{model['name']} Roofline Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; background-color: #0b0f19; color: #d1d5db; }}
        .markdown-body h1 {{ font-size: 1.875rem; font-weight: 700; color: #fff; border-bottom: 1px solid #1f2937; padding-bottom: 0.5rem; margin-top: 2rem; }}
        .markdown-body h2 {{ font-size: 1.5rem; font-weight: 600; color: #f3f4f6; margin-top: 1.75rem; }}
        .markdown-body table {{ width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; margin-top: 1rem; }}
        .markdown-body th {{ background-color: #111827; color: #f3f4f6; padding: 0.75rem; border: 1px solid #1f2937; }}
        .markdown-body td {{ padding: 0.75rem; border: 1px solid #1f2937; color: #d1d5db; background-color: #111827/50%; }}
    </style>
</head>
<body class="p-8 max-w-6xl mx-auto">
    <div id="content" class="markdown-body"></div>
    <script>
        mermaid.initialize({{ startOnLoad: false, theme: 'dark' }});
        const markdown = `{escaped_md}`;
        document.getElementById('content').innerHTML = marked.parse(markdown);
        const codeBlocks = document.querySelectorAll('pre code.language-mermaid');
        codeBlocks.forEach((codeBlock) => {{
            const pre = codeBlock.parentElement;
            const div = document.createElement('div');
            div.className = 'mermaid';
            div.textContent = codeBlock.textContent;
            pre.replaceWith(div);
        }});
        mermaid.run();
    </script>
</body>
</html>
"""
    html_filepath = os.path.join(args.output_dir, html_filename)
    with open(html_filepath, 'w') as f:
        f.write(html_content)

    print(f"Reports successfully generated at: {html_filepath}")

if __name__ == "__main__":
    main()
