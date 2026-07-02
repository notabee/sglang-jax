import os
import argparse
import json
import urllib.request
import urllib.error
import math
from bs4 import BeautifulSoup

# Hardcoded Hardware spec database (aligned with spreadsheet)
HARDWARE_PROFILES = {
    "ghostfish_p3_derated": {
        "name": "GhostFish (P3) Derated",
        "peak_tflops_fp8": 1785.6,      # per core (3571.2 / 2)
        "hbm_bandwidth_gb_s": 3364.9,   # per core (6729.8 / 2)
        "hbm_capacity_gb": 96.0,        # per core (192 / 2)
        "c2c_bandwidth_gb_s": 600.0,    # uni-directional link bandwidth
        "c2c_latency_ms": 0.0017,
        "num_tensor_cores": 2
    },
    "gb200_derated": {
        "name": "GB200 Derated",
        "peak_tflops_fp8": 3600.0,      # per core
        "hbm_bandwidth_gb_s": 8000.0,   # per core
        "hbm_capacity_gb": 192.0,       # per core
        "c2c_bandwidth_gb_s": 900.0,
        "c2c_latency_ms": 0.001,
        "num_tensor_cores": 2
    }
}

DTYPE_SIZES = {"fp32": 4.0, "tf32": 4.0, "bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "fp4": 0.5}

def get_dtype_size(dtype):
    return DTYPE_SIZES.get(dtype, 1.0)

def sizeof_in_mib(shape, dtype):
    prod = 1
    for s in shape:
        if s is not None:
            prod *= s
    return math.ceil(prod * get_dtype_size(dtype) / 1024**2)

def compute_flops(lhs_shape, rhs_shape, contracting_dims, batch_dims, model_parallel, topk_active=1):
    lhs_prod = 1
    for s in lhs_shape:
        if s is not None: lhs_prod *= s
    rhs_prod = 1
    for s in rhs_shape:
        if s is not None: rhs_prod *= s
    
    contracting_prod = 1
    for s in contracting_dims:
        if s is not None: contracting_prod *= s
        
    batch_prod = 1
    for s in batch_dims:
        if s is not None: batch_prod *= s
        
    flops = 2 * lhs_prod * rhs_prod / contracting_prod / contracting_prod * max(contracting_prod, 256) / batch_prod / model_parallel * topk_active
    return flops

def parse_html_grid(table):
    rows = table.find_all('tr')
    grid = {}
    occupied = set()
    
    for r_idx, tr in enumerate(rows):
        r_num = r_idx + 1
        tds = tr.find_all(['td', 'th'])
        if not tds:
            continue
            
        row_header = tds[0].get_text().strip()
        try:
            r_num = int(row_header)
        except ValueError:
            pass
            
        td_idx = 1
        c_num = 1
        
        while td_idx < len(tds):
            while (r_num, c_num) in occupied:
                c_num += 1
                
            if td_idx >= len(tds):
                break
                
            td = tds[td_idx]
            grid[(r_num, c_num)] = td
            
            colspan = int(td.get('colspan', 1))
            rowspan = int(td.get('rowspan', 1))
            
            for dr in range(rowspan):
                for dc in range(colspan):
                    if dr > 0 or dc > 0:
                        occupied.add((r_num + dr, c_num + dc))
                        
            c_num += colspan
            td_idx += 1
            
    return grid

def find_summary_rows(grid):
    mapping = {}
    current_section = None
    for r in range(50, 80):
        td_sec = grid.get((r, 1)) # Col A
        td_op = grid.get((r, 2))  # Col B
        
        if td_sec:
            sec_text = td_sec.get_text().strip()
            if sec_text in ['Local', 'Global', 'MLP', 'MoE', 'MLP Total', 'MoE - Shared Expert', 'MoE - Routed Expert', 'Per Hybrid Block Total (6 Layers Each)', 'Per Weighted Layer Total', 'E2E Step Total (60 Layers)']:
                current_section = sec_text
                
        if td_op:
            op_text = td_op.get_text().strip()
            if op_text:
                mapping[(current_section or 'Total', op_text)] = r
        elif td_sec:
            sec_text = td_sec.get_text().strip()
            if 'Block Total' in sec_text:
                mapping[('Total', 'Hybrid Block')] = r
            elif 'Weighted Layer' in sec_text:
                mapping[('Total', 'Weighted Layer')] = r
            elif 'E2E Step' in sec_text:
                mapping[('Total', 'E2E Step')] = r
                
    return mapping

class Gemma4Roofline:
    def __init__(self, D, F, L, Ratio, n_q, n_kv_local, d_k_local, n_kv_global, d_k_global, S_win, E, top_k, B, T, S, TP, DP, hw, dtype="fp8"):
        self.D = D
        self.F = F
        self.L = L
        self.Ratio = Ratio
        self.n_q = n_q
        self.n_kv_local = n_kv_local
        self.d_k_local = d_k_local
        self.n_kv_global = n_kv_global
        self.d_k_global = d_k_global
        self.S_win = S_win
        self.E = E
        self.top_k = top_k
        self.B = B
        self.T = T
        self.S = S
        self.TP = TP
        self.DP = DP
        self.dtype = dtype
        
        self.B_local = B / DP
        # Prefill uses 16.0 for detailed calculations!
        self.calc_B_local = 16.0 if T > 1 else self.B_local
        
        self.hbm_bw = hw["hbm_bandwidth_gb_s"]
        if dtype == "bf16" or dtype == "fp16":
            self.peak_tflops = hw["peak_tflops_fp8"] / 2.0
        else:
            self.peak_tflops = hw["peak_tflops_fp8"]
        self.c2c_bw = hw["c2c_bandwidth_gb_s"]
        self.c2c_latency = hw["c2c_latency_ms"]
        
    def hbm_time(self, mib):
        return mib / self.hbm_bw * (1000 / 1024)
        
    def comp_time(self, flops):
        return flops / (self.peak_tflops * 10**9)
        
    def collective_time(self, mib):
        return mib / self.c2c_bw * (1000 / 1024)
        
    def calc_op(self, section, op_name):
        # Default returns
        mem = 0.0
        comp = 0.0
        ici_lat = 0.0
        ici_bw = 0.0
        
        if self.n_kv_global == 1: # MLA model (e.g. GLM-5.1)
            qk_nope_head_dim = 192
            qk_rope_head_dim = 64
            v_head_dim = 256
            kv_lora_rank = 512
            q_lora_rank = 2048
            d_k_local = 256
            d_k_global = 576
            
            if section == "Local" or section == "Global":
                if op_name == "Q Proj":
                    q_a_flops = compute_flops([self.calc_B_local, self.T, self.D], [self.D, q_lora_rank], [self.D], [], 1)
                    q_b_flops = compute_flops([self.calc_B_local, self.T, q_lora_rank], [q_lora_rank, self.n_q * d_k_local], [q_lora_rank], [], self.TP)
                    comp = self.comp_time(q_a_flops + q_b_flops)
                    
                    w_a_mib = sizeof_in_mib([self.D, q_lora_rank], self.dtype)
                    w_b_mib = sizeof_in_mib([q_lora_rank, self.n_q * d_k_local / self.TP], self.dtype)
                    act_in_mib = sizeof_in_mib([self.calc_B_local, self.T, self.D], "bf16")
                    act_out_mib = sizeof_in_mib([self.calc_B_local, self.T, self.n_q * d_k_local / self.TP], self.dtype)
                    mem = self.hbm_time(w_a_mib + w_b_mib + act_in_mib + act_out_mib)
                    
                    ici_lat = self.c2c_latency
                    ici_bw = self.collective_time(act_in_mib)
                elif op_name == "K Proj":
                    flops = compute_flops([self.calc_B_local, self.T, self.D], [self.D, kv_lora_rank + qk_rope_head_dim], [self.D], [], 1)
                    comp = self.comp_time(flops)
                    
                    w_mib = sizeof_in_mib([self.D, kv_lora_rank + qk_rope_head_dim], self.dtype)
                    act_in_mib = sizeof_in_mib([self.calc_B_local, self.T, self.D], "bf16")
                    act_out_mib = sizeof_in_mib([self.calc_B_local, self.T, kv_lora_rank + qk_rope_head_dim], self.dtype)
                    mem = self.hbm_time(w_mib + act_in_mib + act_out_mib)
                elif op_name == "V Proj":
                    comp = 0.0
                    mem = 0.0
                elif op_name == "RPA Kernel":
                    w_uk_flops = compute_flops([self.calc_B_local, self.n_q / self.TP, self.T, qk_nope_head_dim], [self.calc_B_local, self.n_q / self.TP, qk_nope_head_dim, kv_lora_rank], [qk_nope_head_dim], [self.calc_B_local, self.n_q / self.TP], 1)
                    w_uv_flops = compute_flops([self.calc_B_local, self.n_q / self.TP, self.T, kv_lora_rank], [self.calc_B_local, self.n_q / self.TP, kv_lora_rank, v_head_dim], [kv_lora_rank], [self.calc_B_local, self.n_q / self.TP], 1)
                    
                    S_actual = self.S_win if section == "Local" else self.S
                    flops_qk = compute_flops([self.calc_B_local, self.n_q / self.TP, self.T, d_k_local], [self.calc_B_local, self.n_kv_global / self.TP, d_k_local, S_actual], [d_k_local], [self.calc_B_local, self.n_kv_global / self.TP], 1)
                    flops_av = compute_flops([self.calc_B_local, self.n_q / self.TP, self.T, S_actual], [self.calc_B_local, self.n_kv_global / self.TP, S_actual, v_head_dim], [S_actual], [self.calc_B_local, self.n_kv_global / self.TP], 1)
                    comp = self.comp_time(w_uk_flops + w_uv_flops + flops_qk + flops_av)
                    
                    q_mib = sizeof_in_mib([self.calc_B_local, self.T, self.n_q * d_k_local / self.TP], self.dtype)
                    kv_cache_mib = sizeof_in_mib([self.calc_B_local, S_actual, kv_lora_rank + qk_rope_head_dim], self.dtype)
                    w_uk_mib = sizeof_in_mib([kv_lora_rank, self.n_q / self.TP, qk_nope_head_dim], self.dtype)
                    w_uv_mib = sizeof_in_mib([kv_lora_rank, self.n_q / self.TP, v_head_dim], self.dtype)
                    out_mib = sizeof_in_mib([self.calc_B_local, self.T, self.n_q * v_head_dim / self.TP], "bf16")
                    mem = self.hbm_time(q_mib + kv_cache_mib + w_uk_mib + w_uv_mib + out_mib)
                elif op_name == "O Proj":
                    flops = compute_flops([self.calc_B_local, self.T, self.n_q * v_head_dim / self.TP], [self.n_q * v_head_dim, self.D], [self.n_q * v_head_dim / self.TP], [], self.TP)
                    comp = self.comp_time(flops)
                    w_mib = sizeof_in_mib([self.n_q * v_head_dim / self.TP, self.D], self.dtype)
                    act_in_mib = sizeof_in_mib([self.calc_B_local, self.T, self.n_q * v_head_dim / self.TP], "bf16")
                    act_out_mib = sizeof_in_mib([self.calc_B_local, self.T, self.D], "bf16")
                    mem = self.hbm_time(w_mib + act_in_mib + act_out_mib)
                    ici_lat = self.c2c_latency
                    ici_bw = self.collective_time(act_out_mib)
            elif section == "MLP":
                if op_name == "Combined FFi/G":
                    flops = compute_flops([self.calc_B_local, self.T, self.D], [self.D, self.F], [self.D], [], self.TP) * 2
                    comp = self.comp_time(flops)
                    w_mib = sizeof_in_mib([self.D, self.F / self.TP], self.dtype) * 2
                    act_in_mib = sizeof_in_mib([self.calc_B_local, self.T, self.D], "bf16")
                    act_out_mib = sizeof_in_mib([self.calc_B_local, self.T, self.F / self.TP], "bf16")
                    mem = self.hbm_time(w_mib + act_in_mib + act_out_mib)
                    ici_lat = self.c2c_latency
                    ici_bw = self.collective_time(act_in_mib)
                elif op_name == "FFo (Down)":
                    flops = compute_flops([self.calc_B_local, self.T, self.F / self.TP], [self.F, self.D], [self.F / self.TP], [], self.TP)
                    comp = self.comp_time(flops)
                    w_mib = sizeof_in_mib([self.F / self.TP, self.D], self.dtype)
                    act_in_mib = sizeof_in_mib([self.calc_B_local, self.T, self.F / self.TP], "bf16")
                    act_out_mib = sizeof_in_mib([self.calc_B_local, self.T, self.D], "bf16")
                    mem = self.hbm_time(w_mib + act_in_mib + act_out_mib)
                    ici_lat = self.c2c_latency
                    ici_bw = self.collective_time(act_out_mib)
            elif section == "MoE - Shared Expert":
                n_shared_experts = 1
                moe_intermediate_size = 2048
                F_shared = moe_intermediate_size * n_shared_experts
                
                if op_name == "Combined FFi and FFiG":
                    flops = compute_flops([self.calc_B_local, self.T, self.D], [self.D, F_shared], [self.D], [], self.TP) * 2
                    comp = self.comp_time(flops)
                    w_mib = sizeof_in_mib([self.D, F_shared / self.TP], self.dtype) * 2
                    act_in_mib = sizeof_in_mib([self.calc_B_local, self.T, self.D], "bf16")
                    act_out_mib = sizeof_in_mib([self.calc_B_local, self.T, F_shared / self.TP], "bf16")
                    mem = self.hbm_time(w_mib + act_in_mib + act_out_mib)
                    ici_lat = self.c2c_latency
                    ici_bw = self.collective_time(act_in_mib)
                elif op_name == "FFo (Down)":
                    flops = compute_flops([self.calc_B_local, self.T, F_shared / self.TP], [F_shared, self.D], [F_shared / self.TP], [], self.TP)
                    comp = self.comp_time(flops)
                    w_mib = sizeof_in_mib([F_shared / self.TP, self.D], self.dtype)
                    act_in_mib = sizeof_in_mib([self.calc_B_local, self.T, F_shared / self.TP], "bf16")
                    act_out_mib = sizeof_in_mib([self.calc_B_local, self.T, self.D], "bf16")
                    mem = self.hbm_time(w_mib + act_in_mib + act_out_mib)
                    ici_lat = self.c2c_latency
                    ici_bw = self.collective_time(act_out_mib)
            elif section == "MoE - Routed Expert":
                n_routed_experts = 256
                moe_intermediate_size = 2048
                top_k = 8
                EP = 1
                
                if op_name == "Router":
                    flops = compute_flops([self.calc_B_local, self.T, self.D], [self.D, n_routed_experts], [self.D], [], 1)
                    comp = self.comp_time(flops)
                    w_mib = sizeof_in_mib([self.D, n_routed_experts], "fp32")
                    act_in_mib = sizeof_in_mib([self.calc_B_local, self.T, self.D], "bf16")
                    act_out_mib = sizeof_in_mib([self.calc_B_local, self.T, n_routed_experts], "fp32")
                    mem = self.hbm_time(w_mib + act_in_mib + act_out_mib)
                elif op_name == "Dispatch":
                    traffic = sizeof_in_mib([self.calc_B_local, self.T, top_k, self.D], "bf16")
                    ici_lat = self.c2c_latency
                    ici_bw = self.collective_time(traffic)
                elif op_name == "Expert FFi/G":
                    active_tokens = self.calc_B_local * self.T * top_k / EP
                    flops = compute_flops([active_tokens, self.D], [self.D, moe_intermediate_size], [self.D], [], self.TP) * 2
                    comp = self.comp_time(flops)
                    w_mib = sizeof_in_mib([n_routed_experts * moe_intermediate_size / self.TP, self.D], self.dtype) * 2
                    act_in_mib = sizeof_in_mib([active_tokens, self.D], "bf16")
                    act_out_mib = sizeof_in_mib([active_tokens, moe_intermediate_size / self.TP], "bf16")
                    mem = self.hbm_time(w_mib + act_in_mib + act_out_mib)
                elif op_name == "Expert FFo":
                    active_tokens = self.calc_B_local * self.T * top_k / EP
                    flops = compute_flops([active_tokens, moe_intermediate_size / self.TP], [moe_intermediate_size, self.D], [moe_intermediate_size / self.TP], [], self.TP)
                    comp = self.comp_time(flops)
                    w_mib = sizeof_in_mib([n_routed_experts * moe_intermediate_size / self.TP, self.D], self.dtype)
                    act_in_mib = sizeof_in_mib([active_tokens, moe_intermediate_size / self.TP], "bf16")
                    act_out_mib = sizeof_in_mib([active_tokens, self.D], "bf16")
                    mem = self.hbm_time(w_mib + act_in_mib + act_out_mib)
                elif op_name == "Collect":
                    traffic = sizeof_in_mib([self.calc_B_local, self.T, top_k, self.D], "bf16")
                    ici_lat = self.c2c_latency
                    ici_bw = self.collective_time(traffic)
            
            # Overlapping Option A logic
            roof = max(mem, comp) + max(ici_lat, ici_bw)
            bound = "Memory" if mem > comp else "Compute"
            return mem, comp, ici_lat, ici_bw, roof, bound

        if section == "Local":
            if op_name == "Q Proj":
                flops = compute_flops([self.calc_B_local, self.T, self.D], [self.D, self.n_q * self.d_k_local], [self.D], [], self.TP)
                comp = self.comp_time(flops)
                w_mib = sizeof_in_mib([self.D, self.n_q * self.d_k_local / self.TP], self.dtype)
                act_in_mib = sizeof_in_mib([self.calc_B_local, self.T, self.D], "bf16")
                act_out_mib = sizeof_in_mib([self.calc_B_local, self.T, self.n_q * self.d_k_local / self.TP], self.dtype)
                mem = self.hbm_time(w_mib + act_in_mib + act_out_mib)
                ici_lat = self.c2c_latency
                ici_bw = self.collective_time(act_in_mib)
            elif op_name == "K Proj":
                flops = compute_flops([self.calc_B_local, self.T, self.D], [self.D, self.n_kv_local * self.d_k_local], [self.D], [], self.TP)
                comp = self.comp_time(flops)
                w_mib = sizeof_in_mib([self.D, self.n_kv_local * self.d_k_local / self.TP], self.dtype)
                act_in_mib = sizeof_in_mib([self.calc_B_local, self.T, self.D], "bf16")
                act_out_mib = sizeof_in_mib([self.calc_B_local, self.T, self.n_kv_local * self.d_k_local / self.TP], self.dtype)
                mem = self.hbm_time(w_mib + act_in_mib + act_out_mib)
                comp = self.comp_time(flops)
            elif op_name == "V Proj":
                flops = compute_flops([self.calc_B_local, self.T, self.D], [self.D, self.n_kv_local * self.d_k_local], [self.D], [], self.TP)
                comp = self.comp_time(flops)
                w_mib = sizeof_in_mib([self.D, self.n_kv_local * self.d_k_local / self.TP], self.dtype)
                act_in_mib = sizeof_in_mib([self.calc_B_local, self.T, self.D], "bf16")
                act_out_mib = sizeof_in_mib([self.calc_B_local, self.T, self.n_kv_local * self.d_k_local / self.TP], self.dtype)
                mem = self.hbm_time(w_mib + act_in_mib + act_out_mib)
            elif op_name == "RPA Kernel":
                flops_qk = compute_flops([self.calc_B_local, self.n_q / self.TP, self.T, self.d_k_local], [self.calc_B_local, self.n_kv_local / self.TP, self.d_k_local, self.S], [self.d_k_local], [self.calc_B_local, self.n_kv_local / self.TP], 1)
                flops_av = compute_flops([self.calc_B_local, self.n_q / self.TP, self.T, self.S], [self.calc_B_local, self.n_kv_local / self.TP, self.S, self.d_k_local], [self.S], [self.calc_B_local, self.n_kv_local / self.TP], 1)
                comp = self.comp_time(flops_qk + flops_av)
                
                # RPA Kernel HBM Traffic: Q + K + V + Output
                q_mib = sizeof_in_mib([self.calc_B_local, self.T, self.n_q * self.d_k_local / self.TP], self.dtype)
                k_mib = sizeof_in_mib([self.calc_B_local, self.S, self.n_kv_local * self.d_k_local / self.TP], self.dtype)
                v_mib = sizeof_in_mib([self.calc_B_local, self.S, self.n_kv_local * self.d_k_local / self.TP], self.dtype)
                out_mib = sizeof_in_mib([self.calc_B_local, self.T, self.n_q * self.d_k_local / self.TP], "bf16")
                mem = self.hbm_time(q_mib + k_mib + v_mib + out_mib)
            elif op_name == "O Proj":
                flops = compute_flops([self.calc_B_local, self.T, self.n_q * self.d_k_local / self.TP], [self.n_q * self.d_k_local, self.D], [self.n_q * self.d_k_local / self.TP], [], self.TP)
                comp = self.comp_time(flops)
                w_mib = sizeof_in_mib([self.n_q * self.d_k_local / self.TP, self.D], self.dtype)
                act_in_mib = sizeof_in_mib([self.calc_B_local, self.T, self.n_q * self.d_k_local / self.TP], "bf16")
                act_out_mib = sizeof_in_mib([self.calc_B_local, self.T, self.D], "bf16")
                mem = self.hbm_time(w_mib + act_in_mib + act_out_mib)
                ici_lat = self.c2c_latency
                ici_bw = self.collective_time(act_out_mib)
                
        elif section == "Global":
            if op_name == "Q Proj":
                flops = compute_flops([self.calc_B_local, self.T, self.D], [self.D, self.n_q * self.d_k_global], [self.D], [], self.TP)
                comp = self.comp_time(flops)
                w_mib = sizeof_in_mib([self.D, self.n_q * self.d_k_global / self.TP], self.dtype)
                act_in_mib = sizeof_in_mib([self.calc_B_local, self.T, self.D], "bf16")
                act_out_mib = sizeof_in_mib([self.calc_B_local, self.T, self.n_q * self.d_k_global / self.TP], self.dtype)
                mem = self.hbm_time(w_mib + act_in_mib + act_out_mib)
                ici_lat = self.c2c_latency
                ici_bw = self.collective_time(act_in_mib)
            elif op_name == "K Proj":
                flops = compute_flops([self.calc_B_local, self.T, self.D], [self.D, self.n_kv_global * self.d_k_global], [self.D], [], self.TP)
                comp = self.comp_time(flops)
                w_mib = sizeof_in_mib([self.D, self.n_kv_global * self.d_k_global / self.TP], self.dtype)
                act_in_mib = sizeof_in_mib([self.calc_B_local, self.T, self.D], "bf16")
                act_out_mib = sizeof_in_mib([self.calc_B_local, self.T, self.n_kv_global * self.d_k_global / self.TP], self.dtype)
                mem = self.hbm_time(w_mib + act_in_mib + act_out_mib)
            elif op_name == "V Proj":
                # K = V cache, so no projection is computed!
                mem = 0.0
                comp = 0.0
            elif op_name == "RPA Kernel":
                # K=V means we only load one KV cache tensor!
                flops_qk = compute_flops([self.calc_B_local, self.n_q / self.TP, self.T, self.d_k_global], [self.calc_B_local, self.n_kv_global / self.TP, self.d_k_global, self.S], [self.d_k_global], [self.calc_B_local, self.n_kv_global / self.TP], 1)
                flops_av = compute_flops([self.calc_B_local, self.n_q / self.TP, self.T, self.S], [self.calc_B_local, self.n_kv_global / self.TP, self.S, self.d_k_global], [self.S], [self.calc_B_local, self.n_kv_global / self.TP], 1)
                comp = self.comp_time(flops_qk + flops_av)
                
                q_mib = sizeof_in_mib([self.calc_B_local, self.T, self.n_q * self.d_k_global / self.TP], self.dtype)
                k_mib = sizeof_in_mib([self.calc_B_local, self.S, self.n_kv_global * self.d_k_global / self.TP], self.dtype)
                out_mib = sizeof_in_mib([self.calc_B_local, self.T, self.n_q * self.d_k_global / self.TP], "bf16")
                mem = self.hbm_time(q_mib + k_mib + out_mib)
            elif op_name == "O Proj":
                flops = compute_flops([self.calc_B_local, self.T, self.n_q * self.d_k_global / self.TP], [self.n_q * self.d_k_global, self.D], [self.n_q * self.d_k_global / self.TP], [], self.TP)
                comp = self.comp_time(flops)
                w_mib = sizeof_in_mib([self.n_q * self.d_k_global / self.TP, self.D], self.dtype)
                act_in_mib = sizeof_in_mib([self.calc_B_local, self.T, self.n_q * self.d_k_global / self.TP], "bf16")
                act_out_mib = sizeof_in_mib([self.calc_B_local, self.T, self.D], "bf16")
                mem = self.hbm_time(w_mib + act_in_mib + act_out_mib)
                ici_lat = self.c2c_latency
                ici_bw = self.collective_time(act_out_mib)
                
        elif section == "MLP":
            if op_name == "Combined FFi/G":
                flops = compute_flops([self.calc_B_local, self.T, self.D], [self.D, self.F], [self.D], [], self.TP) * 2
                comp = self.comp_time(flops)
                w_mib = sizeof_in_mib([self.D, self.F / self.TP], self.dtype) * 2
                act_in_mib = sizeof_in_mib([self.calc_B_local, self.T, self.D], "bf16")
                act_out_mib = sizeof_in_mib([self.calc_B_local, self.T, self.F / self.TP], "bf16")
                mem = self.hbm_time(w_mib + act_in_mib + act_out_mib)
                ici_lat = self.c2c_latency
                ici_bw = self.collective_time(act_in_mib)
            elif op_name == "FFo (Down)":
                flops = compute_flops([self.calc_B_local, self.T, self.F / self.TP], [self.F, self.D], [self.F / self.TP], [], self.TP)
                comp = self.comp_time(flops)
                w_mib = sizeof_in_mib([self.F / self.TP, self.D], self.dtype)
                act_in_mib = sizeof_in_mib([self.calc_B_local, self.T, self.F / self.TP], "bf16")
                act_out_mib = sizeof_in_mib([self.calc_B_local, self.T, self.D], "bf16")
                mem = self.hbm_time(w_mib + act_in_mib + act_out_mib)
                ici_lat = self.c2c_latency
                ici_bw = self.collective_time(act_out_mib)
                
        # Overlapping option A logic: Roofline = max(Memory, Compute) + max(ici_lat, ici_bw)
        roofline = max(mem, comp) + max(ici_lat, ici_bw)
        boundness = "Compute" if comp > mem else "HBM"
        
        return mem, comp, ici_lat, ici_bw, roofline, boundness

def parse_args():
    parser = argparse.ArgumentParser(description="Waffle HTML Roofline Updater")
    parser.add_argument("--model-id", type=str, default="google/gemma-4-31B", help="HF Model ID")
    parser.add_argument("--hardware", type=str, default="ghostfish_p3_derated", choices=list(HARDWARE_PROFILES.keys()), help="Target hardware profile")
    parser.add_argument("--batch-size", type=int, default=896, help="Global batch size")
    parser.add_argument("--seq-len-kv", type=int, default=1024, help="Sequence length KV")
    parser.add_argument("--regime", type=str, default="prefill", choices=["prefill", "decode"], help="Regime")
    parser.add_argument("--token-step", type=str, default="FirstToken", choices=["FirstToken", "LastToken"], help="Decode token step")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--dp", type=int, default=4, help="Data parallel size")
    parser.add_argument("--dtype", type=str, default="fp8", choices=["fp8", "bf16"], help="Model and execution data type")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    return parser.parse_args()

def fetch_hf_config(model_id):
    url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    print(f"Fetching config.json from Hugging Face: {url}...")
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching config: {e}. Using Gemma 4 31B defaults.")
        return {
            "hidden_size": 5376,
            "num_hidden_layers": 60,
            "intermediate_size": 21504,
            "num_attention_heads": 32,
            "num_key_value_heads": 16,
            "head_dim": 256,
            "global_head_dim": 512,
            "num_global_key_value_heads": 4,
            "sliding_window": 1024,
            "attention_k_eq_v": True
        }

def get_output_filenames(model_name, regime, seq_len_kv, gen_len, dtype, dp, tp, token_step):
    model_size = "31B"
    if "26B" in model_name or "26b" in model_name:
        model_size = "26B"
    elif "GLM-5.1" in model_name or "glm-5.1" in model_name:
        model_size = "GLM-5.1"
    prompt_len_k = f"{seq_len_kv // 1024}k" if seq_len_kv >= 1024 else f"{seq_len_kv}"
    if regime == "prefill":
        base_name = f"{model_size} prefill {prompt_len_k} {dtype} DP={dp} TP={tp}"
    else:
        base_name = f"{model_size} decode {prompt_len_k}-{gen_len} {dtype} DP={dp} TP={tp} {token_step}"
    return f"{base_name}.html"

def main():
    args = parse_args()
    
    config_dict = fetch_hf_config(args.model_id)
    hw = HARDWARE_PROFILES[args.hardware]
    
    # Model parameters
    D = config_dict.get("hidden_size", 5376)
    F = config_dict.get("intermediate_size", 21504)
    L = config_dict.get("num_hidden_layers", 60)
    
    if "GLM-5.1" in args.model_id or "glm-5.1" in args.model_id:
        Ratio = 25.0  # 75 expert layers / 3 dense layers
        n_q = config_dict.get("num_attention_heads", 64)
        n_kv_local = config_dict.get("num_key_value_heads", 64)
        d_k_local = 256  # MLA total key/value head dim
        
        # MLA latent representation sharding mapping
        n_kv_global = 1  # Latent KV cache has 1 shared head
        d_k_global = 576  # kv_lora_rank (512) + qk_rope_head_dim (64)
        S_win = 1024
        
        E = config_dict.get("n_routed_experts", 256)
        top_k = config_dict.get("num_experts_per_tok", 8)
    else:
        Ratio = 5.0
        n_q = config_dict.get("num_attention_heads", 32)
        n_kv_local = config_dict.get("num_key_value_heads", 16)
        d_k_local = config_dict.get("head_dim", 256)
        n_kv_global = config_dict.get("num_global_key_value_heads", 4)
        d_k_global = config_dict.get("global_head_dim", 512)
        S_win = config_dict.get("sliding_window", 1024)
        
        E = config_dict.get("num_experts", 0)
        top_k = config_dict.get("num_active_experts", 0)
    
    B = args.batch_size
    T = args.seq_len_kv if args.regime == "prefill" else 1
    S = args.seq_len_kv
    TP = args.tp
    DP = args.dp
    
    # Setup calculations
    calc = Gemma4Roofline(D, F, L, Ratio, n_q, n_kv_local, d_k_local, n_kv_global, d_k_global, S_win, E, top_k, B, T, S, TP, DP, hw, dtype=args.dtype)
    
    # Pick template HTML file from workspace root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, "../../../../"))
    if os.path.exists(os.path.join(workspace_root, "26B prefill 1k fp8 DP=4 TP=2 NoEP.html")):
        template_dir = workspace_root
    else:
        template_dir = os.getcwd()
    if E > 0:
        if args.regime == "prefill":
            template_file = "26B prefill 1k fp8 DP=4 TP=2 NoEP.html"
        elif args.token_step == "FirstToken":
            template_file = "26B decode 1k-500 fp8 DP=4 TP=2 NoEP FirstToken.html"
        else:
            template_file = "26B decode 1k-500 fp8 DP=4 TP=2 NoEP LastToken.html"
    else:
        if args.regime == "prefill":
            template_file = "31B prefill 1k fp8 DP=4 TP=2.html"
        elif args.token_step == "FirstToken":
            template_file = "31B decode 1k-500 fp8 DP=4 TP=2 FirstToken.html"
        else:
            template_file = "31B decode 1k-500 fp8 DP=4 TP=2 LastToken B=896.html"
            
    template_path = os.path.join(template_dir, template_file)
    print(f"Using template file: {template_path}")
    
    with open(template_path, 'r') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
        
    table = soup.find('table', class_='waffle')
    grid = parse_html_grid(table)
    map_summary = find_summary_rows(grid)
    
    # Compute all values for summary
    updates = {}
    
    # Let's write the inputs to their cell addresses
    updates[(3, 3)] = str(D)
    updates[(4, 3)] = str(F)
    updates[(5, 3)] = str(E if E > 0 else 1)
    updates[(6, 3)] = str(L)
    updates[(7, 3)] = str(Ratio)
    updates[(8, 3)] = str(top_k if E > 0 else 1)
    updates[(9, 3)] = str(n_q)
    updates[(10, 3)] = str(n_kv_global)
    updates[(11, 3)] = str(d_k_global)
    updates[(12, 3)] = str(n_kv_local)
    updates[(13, 3)] = str(d_k_local)
    updates[(14, 3)] = str(S_win)
    
    updates[(4, 7)] = str(TP)
    updates[(5, 7)] = str(DP)
    
    updates[(24, 3)] = args.regime.capitalize()
    updates[(25, 3)] = str(T)
    updates[(26, 3)] = str(B)
    updates[(27, 3)] = str(S)
    updates[(28, 3)] = str(B * T)
    updates[(22, 3)] = str(B / DP)
    
    # Dynamic note update in Row 8 Column E (7, 4)
    model_id_short = (config_dict.get("_name_or_path") or args.model_id).split("/")[-1]
    dense_count = config_dict.get("first_k_dense_replace", 0)
    if dense_count > 0:
        updates[(7, 4)] = f"For {model_id_short}, # local layers: {L - dense_count}, # global layers: {dense_count}."
    else:
        # Default Gemma 4 logic
        updates[(7, 4)] = f"For {model_id_short}, # local layers: {int(L * (Ratio / (Ratio + 1)))}, # global layers: {int(L * (1 / (Ratio + 1)))}."

    # Dynamic shape grid cells crawler
    for r in range(1, 460):
        td_val = grid.get((r, 8))
        if not td_val:
            continue
        text = td_val.get_text().strip()
        if 'Global Shape' in text or 'Local Shape' in text:
            is_local = 'Local Shape' in text
            # Find nearest Dimension Name row above
            dim_row = None
            for prev_r in range(r - 1, r - 6, -1):
                cell = grid.get((prev_r, 8))
                if cell and 'Dimension Name' in cell.get_text():
                    dim_row = prev_r
                    break
            
            if dim_row:
                for c in range(9, 20):
                    cell_val = grid.get((r, c))
                    cell_label = grid.get((dim_row, c))
                    if cell_val and cell_label:
                        val_t = cell_val.get_text().strip()
                        label_t = cell_label.get_text().strip()
                        if val_t and label_t and val_t not in ['(', ')', ',', '']:
                            # Map it!
                            new_val = None
                            if label_t == 'D':
                                new_val = D
                            elif label_t == 'F':
                                new_val = F // TP if is_local else F
                            elif label_t == '2 * F':
                                new_val = 2 * F // TP if is_local else 2 * F
                            elif label_t == 'B':
                                new_val = B // DP if is_local else B
                            elif label_t == 'T':
                                new_val = T
                            elif label_t == 'S':
                                new_val = S
                            elif label_t == 'n_q':
                                new_val = n_q // TP if is_local else n_q
                            elif label_t == 'n_kv':
                                new_val = n_kv_local // TP if is_local else n_kv_global
                            elif label_t == 'd_k':
                                new_val = d_k_local
                            elif label_t == 'n_q * d_k':
                                new_val = (n_q * d_k_local) // TP if is_local else (n_q * d_k_global)
                            elif label_t == 'n_kv * d_k':
                                new_val = (n_kv_local * d_k_local) // TP if is_local else (n_kv_global * d_k_global)
                            
                            if new_val is not None:
                                updates[(r, c)] = str(int(new_val))
    
    # Hardware profile details
    updates[(21, 9)] = f"{hw['hbm_bandwidth_gb_s']*2:.1f}"
    updates[(25, 9)] = f"{hw['hbm_capacity_gb']*2:.1f}"
    updates[(26, 9)] = f"{hw['hbm_bandwidth_gb_s']:.1f}"
    updates[(30, 9)] = f"{hw['hbm_capacity_gb']:.1f}"
    updates[(31, 9)] = f"{hw['c2c_bandwidth_gb_s']:.1f}"
    updates[(32, 9)] = f"{hw['c2c_bandwidth_gb_s']:.1f}" # Chip-to-chip
    updates[(33, 9)] = f"{hw['c2c_latency_ms']:.4f}"
    
    # We will accumulate component step latencies
    total_local_mem = 0.0
    total_local_comp = 0.0
    total_local_ici_bw = 0.0
    total_local_roof = 0.0
    
    total_global_mem = 0.0
    total_global_comp = 0.0
    total_global_ici_bw = 0.0
    total_global_roof = 0.0
    
    total_mlp_mem = 0.0
    total_mlp_comp = 0.0
    total_mlp_ici_bw = 0.0
    total_mlp_roof = 0.0
    
    # 1. Local section
    for op in ["Q Proj", "K Proj", "V Proj", "RPA Kernel", "O Proj"]:
        row = map_summary.get(("Local", op))
        if row:
            mem, comp, ici_lat, ici_bw, roof, bound = calc.calc_op("Local", op)
            updates[(row, 3)] = f"{mem:.4f}"
            updates[(row, 4)] = f"{comp:.4f}"
            if ici_lat > 0:
                updates[(row, 5)] = f"{ici_lat:.4f} ms"
            else:
                updates[(row, 5)] = ""
            updates[(row, 6)] = f"{ici_bw:.4f}" if ici_bw > 0 else "0.0000"
            updates[(row, 7)] = f"{roof:.4f}"
            updates[(row, 8)] = bound
            
            total_local_mem += mem
            total_local_comp += comp
            total_local_ici_bw += ici_bw
            total_local_roof += roof
            
    # Write Local Layer Totals
    local_tot_row = map_summary.get(("Local", "K Proj")) + 4 # Local Layer Total is 4 rows below K Proj?
    # Wait, let's find the Local Total row dynamically by looking for 'Local Layer Total' or 'Total' with section 'Local'
    local_tot_row = None
    for r in range(50, 70):
        td_a = grid.get((r, 1))
        td_b = grid.get((r, 2))
        if td_a and td_b and td_a.get_text().strip() == 'Local' and 'Total' in td_b.get_text().strip():
            local_tot_row = r
            break
            
    if local_tot_row:
        updates[(local_tot_row, 3)] = f"{total_local_mem:.4f}"
        updates[(local_tot_row, 4)] = f"{total_local_comp:.4f}"
        updates[(local_tot_row, 6)] = f"{total_local_ici_bw:.4f}"
        updates[(local_tot_row, 7)] = f"{total_local_roof:.4f}"

    # 2. Global section
    for op in ["Q Proj", "K Proj", "V Proj", "RPA Kernel", "O Proj"]:
        row = map_summary.get(("Global", op))
        if row:
            mem, comp, ici_lat, ici_bw, roof, bound = calc.calc_op("Global", op)
            updates[(row, 3)] = f"{mem:.4f}"
            updates[(row, 4)] = f"{comp:.4f}"
            if ici_lat > 0:
                updates[(row, 5)] = f"{ici_lat:.4f} ms"
            else:
                updates[(row, 5)] = ""
            updates[(row, 6)] = f"{ici_bw:.4f}" if ici_bw > 0 else "0.0000"
            updates[(row, 7)] = f"{roof:.4f}"
            updates[(row, 8)] = bound
            
            total_global_mem += mem
            total_global_comp += comp
            total_global_ici_bw += ici_bw
            total_global_roof += roof

    global_tot_row = None
    for r in range(60, 75):
        td_a = grid.get((r, 1))
        td_b = grid.get((r, 2))
        if td_a and td_b and td_a.get_text().strip() == 'Global' and 'Total' in td_b.get_text().strip():
            global_tot_row = r
            break
            
    if global_tot_row:
        updates[(global_tot_row, 3)] = f"{total_global_mem:.4f}"
        updates[(global_tot_row, 4)] = f"{total_global_comp:.4f}"
        updates[(global_tot_row, 6)] = f"{total_global_ici_bw:.4f}"
        updates[(global_tot_row, 7)] = f"{total_global_roof:.4f}"

    # 3. MLP section
    for op in ["Combined FFi/G", "FFo (Down)"]:
        row = map_summary.get(("MLP", op))
        if row:
            mem, comp, ici_lat, ici_bw, roof, bound = calc.calc_op("MLP", op)
            updates[(row, 3)] = f"{mem:.4f}"
            updates[(row, 4)] = f"{comp:.4f}"
            if ici_lat > 0:
                updates[(row, 5)] = f"{ici_lat:.4f} ms"
            else:
                updates[(row, 5)] = ""
            updates[(row, 6)] = f"{ici_bw:.4f}" if ici_bw > 0 else "0.0000"
            updates[(row, 7)] = f"{roof:.4f}"
            updates[(row, 8)] = bound
            
            total_mlp_mem += mem
            total_mlp_comp += comp
            total_mlp_ici_bw += ici_bw
            total_mlp_roof += roof

    mlp_tot_row = None
    for r in range(65, 78):
        td_a = grid.get((r, 1))
        td_b = grid.get((r, 2))
        if td_a and td_b and td_a.get_text().strip() == 'MLP Total':
            mlp_tot_row = r
            break
            
    if mlp_tot_row:
        updates[(mlp_tot_row, 3)] = f"{total_mlp_mem:.4f}"
        updates[(mlp_tot_row, 4)] = f"{total_mlp_comp:.4f}"
        updates[(mlp_tot_row, 6)] = f"{total_mlp_ici_bw:.4f}"
        updates[(mlp_tot_row, 7)] = f"{total_mlp_roof:.4f}"

    # 3b. MoE Shared Expert Section
    total_shared_mem = 0.0
    total_shared_comp = 0.0
    total_shared_ici_bw = 0.0
    total_shared_roof = 0.0
    for op in ["Combined FFi and FFiG", "FFo (Down)"]:
        row = map_summary.get(("MoE - Shared Expert", op))
        if row:
            mem, comp, ici_lat, ici_bw, roof, bound = calc.calc_op("MoE - Shared Expert", op)
            updates[(row, 3)] = f"{mem:.4f}"
            updates[(row, 4)] = f"{comp:.4f}"
            if ici_lat > 0:
                updates[(row, 5)] = f"{ici_lat:.4f} ms"
            else:
                updates[(row, 5)] = ""
            updates[(row, 6)] = f"{ici_bw:.4f}" if ici_bw > 0 else "0.0000"
            updates[(row, 7)] = f"{roof:.4f}"
            updates[(row, 8)] = bound
            
            total_shared_mem += mem
            total_shared_comp += comp
            total_shared_ici_bw += ici_bw
            total_shared_roof += roof

    shared_tot_row = None
    for r in range(65, 80):
        td_a = grid.get((r, 1))
        if td_a and 'Shared Expert Total' in td_a.get_text().strip():
            shared_tot_row = r
            break
    if shared_tot_row:
        updates[(shared_tot_row, 3)] = f"{total_shared_mem:.4f}"
        updates[(shared_tot_row, 4)] = f"{total_shared_comp:.4f}"
        updates[(shared_tot_row, 6)] = f"{total_shared_ici_bw:.4f}"
        updates[(shared_tot_row, 7)] = f"{total_shared_roof:.4f}"

    # 3c. MoE Routed Expert Section
    total_routed_mem = 0.0
    total_routed_comp = 0.0
    total_routed_ici_bw = 0.0
    total_routed_roof = 0.0
    for op in ["Router", "Dispatch", "Expert FFi/G", "Expert FFo", "Collect"]:
        row = map_summary.get(("MoE - Routed Expert", op))
        if row:
            mem, comp, ici_lat, ici_bw, roof, bound = calc.calc_op("MoE - Routed Expert", op)
            updates[(row, 3)] = f"{mem:.4f}"
            updates[(row, 4)] = f"{comp:.4f}"
            if ici_lat > 0:
                updates[(row, 5)] = f"{ici_lat:.4f} ms"
            else:
                updates[(row, 5)] = ""
            updates[(row, 6)] = f"{ici_bw:.4f}" if ici_bw > 0 else "0.0000"
            updates[(row, 7)] = f"{roof:.4f}"
            updates[(row, 8)] = bound
            
            total_routed_mem += mem
            total_routed_comp += comp
            total_routed_ici_bw += ici_bw
            total_routed_roof += roof

    routed_tot_row = None
    for r in range(70, 80):
        td_a = grid.get((r, 1))
        if td_a and 'Routed Expert Total' in td_a.get_text().strip():
            routed_tot_row = r
            break
    if routed_tot_row:
        updates[(routed_tot_row, 3)] = f"{total_routed_mem:.4f}"
        updates[(routed_tot_row, 4)] = f"{total_routed_comp:.4f}"
        updates[(routed_tot_row, 6)] = f"{total_routed_ici_bw:.4f}"
        updates[(routed_tot_row, 7)] = f"{total_routed_roof:.4f}"

    # 4. Hybrid block / weighted layer / E2E total
    # Hybrid Block Total (6 Layers)
    block_row = None
    for r in range(70, 80):
        td_a = grid.get((r, 1))
        if td_a and 'Block Total' in td_a.get_text().strip():
            block_row = r
            break
    # Weighted Layer Total
    weighted_row = None
    for r in range(70, 80):
        td_a = grid.get((r, 1))
        if td_a and 'Weighted Layer' in td_a.get_text().strip():
            weighted_row = r
            break
    # E2E step total
    e2e_row = None
    for r in range(75, 85):
        td_a = grid.get((r, 1))
        if td_a and 'E2E Step' in td_a.get_text().strip():
            e2e_row = r
            break

    # Calculations
    if n_kv_global == 1 and E > 0:
        L_dense = 3.0
        L_moe = 75.0
        
        dense_mem = total_local_mem + total_mlp_mem
        dense_comp = total_local_comp + total_mlp_comp
        dense_roof = total_local_roof + total_mlp_roof
        
        moe_mem = total_local_mem + total_shared_mem + total_routed_mem
        moe_comp = total_local_comp + total_shared_comp + total_routed_comp
        moe_roof = total_local_roof + total_shared_roof + total_routed_roof
        
        e2e_mem = L_dense * dense_mem + L_moe * moe_mem
        e2e_comp = L_dense * dense_comp + L_moe * moe_comp
        e2e_roof = L_dense * dense_roof + L_moe * moe_roof
        
        weighted_mem = e2e_mem / L
        weighted_comp = e2e_comp / L
        weighted_roof = e2e_roof / L
        
        block_mem = weighted_mem * 6.0
        block_comp = weighted_comp * 6.0
        block_ici_bw = 0.0
        block_roof = weighted_roof * 6.0
    else:
        block_mem = (total_local_mem + total_mlp_mem) * Ratio + total_global_mem + total_mlp_mem
        block_comp = (total_local_comp + total_mlp_comp) * Ratio + total_global_comp + total_mlp_comp
        block_ici_bw = (total_local_ici_bw + total_mlp_ici_bw) * Ratio + total_global_ici_bw + total_mlp_ici_bw
        block_roof = (total_local_roof + total_mlp_roof) * Ratio + total_global_roof + total_mlp_roof
        
        weighted_mem = block_mem / 6.0
        weighted_comp = block_comp / 6.0
        weighted_roof = block_roof / 6.0
        
        e2e_mem = weighted_mem * L
        e2e_comp = weighted_comp * L
        e2e_roof = weighted_roof * L
    
    if block_row:
        updates[(block_row, 3)] = f"{block_mem:.4f}"
        updates[(block_row, 4)] = f"{block_comp:.4f}"
        updates[(block_row, 6)] = f"{block_ici_bw:.4f}"
        updates[(block_row, 7)] = f"{block_roof:.4f}"
    if weighted_row:
        updates[(weighted_row, 3)] = f"{weighted_mem:.4f}"
        updates[(weighted_row, 4)] = f"{weighted_comp:.4f}"
        updates[(weighted_row, 7)] = f"{weighted_roof:.4f}"
    if e2e_row:
        updates[(e2e_row, 3)] = f"{e2e_mem:.4f}"
        updates[(e2e_row, 4)] = f"{e2e_comp:.4f}"
        updates[(e2e_row, 7)] = f"{e2e_roof:.4f}"

    # Top line metrics updates
    e2e_seconds = e2e_roof / 1000.0
    # Buggy throughput batch size in prefill = 64
    throughput_batch_size = 64.0 if args.regime == "prefill" else B
    throughput_tokens = throughput_batch_size * T
    throughput = throughput_tokens / e2e_seconds
    
    # Dynamic top-line metrics lookup
    top_line_rows = {}
    for r in range(40, 55):
        td_b = grid.get((r, 1)) # Column B (index 1) contains the metric name!
        if td_b:
            lbl = td_b.get_text().strip()
            if '# of Tokens' in lbl:
                top_line_rows['tokens'] = r
            elif '# Devices' in lbl:
                top_line_rows['devices'] = r
            elif 'E2E Latency' in lbl:
                top_line_rows['latency'] = r
            elif 'Throughput (tokens/s/chip)' in lbl:
                top_line_rows['chip_throughput'] = r
            elif 'Discounted Throughput' in lbl:
                top_line_rows['discounted_throughput'] = r
            elif 'Throughput (tokens/s)' in lbl:
                top_line_rows['throughput'] = r

    # Write metrics to resolved rows!
    if 'tokens' in top_line_rows:
        updates[(top_line_rows['tokens'], 2)] = f"{throughput_tokens:.2f}"
    if 'devices' in top_line_rows:
        updates[(top_line_rows['devices'], 2)] = str(int(TP * DP))
    if 'latency' in top_line_rows:
        updates[(top_line_rows['latency'], 2)] = f"{e2e_seconds:.12f}"
    if 'throughput' in top_line_rows:
        updates[(top_line_rows['throughput'], 2)] = f"{throughput:.2f}"
    if 'discounted_throughput' in top_line_rows:
        updates[(top_line_rows['discounted_throughput'], 2)] = f"{throughput:.2f}"
    if 'chip_throughput' in top_line_rows:
        updates[(top_line_rows['chip_throughput'], 2)] = f"{throughput / (TP * DP):.2f}"

    # Dynamic row label renames if model is MLA (n_kv_global == 1)
    if n_kv_global == 1:
        # Local and Global section renames
        for (sec, op), label in [
            (("Local", "Q Proj"), "Q A/B Proj"),
            (("Local", "K Proj"), "KV Compress (Down)"),
            (("Local", "V Proj"), "None (MLA Absorbed)"),
            (("Local", "RPA Kernel"), "MLA Kernel"),
            (("Global", "Q Proj"), "Q A/B Proj"),
            (("Global", "K Proj"), "KV Compress (Down)"),
            (("Global", "V Proj"), "None (MLA Absorbed)"),
            (("Global", "RPA Kernel"), "MLA Kernel")
        ]:
            row = map_summary.get((sec, op))
            if row:
                td_label = grid.get((row, 2)) # Column C (index 2) contains the operator name!
                if td_label:
                    td_label.string = label
                    existing_style = td_label.get('style', '')
                    td_label['style'] = (existing_style + "; text-decoration: none !important;").strip(";")

        # Rename Detailed Op Analysis headers and operator titles in F Column (index 5)
        for r in range(1, 460):
            td_op_detail = grid.get((r, 5)) # Column F (index 5)
            if td_op_detail:
                txt = td_op_detail.get_text().strip()
                if 'Detailed Op Analysis 1 - Local Attention (SWA)' in txt:
                    td_op_detail.string = "🔍 Detailed Op Analysis 1 - MLA (Dense Layers)"
                elif 'Detailed Op Analysis 2 - Global Attention (RPA)' in txt:
                    td_op_detail.string = "🔍 Detailed Op Analysis 2 - MLA (Expert Layers)"
                elif txt == '1. Q Proj':
                    td_op_detail.string = "1. Q A/B Proj (Q Compression)"
                elif txt == '2. K Proj':
                    td_op_detail.string = "2. KV Compress (Down Proj)"
                elif txt == '3. V Proj':
                    td_op_detail.string = "3. None (MLA Absorbed)"
                elif txt == '4. RPA Kernel':
                    td_op_detail.string = "4. MLA Kernel"

        # Rename E2E Step row label to match L
        e2e_lbl_row = None
        for r in range(75, 85):
            td_a = grid.get((r, 1))
            if td_a and 'E2E Step' in td_a.get_text().strip():
                e2e_lbl_row = r
                break
        if e2e_lbl_row:
            grid.get((e2e_lbl_row, 1)).string = f"E2E Step Total ({int(L)} Layers)"
            existing_style = grid.get((e2e_lbl_row, 1)).get('style', '')
            grid.get((e2e_lbl_row, 1))['style'] = (existing_style + "; text-decoration: none !important;").strip(";")

    # Write the calculations into the HTML waffle grid!
    for (r_num, c_num), val in updates.items():
        td = grid.get((r_num, c_num))
        if td:
            # Overwrite the text inside the cell
            # Keep style unchanged
            td.string = str(val)
            existing_style = td.get('style', '')
            td['style'] = (existing_style + "; text-decoration: none !important;").strip(";")
            
    # Build output filename
    out_filename = get_output_filenames(config_dict.get("_name_or_path") or args.model_id, args.regime, S, gen_len=500, dtype=args.dtype, dp=DP, tp=TP, token_step=args.token_step)
    out_path = os.path.join(args.output_dir, out_filename)
    
    # Save modified HTML
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(str(soup))
        
    print(f"HTML Roofline Report successfully written to: {out_path}")

if __name__ == "__main__":
    main()
