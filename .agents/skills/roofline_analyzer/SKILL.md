---
name: perform-roofline-analysis
description: Perform a theoretical performance roofline analysis for any ML model on target hardware, calculating memory bandwidth, compute FLOPS, and network collectives.
---

# Perform Roofline Analysis Skill

This skill allows the agent to analyze any model architecture configuration (e.g., MHA, MLA, SwiGLU, MoE) from a Hugging Face Model ID or configuration file, and compute its roofline performance (HBM, Compute, and ICI link bounds) against hardware targets.

The skill generates waffle-style HTML sheets (Google Sheets style HTML exports) that are 100% matched to the formatting and layouts in the workspace.

## 1. Supported Hardware Profiles
The skill has built-in performance profiles for target chips:
*   `ghostfish_p3_derated`: GhostFish (P3) Derated
*   `gb200_derated`: GB200 Derated

## 2. Dynamic Architecture Capabilities
*   **Attention Backends**: The tool automatically checks model configuration at runtime. If it detects `num_global_key_value_heads == 1` or an MLA model (e.g. GLM-5.1), it dynamically renames attention operators to `Q A/B Proj`, `KV Compress`, and `MLA Kernel` and uses MLA mathematical formulations.
*   **Dense / MoE Layer Splitting**: For models that are hybrid or have separate dense and expert layers (such as GLM-5.1's 3 dense layers and 75 MoE layers), the script sums up dense and MoE latencies separately using the correct layer counts.
*   **Coordinate-Agnostic Top-Line Metric Lookup**: Instead of hardcoding spreadsheet cells, it scans for metric labels like `E2E Latency (s)` or `Throughput (tokens/s)` and overwrites the values at their dynamically resolved locations.
*   **Style Stripping**: It appends `style="text-decoration: none !important;"` inline overrides to prevent line-through text decorations on updated fields.

## 3. CLI Argument Reference

| Flag | Description | Default |
| --- | --- | --- |
| `--model-id` | Hugging Face model ID (e.g., `google/gemma-4-31B-it` or `zai-org/GLM-5.1`) | **Required** |
| `--hardware` | Target hardware profile (`ghostfish_p3_derated` or `gb200_derated`) | `ghostfish_p3_derated` |
| `--batch-size` | Batch size | `896` |
| `--seq-len-kv` | Context length / sequence length | `1024` |
| `--regime` | Working regime (`prefill` or `decode`) | `prefill` |
| `--dtype` | Precision and data format (`bf16` or `fp8`) | `bf16` |
| `--tp` | Tensor Parallelism degree | `2` |
| `--dp` | Data Parallelism degree | `4` |
| `--token-step` | Target decode token step (`FirstToken` or `LastToken`) | `FirstToken` |
| `--output-dir` | Target folder to write HTML waffle grid reports | `.` |

## 4. Example Invocation

```bash
python3 .agents/skills/roofline_analyzer/scripts/generate_waffle_roofline.py \
  --model-id zai-org/GLM-5.1 \
  --hardware ghostfish_p3_derated \
  --batch-size 896 \
  --seq-len-kv 1024 \
  --regime prefill \
  --dtype bf16 \
  --tp 2 \
  --dp 4 \
  --output-dir ./reports
```
