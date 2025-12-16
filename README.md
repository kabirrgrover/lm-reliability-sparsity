# LM Reliability Under Sparsity 🔬

**Investigating how temperature sampling affects reliability in sparse vs dense language models**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Finding

> **Instruction tuning, rather than architectural sparsity, is the primary factor determining robustness to temperature variation.**

We evaluated 9,360 generations across three models and found that sparse MoE models **don't degrade faster** than dense models with increasing temperature—as long as they're instruction-tuned.

## Models Evaluated

| Model | Architecture | Active Params | Type | Temperature Stability |
|-------|-------------|---------------|------|----------------------|
| **Mixtral-8x7B** | Sparse MoE | ~12.9B | Instruct | ✅ Stable |
| **Qwen2.5-3B** | Dense | 3B | Instruct | ✅ Stable |
| **OLMoE-7B** | Sparse MoE | ~1B | Base | ❌ Degrades |

## Results at a Glance

- **Accuracy**: OLMoE degraded 5.8% → 3.8% with temperature; Mixtral & Qwen stayed flat
- **Compliance**: Qwen 100%, Mixtral 58%, OLMoE 0%
- **Key insight**: The sparse instruct model (Mixtral) was just as stable as the dense instruct model (Qwen)

## Quick Start

```bash
# Clone the repo
git clone https://github.com/kabirrgrover/lm-reliability-sparsity.git
cd lm-reliability-sparsity

# Install dependencies
pip install -r requirements.txt

# Run evaluation
python run_eval.py --model qwen --temperature 0.0 0.1 0.5 1.0
```

## Repository Structure

```
├── run_eval.py          # Main evaluation script (supports OLMoE, Mixtral, Qwen)
├── generate_data.py     # Generate arithmetic & variable binding tasks
├── analyze_failures.py  # Failure taxonomy analysis
├── plot_results.py      # Generate result visualizations
├── utils.py             # Utility functions
├── data.jsonl           # 240 evaluation examples
└── requirements.txt     # Python dependencies
```

## Tasks

We evaluate on two deterministic reasoning tasks:

1. **Variable Binding** (120 examples)
   ```
   a=3, b=2, c=a+b, d=c+1. d = ?  → 6
   ```

2. **Multi-step Arithmetic** (120 examples)
   ```
   (45-12)*3+7 = ?  → 106
   ```

## Usage

### Generate Data
```bash
python generate_data.py --num_examples 120 --output data.jsonl
```

### Run Evaluation
```bash
# Single model, multiple temperatures
python run_eval.py --model mixtral --temperature 0.0 0.1 0.5 1.0 --runs 4

# Available models: olmoe, qwen, mixtral
```

### Analyze Results
```bash
python analyze_failures.py --input results.json
python plot_results.py --input results.json --output figures/
```

## Compute Requirements

- **GPU**: NVIDIA A100 (40GB VRAM) recommended
- **Quantization**: OLMoE and Mixtral use 4-bit (bitsandbytes NF4)
- **Runtime**: ~3.5 hours for full evaluation (9,360 generations)

## Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Correct numerical answer extracted |
| **Compliance** | Output is *only* a number (no extra text) |
| **Consistency** | Std dev of accuracy across repeated runs |
| **Confidence** | Mean log-probability of generated tokens |

## Citation

If you use this code, please cite:

```bibtex
@misc{grover2024lmreliability,
  title={Reliability Under Randomness: How Sparse and Dense Language Models Behave Across Decoding Temperatures},
  author={Grover, Kabir},
  year={2024},
  url={https://github.com/kabirrgrover/lm-reliability-sparsity}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
