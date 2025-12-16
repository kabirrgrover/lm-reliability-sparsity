# LM Reliability Under Sparsity 🔬

**Investigating how temperature sampling affects reliability in sparse vs dense language models**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-cs.CL-b31b1b.svg)](https://arxiv.org/)

---

## 🎯 Key Finding

> **Instruction tuning, rather than architectural sparsity, is the primary factor determining robustness to temperature variation.**

We evaluated **9,360 generations** across three models and found that sparse MoE models **don't degrade faster** than dense models with increasing temperature—as long as they're instruction-tuned.

---

## 📊 Models Evaluated

| Model | Architecture | Total Params | Active Params | Type | Temperature Stability |
|-------|-------------|--------------|---------------|------|----------------------|
| **Mixtral-8x7B** | Sparse MoE (8 experts, top-2) | 46.7B | ~12.9B | Instruct | ✅ Stable |
| **Qwen2.5-3B** | Dense | 3B | 3B | Instruct | ✅ Stable |
| **OLMoE-7B** | Sparse MoE | 7B | ~1B | Base | ❌ Degrades |

---

## 📈 Results at a Glance

### Accuracy by Temperature
| Model | Greedy | T=0.1 | T=0.5 | T=1.0 | Trend |
|-------|--------|-------|-------|-------|-------|
| OLMoE-7B | 5.8% | 6.0% | 4.9% | 3.8% | ↓ Degrades |
| Qwen2.5-3B | 11.7% | 11.8% | 11.5% | 12.1% | → Stable |
| Mixtral-8x7B | 30.4% | 30.1% | 30.4% | 30.3% | → Stable |

### Key Metrics
- **Compliance**: Qwen 100%, Mixtral 58%, OLMoE 0%
- **Consistency**: Instruction-tuned models maintain stable accuracy across runs
- **Confidence**: Higher confidence correlates with instruction tuning quality

### Main Insight
The sparse instruct model (Mixtral) was **just as stable** as the dense instruct model (Qwen). Temperature sensitivity appears to be a property of **base models**, not sparse architectures.

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/kabirrgrover/lm-reliability-sparsity.git
cd lm-reliability-sparsity

# Install dependencies
pip install -r requirements.txt

# Run evaluation on a single model
python run_eval.py --model qwen --temperature 0.0 0.1 0.5 1.0
```

---

## 📁 Repository Structure

```
├── run_eval.py          # Main evaluation script (supports OLMoE, Mixtral, Qwen)
├── generate_data.py     # Generate arithmetic & variable binding tasks
├── analyze_failures.py  # Failure taxonomy analysis
├── plot_results.py      # Generate result visualizations
├── utils.py             # Utility functions
├── data.jsonl           # 240 evaluation examples (120 per task)
└── requirements.txt     # Python dependencies
```

---

## 📝 Tasks

We evaluate on two deterministic reasoning tasks with unambiguous answers:

### 1. Variable Binding (120 examples)
Track variable assignments through sequential operations:
```
a=3, b=2, c=a+b, d=c+1. d = ?  → 6
```

### 2. Multi-step Arithmetic (120 examples)
Evaluate expressions respecting order of operations:
```
(45-12)*3+7 = ?  → 106
```

---

## 💻 Usage

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

---

## ⚙️ Compute Requirements

| Requirement | Specification |
|-------------|---------------|
| **GPU** | NVIDIA A100 (40GB VRAM) recommended |
| **Quantization** | OLMoE and Mixtral use 4-bit (bitsandbytes NF4) |
| **Runtime** | ~3.5 hours for full evaluation (9,360 generations) |
| **Python** | 3.8+ |

---

## 📏 Metrics

| Metric | Description | How Measured |
|--------|-------------|--------------|
| **Accuracy** | Correct numerical answer | First integer extracted from output vs ground truth |
| **Compliance** | Format adherence | Output is *only* a number (no extra text) |
| **Consistency** | Run-to-run stability | Std dev of accuracy across 4 repeated runs |
| **Confidence** | Model certainty | Mean log-probability of generated tokens |

---

## 🔬 Methodology

### Prompt Template
```
Solve the following problem. Output only the final number, nothing else.

Problem: [task text]
Answer:
```

### Generation Parameters
- `max_new_tokens`: 64
- Stop condition: EOS token only
- Random seeds: Not fixed (to measure natural variance)
- Runs per temperature: 4

### Decoding Configurations
| Setting | Sampling | Temperature | Runs |
|---------|----------|-------------|------|
| Greedy | No | — | 1 |
| T=0.1 | Yes | 0.1 | 4 |
| T=0.5 | Yes | 0.5 | 4 |
| T=1.0 | Yes | 1.0 | 4 |

---

## 🔍 Failure Analysis

When models fail, they fail differently:

| Model | Primary Failure Mode | Explanation |
|-------|---------------------|-------------|
| **OLMoE-7B** | Extra text (100%) | Ignores format instructions, always explains |
| **Qwen2.5-3B** | Numeric error (100%) | Perfect compliance, but computation errors |
| **Mixtral-8x7B** | Mixed (57% numeric, 22% hallucinated steps) | Mostly compliant, occasional explanations |

---

## 📚 Citation

If you use this code, please cite:

```bibtex
@misc{grover2024lmreliability,
  title={Reliability Under Randomness: How Sparse and Dense Language Models Behave Across Decoding Temperatures},
  author={Grover, Kabir},
  year={2024},
  url={https://github.com/kabirrgrover/lm-reliability-sparsity}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Models: [OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0924), [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1), [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- Compute: Google Colab (A100 GPU)
- Libraries: Hugging Face Transformers, bitsandbytes
