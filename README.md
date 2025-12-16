# LM Reliability Under Sparsity

Code for evaluating language model reliability under varying temperature settings, comparing sparse Mixture-of-Experts (MoE) and dense architectures.

## Overview

This repository contains the evaluation framework used in our study comparing:
- **OLMoE-7B** (sparse, base)
- **Mixtral-8x7B** (sparse, instruct)  
- **Qwen2.5-3B** (dense, instruct)

## Files

- `run_eval.py` - Main evaluation script
- `generate_data.py` - Generate arithmetic and variable binding test data
- `analyze_failures.py` - Failure analysis and taxonomy
- `plot_results.py` - Generate result visualizations
- `utils.py` - Utility functions
- `data.jsonl` - Evaluation dataset (240 examples)
- `requirements.txt` - Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Generate evaluation data:
```bash
python generate_data.py
```

2. Run evaluation:
```bash
python run_eval.py --model olmoe --temperature 0.0 0.1 0.5 1.0
```

3. Analyze results:
```bash
python analyze_failures.py --input results.json
python plot_results.py --input results.json
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- bitsandbytes (for quantization)

## License

MIT
