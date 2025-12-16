"""
Shared utilities and logging configuration for the LM reliability experiment.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(name: str = "lm_reliability", log_dir: str = "logs") -> logging.Logger:
    """
    Set up logging to both console and file.
    
    Args:
        name: Logger name
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger
    
    # File handler - detailed logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        log_path / f"{name}_{timestamp}.log",
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    
    # Console handler - info and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(levelname)-8s | %(message)s"
    )
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Model configurations
# 4-way comparison: Sparse vs Dense × Aligned vs Unaligned
#
# Mixtral 8x7B Instruct: Sparse MoE + Instruction-tuned (46.7B total, ~12B active)
# OLMoE-1B-7B: Sparse MoE + Research model (7B total, 1B active)
# Qwen2.5-3B Instruct: Dense + Instruction-tuned (3B params)
# GPT-2: Dense + Base model (124M params)
MODELS = {
    "OLMoE-7B": "allenai/OLMoE-1B-7B-0924",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
    "Mixtral-8x7B": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

# Models requiring quantization (too large for full precision)
QUANTIZED_MODELS = {
    "allenai/OLMoE-1B-7B-0924",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

# Decoding settings
SETTINGS = [
    {"name": "greedy", "do_sample": False, "temperature": None},
    {"name": "temp_0.1", "do_sample": True, "temperature": 0.1},
    {"name": "temp_0.5", "do_sample": True, "temperature": 0.5},
    {"name": "temp_1.0", "do_sample": True, "temperature": 1.0},
]

# Number of runs for consistency measurement
NUM_RUNS = 4

# Paths
DATA_PATH = "data.jsonl"
OUTPUTS_PATH = "outputs.jsonl"
RESULTS_PATH = "results.json"
FAILURE_ANALYSIS_PATH = "failure_analysis.json"
FIGURES_DIR = "figures"
