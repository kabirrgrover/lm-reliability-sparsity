"""
Main evaluation script for LM reliability experiment.

Runs both models across all decoding settings, collecting:
- Accuracy and compliance metrics
- Token-level confidence (log probabilities)
- Multiple runs for consistency measurement
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import (
    setup_logging, 
    MODELS, SETTINGS, NUM_RUNS,
    DATA_PATH, OUTPUTS_PATH, RESULTS_PATH,
    QUANTIZED_MODELS
)

# Initialize logger
logger = setup_logging("run_eval")

# Regex for compliance checking
NUMBER_ONLY_RE = re.compile(r"^\s*-?\d+\s*$")


def load_data(path: str = DATA_PATH) -> list[dict]:
    """Load dataset from JSONL file."""
    logger.info(f"Loading data from {path}")
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    logger.info(f"✓ Loaded {len(data)} examples")
    return data


def extract_answer(text: str) -> tuple[str | None, bool]:
    """
    Extract answer from model output.
    
    Returns:
        (extracted_number, is_compliant)
    """
    stripped = text.strip()
    compliant = bool(NUMBER_ONLY_RE.match(stripped))
    
    # Extract first integer for accuracy check
    m = re.search(r"-?\d+", stripped)
    extracted = m.group(0) if m else None
    
    return extracted, compliant


def format_prompt_for_model(prompt: str, tokenizer, model_id: str) -> str:
    """
    Format prompt using chat template for instruct models.
    """
    # Check if tokenizer has chat template
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return formatted
        except Exception:
            pass  # Fall back to raw prompt
    return prompt


def generate_with_confidence(
    model,
    tokenizer,
    prompt: str,
    model_id: str,
    do_sample: bool,
    temperature: float | None,
    max_new_tokens: int = 16
) -> dict:
    """
    Generate text and capture token-level confidence.
    
    Returns:
        Dict with generated text and confidence metrics
    """
    # Format prompt with chat template if available
    formatted_prompt = format_prompt_for_model(prompt, tokenizer, model_id)
    
    # Use add_special_tokens=True for proper model behavior
    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_scores": True,  # Get logits for confidence
    }

    if do_sample and temperature is not None:
        gen_kwargs["temperature"] = temperature
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    # Extract generated tokens (excluding prompt)
    prompt_length = inputs["input_ids"].shape[1]
    gen_ids = outputs.sequences[0][prompt_length:]
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    # Compute token-level log probabilities
    token_logprobs = []
    if outputs.scores:
        for i, score in enumerate(outputs.scores):
            if i >= len(gen_ids):
                break
            # Get log softmax of logits
            log_probs = torch.log_softmax(score[0], dim=-1)
            # Get log prob of the chosen token
            token_id = gen_ids[i].item()
            token_logprob = log_probs[token_id].item()
            token_logprobs.append(token_logprob)
    
    # Compute confidence metrics
    if token_logprobs:
        mean_confidence = sum(token_logprobs) / len(token_logprobs)
        min_confidence = min(token_logprobs)
        max_confidence = max(token_logprobs)
    else:
        mean_confidence = min_confidence = max_confidence = 0.0
    
    return {
        "generated": generated_text,
        "token_logprobs": token_logprobs,
        "mean_confidence": mean_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,
        "num_tokens": len(token_logprobs),
    }


def run_evaluation():
    """Main evaluation loop."""
    logger.info("=" * 60)
    logger.info("Starting LM Reliability Evaluation")
    logger.info("=" * 60)
    
    # Load data
    data = load_data()
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Storage for all outputs and aggregated results
    all_outputs = []
    aggregated_results = []
    
    for model_key, model_id in MODELS.items():
        logger.info("=" * 60)
        logger.info(f"Loading model: {model_key} ({model_id})")
        logger.info("=" * 60)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            # Check if model needs quantization (too large for full precision)
            if model_id in QUANTIZED_MODELS:
                logger.info("  Using 4-bit quantization for large model...")
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    device_map="auto",  # Automatic device placement for quantized
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype="auto",
                )
                model.to(device)
            
            model.eval()
            logger.info(f"✓ Model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            continue
        
        for setting in SETTINGS:
            setting_name = setting["name"]
            do_sample = setting["do_sample"]
            temperature = setting["temperature"]
            
            # Greedy only needs 1 run, sampling needs NUM_RUNS
            num_runs = 1 if not do_sample else NUM_RUNS
            
            logger.info(f"\n--- Setting: {setting_name} ({num_runs} run(s)) ---")
            
            # Track metrics across runs
            run_metrics = []
            
            for run_idx in range(num_runs):
                # Set seed for reproducibility
                seed = 42 + run_idx
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                
                logger.info(f"  Run {run_idx + 1}/{num_runs}")
                
                counts = defaultdict(int)
                confidence_values = []
                
                for ex_idx, ex in enumerate(data):
                    # Generate with confidence tracking
                    result = generate_with_confidence(
                        model, tokenizer, ex["prompt"],
                        model_id=model_id,
                        do_sample=do_sample,
                        temperature=temperature
                    )

                    # Extract answer and check compliance
                    extracted, compliant = extract_answer(result["generated"])

                    # DEBUG: Print first example to verify output format
                    if ex_idx == 0 and run_idx == 0:
                        print(f"\n=== DEBUG Example {ex_idx + 1} ===")
                        print(f"Expected: {ex['answer']}")
                        print(f"Generated (repr): {repr(result['generated'])}")
                        print(f"Num tokens: {result['num_tokens']}")
                        print(f"Extracted: {extracted}, Compliant: {compliant}")
                        print("=" * 40)
                    
                    # Compare as integers for robustness
                    try:
                        correct = (int(extracted) == int(ex["answer"])) if extracted else False
                    except (ValueError, TypeError):
                        correct = (str(extracted) == str(ex["answer"]))
                    
                    # Store output
                    output_record = {
                        "model": model_key,
                        "setting": setting_name,
                        "run": run_idx,
                        "task": ex["task"],
                        "prompt": ex["prompt"],
                        "expected": ex["answer"],
                        "generated": result["generated"],
                        "extracted": extracted,
                        "correct": correct,
                        "compliant": compliant,
                        "token_logprobs": result["token_logprobs"],
                        "mean_confidence": result["mean_confidence"],
                        "min_confidence": result["min_confidence"],
                        "num_tokens": result["num_tokens"],
                    }
                    all_outputs.append(output_record)
                    
                    # Update counts
                    counts["total"] += 1
                    counts[f"task_{ex['task']}_total"] += 1
                    counts["correct"] += int(correct)
                    counts["compliant"] += int(compliant)
                    counts[f"task_{ex['task']}_correct"] += int(correct)
                    counts[f"task_{ex['task']}_compliant"] += int(compliant)
                    
                    # Track confidence
                    confidence_values.append(result["mean_confidence"])
                    
                    if (ex_idx + 1) % 50 == 0:
                        logger.debug(f"    Processed {ex_idx + 1}/{len(data)} examples")
                
                # Compute run metrics
                def rate(num, den): 
                    return (num / den) if den else 0.0
                
                run_result = {
                    "accuracy": rate(counts["correct"], counts["total"]),
                    "compliance": rate(counts["compliant"], counts["total"]),
                    "accuracy_var_bind": rate(counts["task_var_bind_correct"], counts["task_var_bind_total"]),
                    "compliance_var_bind": rate(counts["task_var_bind_compliant"], counts["task_var_bind_total"]),
                    "accuracy_arithmetic": rate(counts["task_arithmetic_correct"], counts["task_arithmetic_total"]),
                    "compliance_arithmetic": rate(counts["task_arithmetic_compliant"], counts["task_arithmetic_total"]),
                    "mean_confidence": sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
                }
                run_metrics.append(run_result)
                
                logger.info(f"    Accuracy: {run_result['accuracy']:.3f}, Compliance: {run_result['compliance']:.3f}")
            
            # Aggregate across runs
            def avg(key):
                return sum(r[key] for r in run_metrics) / len(run_metrics)
            
            def std(key):
                if len(run_metrics) < 2:
                    return 0.0
                mean = avg(key)
                variance = sum((r[key] - mean) ** 2 for r in run_metrics) / len(run_metrics)
                return variance ** 0.5
            
            aggregated = {
                "model": model_key,
                "setting": setting_name,
                "num_runs": num_runs,
                "accuracy": avg("accuracy"),
                "accuracy_std": std("accuracy"),
                "compliance": avg("compliance"),
                "compliance_std": std("compliance"),
                "accuracy_var_bind": avg("accuracy_var_bind"),
                "accuracy_var_bind_std": std("accuracy_var_bind"),
                "compliance_var_bind": avg("compliance_var_bind"),
                "accuracy_arithmetic": avg("accuracy_arithmetic"),
                "accuracy_arithmetic_std": std("accuracy_arithmetic"),
                "compliance_arithmetic": avg("compliance_arithmetic"),
                "mean_confidence": avg("mean_confidence"),
            }
            aggregated_results.append(aggregated)
            
            logger.info(f"  → Aggregated: Acc={aggregated['accuracy']:.3f}±{aggregated['accuracy_std']:.3f}")
        
        # Clear model from memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleared model from memory")
    
    # Save all outputs
    logger.info(f"\nSaving {len(all_outputs)} output records to {OUTPUTS_PATH}")
    with open(OUTPUTS_PATH, "w") as f:
        for record in all_outputs:
            f.write(json.dumps(record) + "\n")
    
    # Save aggregated results
    logger.info(f"Saving aggregated results to {RESULTS_PATH}")
    with open(RESULTS_PATH, "w") as f:
        json.dump(aggregated_results, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("✓ Evaluation complete!")
    logger.info("=" * 60)
    
    return aggregated_results


def auto_download_results():
    """Auto-download results if running in Colab."""
    try:
        from google.colab import files
        print("\n" + "=" * 60)
        print("AUTO-DOWNLOADING RESULTS...")
        print("=" * 60)
        files.download('outputs.jsonl')
        files.download('results.json')
        print("✓ Results downloaded!")
    except ImportError:
        print("\nNot running in Colab - skipping auto-download")
    except Exception as e:
        print(f"\nAuto-download failed: {e}")
        print("Please manually download outputs.jsonl and results.json")


if __name__ == "__main__":
    results = run_evaluation()

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Setting':<12} {'Accuracy':>10} {'Compliance':>12} {'Confidence':>12}")
    print("-" * 80)
    for r in results:
        acc_str = f"{r['accuracy']:.3f}±{r['accuracy_std']:.3f}"
        print(f"{r['model']:<20} {r['setting']:<12} {acc_str:>10} {r['compliance']:>12.3f} {r['mean_confidence']:>12.3f}")

    # Auto-download in Colab
    auto_download_results()
