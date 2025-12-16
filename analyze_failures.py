"""
Failure taxonomy analyzer for LM reliability experiment.

Categorizes model errors into types:
- wrong_operation: Computed wrong op (+ vs - vs *)
- numeric_error: Off-by-one or arithmetic mistake
- extra_text: Added explanation/words
- hallucinated_steps: Invented reasoning not requested
- incomplete: Cut off mid-generation
- refusal: Model refuses or asks for clarification
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path

from utils import setup_logging, OUTPUTS_PATH, FAILURE_ANALYSIS_PATH

# Initialize logger
logger = setup_logging("analyze_failures")


def classify_failure(record: dict) -> str | None:
    """
    Classify a failure into a category.
    
    Returns None if the prediction was correct.
    """
    if record["correct"]:
        return None  # Not a failure
    
    generated = record["generated"].strip()
    expected = record["expected"]
    extracted = record["extracted"]
    compliant = record["compliant"]
    
    # Check for empty or very short output
    if len(generated) < 1:
        return "incomplete"
    
    # Check for refusal patterns
    refusal_patterns = [
        r"\bi\b", r"\bsorry\b", r"\bcannot\b", r"\bcan't\b", 
        r"\bunable\b", r"\bplease\b", r"\bclarify\b"
    ]
    lower_gen = generated.lower()
    if any(re.search(p, lower_gen) for p in refusal_patterns):
        return "refusal"
    
    # Check for hallucinated reasoning steps
    reasoning_patterns = [
        r"\bbecause\b", r"\bso\b", r"\btherefore\b", r"\bstep\b",
        r"\bfirst\b", r"\bthen\b", r"\bnext\b", r"\blet'?s\b",
        r"\bcalculat", r"\bthe answer is\b"
    ]
    if any(re.search(p, lower_gen) for p in reasoning_patterns):
        return "hallucinated_steps"
    
    # Check for extra text (non-compliant but has a number)
    if not compliant and extracted is not None:
        # Has extra text beyond just the number
        if len(generated) > len(extracted) + 5:  # More than just whitespace
            return "extra_text"
    
    # Check for incomplete (ends mid-number or very short)
    if len(generated) < 3 and extracted is None:
        return "incomplete"
    
    # If we extracted a number but it's wrong, it's a numeric error
    if extracted is not None:
        return "numeric_error"
    
    # Default: couldn't extract any number
    return "incomplete"


def analyze_failures():
    """Main failure analysis loop."""
    logger.info("=" * 60)
    logger.info("Starting Failure Taxonomy Analysis")
    logger.info("=" * 60)
    
    # Load outputs
    if not Path(OUTPUTS_PATH).exists():
        logger.error(f"Output file not found: {OUTPUTS_PATH}")
        logger.error("Run run_eval.py first!")
        return
    
    logger.info(f"Loading outputs from {OUTPUTS_PATH}")
    with open(OUTPUTS_PATH, "r") as f:
        records = [json.loads(line) for line in f]
    
    logger.info(f"✓ Loaded {len(records)} records")
    
    # Separate correct and incorrect
    correct_records = [r for r in records if r["correct"]]
    failure_records = [r for r in records if not r["correct"]]
    
    logger.info(f"  - Correct: {len(correct_records)}")
    logger.info(f"  - Failures: {len(failure_records)}")
    
    # Classify failures
    logger.info("\nClassifying failures...")
    
    classified_failures = []
    category_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    overall_counts = Counter()
    
    for record in failure_records:
        category = classify_failure(record)
        
        classified_record = {
            "model": record["model"],
            "setting": record["setting"],
            "task": record["task"],
            "prompt": record["prompt"][:100] + "..." if len(record["prompt"]) > 100 else record["prompt"],
            "expected": record["expected"],
            "generated": record["generated"],
            "extracted": record["extracted"],
            "failure_category": category,
        }
        classified_failures.append(classified_record)
        
        # Count by model/setting/category
        category_counts[record["model"]][record["setting"]][category] += 1
        overall_counts[category] += 1
    
    logger.info("\n--- Overall Failure Distribution ---")
    for cat, count in overall_counts.most_common():
        pct = count / len(failure_records) * 100 if failure_records else 0
        logger.info(f"  {cat}: {count} ({pct:.1f}%)")
    
    # Detailed breakdown by model and setting
    logger.info("\n--- Breakdown by Model/Setting ---")
    for model in sorted(category_counts.keys()):
        for setting in sorted(category_counts[model].keys()):
            cats = category_counts[model][setting]
            total = sum(cats.values())
            logger.info(f"\n{model}/{setting} ({total} failures):")
            for cat in ["numeric_error", "extra_text", "hallucinated_steps", "incomplete", "refusal"]:
                count = cats.get(cat, 0)
                if count > 0:
                    pct = count / total * 100 if total else 0
                    logger.info(f"    {cat}: {count} ({pct:.1f}%)")
    
    # Prepare summary for JSON output
    summary = {
        "total_records": len(records),
        "correct": len(correct_records),
        "failures": len(failure_records),
        "overall_distribution": dict(overall_counts),
        "by_model_setting": {
            model: {
                setting: dict(cats)
                for setting, cats in settings.items()
            }
            for model, settings in category_counts.items()
        }
    }
    
    # Save results
    output = {
        "summary": summary,
        "failure_examples": classified_failures[:100],  # Save first 100 for inspection
    }
    
    logger.info(f"\nSaving analysis to {FAILURE_ANALYSIS_PATH}")
    with open(FAILURE_ANALYSIS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("✓ Failure analysis complete!")
    logger.info("=" * 60)
    
    return summary


def auto_download_analysis():
    """Auto-download analysis if running in Colab."""
    try:
        from google.colab import files
        print("\n" + "=" * 60)
        print("AUTO-DOWNLOADING ANALYSIS...")
        print("=" * 60)
        files.download('failure_analysis.json')
        print("✓ Analysis downloaded!")
    except ImportError:
        pass
    except Exception as e:
        print(f"Auto-download failed: {e}")


if __name__ == "__main__":
    summary = analyze_failures()

    if summary:
        print("\n" + "=" * 60)
        print("FAILURE TAXONOMY SUMMARY")
        print("=" * 60)
        print(f"Total failures: {summary['failures']} / {summary['total_records']}")
        print("\nDistribution:")
        for cat, count in sorted(summary["overall_distribution"].items(), key=lambda x: -x[1]):
            pct = count / summary["failures"] * 100 if summary["failures"] else 0
            print(f"  {cat:20s}: {count:4d} ({pct:5.1f}%)")

    auto_download_analysis()
