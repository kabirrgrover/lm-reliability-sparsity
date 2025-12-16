"""
Debug script to inspect model outputs.
Run after run_eval.py to see sample outputs per model.
"""

import json
from collections import defaultdict

def main():
    print("=" * 70)
    print("SAMPLE OUTPUTS BY MODEL")
    print("=" * 70)

    # Load outputs
    outputs_by_model = defaultdict(list)
    with open("outputs.jsonl", "r") as f:
        for line in f:
            record = json.loads(line)
            outputs_by_model[record["model"]].append(record)

    for model, records in outputs_by_model.items():
        print(f"\n{'='*70}")
        print(f"MODEL: {model}")
        print(f"{'='*70}")

        # Show first 5 outputs for greedy setting
        greedy_records = [r for r in records if r["setting"] == "greedy"][:5]

        if not greedy_records:
            greedy_records = records[:5]

        for i, r in enumerate(greedy_records):
            print(f"\n--- Example {i+1} ---")
            print(f"Expected: {r['expected']}")
            print(f"Generated: '{r['generated']}'")
            print(f"Extracted: {r['extracted']}")
            print(f"Correct: {r['correct']}, Compliant: {r['compliant']}")

        # Summary stats
        total = len(records)
        correct = sum(1 for r in records if r["correct"])
        compliant = sum(1 for r in records if r["compliant"])
        print(f"\n--- Summary for {model} ---")
        print(f"Total: {total}, Correct: {correct} ({100*correct/total:.1f}%), Compliant: {compliant} ({100*compliant/total:.1f}%)")

if __name__ == "__main__":
    main()
