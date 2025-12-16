"""
Dataset generator for LM reliability experiment.
Creates deterministic reasoning tasks:
1. Variable binding (program-like assignments)
2. Multi-step arithmetic (2-4 operations)
"""

import json
import random
from utils import setup_logging, DATA_PATH

# Initialize logger
logger = setup_logging("generate_data")


def gen_variable_binding(n: int = 120) -> list[dict]:
    """
    Generate variable binding tasks.

    Example:
        a = 3, b = 7, c = a + b, d = c - 2
        Answer: 8
    """
    logger.info(f"Generating {n} variable binding examples...")
    data = []

    # Few-shot examples with strict format instruction
    few_shot_prefix = """Respond with ONLY the final number. No explanation.

a=3, b=2, c=a+b, d=c+1. d = 6
a=5, b=4, c=a*b, d=c-3. d = 17
"""

    for i in range(n):
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        op1 = random.choice(["+", "-", "*"])

        if op1 == "+":
            c = a + b
            expr1 = "a + b"
        elif op1 == "-":
            c = a - b
            expr1 = "a - b"
        else:
            c = a * b
            expr1 = "a * b"

        k = random.randint(1, 9)
        op2 = random.choice(["+", "-"])

        if op2 == "+":
            d = c + k
            expr2 = f"c + {k}"
        else:
            d = c - k
            expr2 = f"c - {k}"

        prompt = (
            few_shot_prefix +
            f"a={a}, b={b}, c={expr1}, d={expr2}. d ="
        )
        
        data.append({
            "task": "var_bind",
            "prompt": prompt,
            "answer": str(d),
            "metadata": {"a": a, "b": b, "c": c, "d": d, "op1": op1, "op2": op2}
        })
        
        if (i + 1) % 50 == 0:
            logger.debug(f"  Generated {i + 1}/{n} var_bind examples")
    
    logger.info(f"✓ Generated {len(data)} variable binding examples")
    return data


def gen_arithmetic(n: int = 120) -> list[dict]:
    """
    Generate multi-step arithmetic tasks.

    Example:
        (45 - 12) * 3 + 7 = 106
    """
    logger.info(f"Generating {n} arithmetic examples...")
    data = []

    # Few-shot examples with strict format instruction
    few_shot_prefix = """Respond with ONLY the final number. No explanation.

(10-5)*2+3 = 13
(20-8)*4+5 = 53
"""

    for i in range(n):
        x = random.randint(10, 99)
        y = random.randint(1, 20)
        z = random.randint(1, 10)
        w = random.randint(1, 10)

        # (x - y) * z + w
        ans = (x - y) * z + w

        prompt = (
            few_shot_prefix +
            f"({x}-{y})*{z}+{w} ="
        )
        
        data.append({
            "task": "arithmetic",
            "prompt": prompt,
            "answer": str(ans),
            "metadata": {"x": x, "y": y, "z": z, "w": w, "expression": f"({x} - {y}) * {z} + {w}"}
        })
        
        if (i + 1) % 50 == 0:
            logger.debug(f"  Generated {i + 1}/{n} arithmetic examples")
    
    logger.info(f"✓ Generated {len(data)} arithmetic examples")
    return data


def main():
    """Generate and save the complete dataset."""
    logger.info("=" * 50)
    logger.info("Starting dataset generation")
    logger.info("=" * 50)
    
    # Set seed for reproducibility
    random.seed(42)
    logger.info("Random seed set to 42")
    
    # Generate both task types
    var_bind_data = gen_variable_binding(120)
    arithmetic_data = gen_arithmetic(120)
    
    # Combine
    data = var_bind_data + arithmetic_data
    
    # Save to JSONL
    with open(DATA_PATH, "w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
    
    logger.info("=" * 50)
    logger.info(f"✓ Saved {len(data)} examples to {DATA_PATH}")
    logger.info(f"  - var_bind: {len(var_bind_data)}")
    logger.info(f"  - arithmetic: {len(arithmetic_data)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
