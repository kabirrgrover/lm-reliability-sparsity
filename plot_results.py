"""
Visualization script for LM reliability experiment.

Generates plots:
1. Accuracy vs Temperature (overall and by task)
2. Compliance vs Temperature
3. Consistency (variance across runs)
4. Confidence profiles
5. Failure taxonomy distribution
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import setup_logging, RESULTS_PATH, FAILURE_ANALYSIS_PATH, FIGURES_DIR

# Initialize logger
logger = setup_logging("plot_results")

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')

# Colors for models
COLORS = {
    "Mixtral-8x7B": "#E63946",        # Red
    "OLMoE-7B": "#F4A261",             # Orange
    "Qwen2.5-3B": "#457B9D",           # Blue
}

MARKERS = {
    "Mixtral-8x7B": "o",
    "OLMoE-7B": "s",
    "Qwen2.5-3B": "^",
}

# Sparse MoE models (for sparse vs dense comparison)
SPARSE_MODELS = {"OLMoE-7B", "Mixtral-8x7B"}

SETTING_ORDER = ["greedy", "temp_0.1", "temp_0.5", "temp_1.0"]
SETTING_LABELS = {
    "greedy": "Greedy",
    "temp_0.1": "T=0.1",
    "temp_0.5": "T=0.5",
    "temp_1.0": "T=1.0",
}


def load_results():
    """Load aggregated results."""
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


def load_failure_analysis():
    """Load failure analysis results."""
    if not Path(FAILURE_ANALYSIS_PATH).exists():
        return None
    with open(FAILURE_ANALYSIS_PATH, "r") as f:
        return json.load(f)


def ensure_figures_dir():
    """Create figures directory if needed."""
    Path(FIGURES_DIR).mkdir(exist_ok=True)


def get_models_from_results(results):
    """Extract unique model names from results."""
    return list(dict.fromkeys(r["model"] for r in results))


def plot_metric_vs_setting(
    results: list,
    metric_key: str,
    title: str,
    ylabel: str,
    filename: str,
    show_std: bool = True,
    ylim: tuple = None
):
    """Generic plotting function for metrics vs decoding settings."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(SETTING_ORDER))
    models = get_models_from_results(results)

    for model in models:
        ys = []
        yerrs = []

        for setting in SETTING_ORDER:
            row = next((r for r in results if r["model"] == model and r["setting"] == setting), None)
            if row:
                ys.append(row.get(metric_key, 0))
                std_key = f"{metric_key}_std"
                yerrs.append(row.get(std_key, 0))
            else:
                ys.append(0)
                yerrs.append(0)

        color = COLORS.get(model, "#333333")
        marker = MARKERS.get(model, "o")

        if show_std and any(yerrs):
            ax.errorbar(
                x, ys, yerr=yerrs,
                marker=marker, markersize=8, linewidth=2,
                label=model, color=color,
                capsize=4, capthick=1.5
            )
        else:
            ax.plot(
                x, ys,
                marker=marker, markersize=8, linewidth=2,
                label=model, color=color
            )

    ax.set_xticks(x)
    ax.set_xticklabels([SETTING_LABELS[s] for s in SETTING_ORDER])
    ax.set_xlabel("Decoding Setting", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, 1.05)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(FIGURES_DIR) / filename, dpi=150)
    plt.close()
    logger.info(f"✓ Saved {filename}")


def plot_confidence_profiles(results: list):
    """Plot mean token confidence by model and setting."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = get_models_from_results(results)
    x = np.arange(len(SETTING_ORDER))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        ys = []
        for setting in SETTING_ORDER:
            row = next((r for r in results if r["model"] == model and r["setting"] == setting), None)
            if row:
                ys.append(row.get("mean_confidence", 0))
            else:
                ys.append(0)

        offset = width * (i - len(models)/2 + 0.5)
        color = COLORS.get(model, "#333333")
        ax.bar(x + offset, ys, width, label=model, color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([SETTING_LABELS[s] for s in SETTING_ORDER])
    ax.set_xlabel("Decoding Setting", fontsize=12)
    ax.set_ylabel("Mean Token Log-Probability", fontsize=12)
    ax.set_title("Token Confidence by Model and Setting", fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(Path(FIGURES_DIR) / "confidence_profiles.png", dpi=150)
    plt.close()
    logger.info("✓ Saved confidence_profiles.png")


def plot_failure_taxonomy(failure_data: dict, results: list):
    """Plot failure taxonomy as stacked bar chart."""
    if not failure_data:
        logger.warning("No failure data available, skipping taxonomy plot")
        return

    by_model_setting = failure_data.get("summary", {}).get("by_model_setting", {})
    if not by_model_setting:
        logger.warning("No model/setting breakdown in failure data")
        return

    categories = ["numeric_error", "extra_text", "hallucinated_steps", "incomplete", "refusal"]
    category_colors = {
        "numeric_error": "#FF6B6B",
        "extra_text": "#4ECDC4",
        "hallucinated_steps": "#FFE66D",
        "incomplete": "#95E1D3",
        "refusal": "#A8A8A8",
    }

    models = get_models_from_results(results)

    # Build data structure
    groups = []
    for model in models:
        if model not in by_model_setting:
            continue
        for setting in SETTING_ORDER:
            if setting not in by_model_setting[model]:
                continue
            groups.append((model, setting))

    if not groups:
        logger.warning("No failure groups found, skipping taxonomy plot")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(groups))
    width = 0.6

    bottom = np.zeros(len(groups))
    for cat in categories:
        values = []
        for model, setting in groups:
            counts = by_model_setting.get(model, {}).get(setting, {})
            values.append(counts.get(cat, 0))

        if sum(values) > 0:
            ax.bar(x, values, width, label=cat.replace("_", " ").title(), bottom=bottom, color=category_colors[cat])
            bottom += np.array(values)

    ax.set_xticks(x)
    labels = [f"{m.split('-')[0]}\n{SETTING_LABELS[s]}" for m, s in groups]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Model / Setting", fontsize=12)
    ax.set_ylabel("Failure Count", fontsize=12)
    ax.set_title("Failure Taxonomy by Model and Setting", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', title="Failure Type", fontsize=9)

    plt.tight_layout()
    plt.savefig(Path(FIGURES_DIR) / "failure_taxonomy.png", dpi=150)
    plt.close()
    logger.info("✓ Saved failure_taxonomy.png")


def plot_consistency(results: list):
    """Plot consistency (std across runs) by model and setting."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(SETTING_ORDER))
    models = get_models_from_results(results)

    for model in models:
        stds = []
        for setting in SETTING_ORDER:
            row = next((r for r in results if r["model"] == model and r["setting"] == setting), None)
            if row:
                stds.append(row.get("accuracy_std", 0))
            else:
                stds.append(0)

        color = COLORS.get(model, "#333333")
        marker = MARKERS.get(model, "o")
        ax.plot(
            x, stds,
            marker=marker, markersize=8, linewidth=2,
            label=model, color=color
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SETTING_LABELS[s] for s in SETTING_ORDER])
    ax.set_xlabel("Decoding Setting", fontsize=12)
    ax.set_ylabel("Accuracy Std Dev (across runs)", fontsize=12)
    ax.set_title("Consistency: Variance in Accuracy Across Runs", fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(FIGURES_DIR) / "consistency.png", dpi=150)
    plt.close()
    logger.info("✓ Saved consistency.png")


def plot_sparse_vs_dense_comparison(results: list):
    """Plot sparse vs dense comparison grouped."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Identify sparse and dense models using the explicit SPARSE_MODELS set
    sparse_models = [m for m in get_models_from_results(results) if m in SPARSE_MODELS]
    dense_models = [m for m in get_models_from_results(results) if m not in SPARSE_MODELS]

    x = np.arange(len(SETTING_ORDER))

    # Plot 1: Sparse models
    ax = axes[0]
    for model in sparse_models:
        ys = []
        yerrs = []
        for setting in SETTING_ORDER:
            row = next((r for r in results if r["model"] == model and r["setting"] == setting), None)
            if row:
                ys.append(row.get("accuracy", 0))
                yerrs.append(row.get("accuracy_std", 0))
            else:
                ys.append(0)
                yerrs.append(0)

        color = COLORS.get(model, "#333333")
        marker = MARKERS.get(model, "o")
        ax.errorbar(x, ys, yerr=yerrs, marker=marker, markersize=8, linewidth=2,
                   label=model, color=color, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([SETTING_LABELS[s] for s in SETTING_ORDER])
    ax.set_xlabel("Decoding Setting", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Sparse (MoE) Models", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.5)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 2: Dense models
    ax = axes[1]
    for model in dense_models:
        ys = []
        yerrs = []
        for setting in SETTING_ORDER:
            row = next((r for r in results if r["model"] == model and r["setting"] == setting), None)
            if row:
                ys.append(row.get("accuracy", 0))
                yerrs.append(row.get("accuracy_std", 0))
            else:
                ys.append(0)
                yerrs.append(0)

        color = COLORS.get(model, "#333333")
        marker = MARKERS.get(model, "o")
        ax.errorbar(x, ys, yerr=yerrs, marker=marker, markersize=8, linewidth=2,
                   label=model, color=color, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([SETTING_LABELS[s] for s in SETTING_ORDER])
    ax.set_xlabel("Decoding Setting", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Dense Models", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.5)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(FIGURES_DIR) / "sparse_vs_dense.png", dpi=150)
    plt.close()
    logger.info("✓ Saved sparse_vs_dense.png")


def auto_download_figures():
    """Auto-download figures if running in Colab."""
    try:
        from google.colab import files
        import shutil
        print("\n" + "=" * 60)
        print("AUTO-DOWNLOADING FIGURES...")
        print("=" * 60)
        shutil.make_archive('figures', 'zip', '.', FIGURES_DIR)
        files.download('figures.zip')
        print("✓ Figures downloaded!")
    except ImportError:
        print("\nNot running in Colab - skipping auto-download")
    except Exception as e:
        print(f"\nAuto-download failed: {e}")
        print(f"Please manually download the {FIGURES_DIR}/ folder")


def main():
    """Generate all visualizations."""
    logger.info("=" * 60)
    logger.info("Generating Visualizations")
    logger.info("=" * 60)

    ensure_figures_dir()

    # Load data
    if not Path(RESULTS_PATH).exists():
        logger.error(f"Results file not found: {RESULTS_PATH}")
        logger.error("Run run_eval.py first!")
        return

    results = load_results()
    logger.info(f"✓ Loaded {len(results)} result rows")

    models = get_models_from_results(results)
    logger.info(f"  Models: {models}")

    failure_data = load_failure_analysis()
    if failure_data:
        logger.info("✓ Loaded failure analysis data")
    else:
        logger.warning("No failure analysis data found")

    # Generate plots
    logger.info("\nGenerating plots...")

    # 1. Overall accuracy
    plot_metric_vs_setting(
        results,
        metric_key="accuracy",
        title="Overall Accuracy vs Decoding Setting",
        ylabel="Accuracy",
        filename="accuracy_overall.png",
        ylim=(0, 0.5)
    )

    # 2. Accuracy by task
    plot_metric_vs_setting(
        results,
        metric_key="accuracy_var_bind",
        title="Accuracy: Variable Binding Task",
        ylabel="Accuracy",
        filename="accuracy_var_bind.png",
        ylim=(0, 0.5)
    )

    plot_metric_vs_setting(
        results,
        metric_key="accuracy_arithmetic",
        title="Accuracy: Arithmetic Task",
        ylabel="Accuracy",
        filename="accuracy_arithmetic.png",
        ylim=(0, 0.5)
    )

    # 3. Compliance
    plot_metric_vs_setting(
        results,
        metric_key="compliance",
        title="Output Compliance vs Decoding Setting",
        ylabel="Compliance Rate",
        filename="compliance.png",
        show_std=False,
        ylim=(0, 1.1)
    )

    # 4. Consistency (variance)
    plot_consistency(results)

    # 5. Confidence profiles
    plot_confidence_profiles(results)

    # 6. Sparse vs Dense comparison
    plot_sparse_vs_dense_comparison(results)

    # 7. Failure taxonomy
    if failure_data:
        plot_failure_taxonomy(failure_data, results)

    logger.info("=" * 60)
    logger.info("✓ All visualizations complete!")
    logger.info(f"  Saved to: {FIGURES_DIR}/")
    logger.info("=" * 60)

    # Auto-download in Colab
    auto_download_figures()


if __name__ == "__main__":
    main()
