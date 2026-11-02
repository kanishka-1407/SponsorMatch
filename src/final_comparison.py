from pathlib import Path
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

BASELINE_FILE = REPORTS / "baseline_results.csv"
MODEL_FILE = REPORTS / "model_results.csv"

OUTPUT_CSV = REPORTS / "final_model_comparison.csv"
OUTPUT_MD = REPORTS / "final_model_comparison.md"


# ============================================================
# Main
# ============================================================

def main():

    print("Loading baseline results...")

    baseline_df = pd.read_csv(BASELINE_FILE)

    print("Loading neural network results...")

    model_df = pd.read_csv(MODEL_FILE)


    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    final_df = pd.concat(
        [
            baseline_df,
            model_df
        ],
        ignore_index=True
    )


    # --------------------------------------------------------
    # Select important columns
    # --------------------------------------------------------

    columns = [
        "split",
        "model",
        "Hit@1",
        "Hit@5",
        "Hit@10",
        "Hit@20",
        "MRR",
        "nDCG@10",
        "Coverage@10",
        "n_test",
    ]

    # Keep only columns that actually exist
    columns = [
        col for col in columns
        if col in final_df.columns
    ]

    final_df = final_df[columns]


    # --------------------------------------------------------
    # Sort results
    # --------------------------------------------------------

    final_df = final_df.sort_values(
        by=["split", "nDCG@10"],
        ascending=[True, False]
    )


    # --------------------------------------------------------
    # Round values
    # --------------------------------------------------------

    metric_columns = [
        "Hit@1",
        "Hit@5",
        "Hit@10",
        "Hit@20",
        "MRR",
        "nDCG@10",
    ]

    for col in metric_columns:

        if col in final_df.columns:

            final_df[col] = final_df[col].round(4)


    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    final_df.to_csv(
        OUTPUT_CSV,
        index=False
    )


    # --------------------------------------------------------
    # Create Markdown report
    # --------------------------------------------------------

    md = [

        "# Final Model Comparison",

        "",

        "## Startup-to-Investor Ranking",

        "",

        "This report compares the baseline recommendation "
        "models with the neural network model.",

        "",

        "## Models Compared",

        "",

        "- **Popularity**: Recommends investors based on their "
        "frequency in the training data.",

        "",

        "- **Affinity (equal weights)**: Ranks investors based "
        "on similarity between startup and investor-related "
        "features using equal feature weights.",

        "",

        "- **Affinity (tuned on validation)**: Uses feature "
        "weights tuned using the validation set.",

        "",

        "- **Neural Network v3**: Uses categorical feature "
        "embeddings and normalized numeric features followed "
        "by a feed-forward neural network.",

        "",

        "## Evaluation Metrics",

        "",

        "- **Hit@1**: Correct investor appears at rank 1.",

        "- **Hit@5**: Correct investor appears in the top 5.",

        "- **Hit@10**: Correct investor appears in the top 10.",

        "- **Hit@20**: Correct investor appears in the top 20.",

        "- **MRR**: Mean Reciprocal Rank; rewards higher-ranked "
        "correct recommendations.",

        "- **nDCG@10**: Measures ranking quality within the top 10.",

        "- **Coverage@10**: Number of unique investors appearing "
        "in the top-10 recommendations.",

        "",

        "## Final Results",

        "",

        final_df.to_markdown(index=False),

        "",

        "## Conclusion",

        "",

        "The models should be interpreted separately for the "
        "random split and the time-based split.",

        "",

        "The random split measures performance when companies are "
        "held out randomly, while the time split provides a more "
        "realistic evaluation of recommending investors for newer "
        "startup funding records.",

    ]


    OUTPUT_MD.write_text(
        "\n".join(md),
        encoding="utf-8"
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("FINAL MODEL COMPARISON")

    print("=" * 70 + "\n")

    print(
        final_df.to_string(
            index=False
        )
    )

    print("\nFiles created successfully:")

    print(OUTPUT_CSV)

    print(OUTPUT_MD)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()