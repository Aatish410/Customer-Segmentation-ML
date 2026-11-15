"""Run the complete Customer Segmentation training and evaluation pipeline."""
from pathlib import Path
import sys
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ml_pipeline import load_transactions, clean_transactions, create_rfm, fit_and_compare, FEATURES
from report_generator import write_project_report

DATA_DIR = ROOT / "Dataset"
MODEL_DIR = ROOT / "Trained_Model"
RESULTS_DIR = ROOT / "Results"
DOC_DIR = ROOT / "Project_Documentation"
for folder in [MODEL_DIR, RESULTS_DIR, DOC_DIR]:
    folder.mkdir(exist_ok=True)


def save_eda(cleaned, rfm):
    # 1. Top countries by number of transaction rows.
    country_counts = cleaned["Country"].value_counts().head(10).sort_values()
    plt.figure(figsize=(10, 5))
    plt.barh(country_counts.index.astype(str), country_counts.values)
    plt.title("Top 10 Countries by Transaction Count")
    plt.xlabel("Number of transaction rows")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "01_top_countries.png", dpi=160)
    plt.close()

    # 2. RFM feature distributions.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, FEATURES):
        ax.hist(rfm[col], bins=40)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Number of customers")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "02_rfm_distributions.png", dpi=160)
    plt.close()

    # 3. Frequency vs Monetary.
    plt.figure(figsize=(8, 5))
    plt.scatter(rfm["Frequency"], rfm["Monetary"], alpha=0.6)
    plt.title("Customer Frequency vs Monetary Value")
    plt.xlabel("Frequency (distinct invoices)")
    plt.ylabel("Monetary (£)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "03_frequency_vs_monetary.png", dpi=160)
    plt.close()

    # 4. Correlation heatmap.
    corr = rfm[FEATURES].corr().values
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(FEATURES)), FEATURES)
    ax.set_yticks(range(len(FEATURES)), FEATURES)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center")
    ax.set_title("Correlation Heatmap of RFM Features")
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "04_rfm_correlation.png", dpi=160)
    plt.close()


def save_cluster_visualizations(rfm_segmented):
    plt.figure(figsize=(9, 6))
    for segment, group in rfm_segmented.groupby("Segment"):
        plt.scatter(group["Frequency"], group["Monetary"], alpha=0.65, label=segment)
    plt.title("Customer Segments: Frequency vs Monetary")
    plt.xlabel("Frequency (distinct invoices)")
    plt.ylabel("Monetary (£)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "05_customer_clusters.png", dpi=160)
    plt.close()

    counts = rfm_segmented["Segment"].value_counts()
    plt.figure(figsize=(9, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Number of Customers in Each Segment")
    plt.ylabel("Number of customers")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "06_segment_counts.png", dpi=160)
    plt.close()


def main():
    print("[1/7] Loading UCI Online Retail dataset...")
    raw = load_transactions(DATA_DIR)
    print(f"Raw dataset shape: {raw.shape}")
    # Save the dataset-understanding outputs required in the project brief.
    raw.head(5).to_csv(RESULTS_DIR / "first_five_records.csv", index=False)
    overview = pd.DataFrame({
        "Column": raw.columns,
        "Data Type": [str(dtype) for dtype in raw.dtypes],
        "Missing Values": [int(value) for value in raw.isna().sum().values],
    })
    overview.to_csv(RESULTS_DIR / "dataset_columns_missing_values.csv", index=False)
    overview_text = [
        f"Dataset dimensions: {raw.shape[0]} rows × {raw.shape[1]} columns",
        f"Duplicate records: {int(raw.duplicated().sum())}",
        "\nFirst five records:",
        raw.head(5).to_string(index=False),
        "\nColumn types and missing values:",
        overview.to_string(index=False),
    ]
    (RESULTS_DIR / "dataset_overview.txt").write_text("\n".join(overview_text), encoding="utf-8")
    print("Dataset dimensions:", raw.shape)
    print("Duplicate records:", int(raw.duplicated().sum()))
    print("Data types and missing values:\n", overview.to_string(index=False))

    print("[2/7] Cleaning transaction records...")
    cleaned, cleaning_summary = clean_transactions(raw)
    with open(RESULTS_DIR / "cleaning_summary.json", "w", encoding="utf-8") as f:
        json.dump(cleaning_summary, f, indent=4)
    print("Cleaning summary:", cleaning_summary)

    print("[3/7] Creating customer-level RFM table...")
    rfm, snapshot_date = create_rfm(cleaned)
    rfm.to_csv(RESULTS_DIR / "rfm_before_clustering.csv")
    print(f"RFM table shape: {rfm.shape}")

    print("[4/7] Generating exploratory data analysis charts...")
    save_eda(cleaned, rfm)

    print("[5/7] Training and comparing clustering algorithms...")
    artifact, segmented, comparison, k_results, scaling_results = fit_and_compare(rfm)
    artifact["snapshot_date"] = str(snapshot_date.date())
    segmented.to_csv(RESULTS_DIR / "customer_segments.csv")
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    k_results.to_csv(RESULTS_DIR / "experiment_1_cluster_count.csv", index=False)
    scaling_results.to_csv(RESULTS_DIR / "experiment_2_scaling.csv", index=False)
    joblib.dump(artifact, MODEL_DIR / "customer_segmentation_kmeans.joblib")
    save_cluster_visualizations(segmented)

    print("[6/7] Saving segment summary and results...")
    summary = segmented.groupby(["Cluster", "Segment"]).agg(
        Customers=("Cluster", "size"),
        Median_Recency_Days=("Recency", "median"),
        Median_Frequency=("Frequency", "median"),
        Median_Monetary_GBP=("Monetary", "median"),
        Average_Monetary_GBP=("Monetary", "mean"),
    ).reset_index()
    summary.to_csv(RESULTS_DIR / "segment_summary.csv", index=False)

    run_summary = {
        "raw_rows": len(raw),
        "cleaned_transaction_rows": len(cleaned),
        "customers_segmented": len(segmented),
        "selected_number_of_clusters": int(artifact["selected_k"]),
        "snapshot_date": str(snapshot_date.date()),
        "final_deployed_algorithm": "K-Means",
        "second_algorithm_compared": "Hierarchical (Agglomerative) Clustering",
        "feature_columns": FEATURES,
    }
    with open(RESULTS_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=4)
    with open(RESULTS_DIR / "train_test_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(artifact["split_summary"], f, indent=4)

    report_path = write_project_report(ROOT, raw, cleaned, rfm, cleaning_summary, snapshot_date, artifact, comparison, k_results, scaling_results, summary)
    print(f"Project report saved to: {report_path}")

    print("\nModel comparison:\n", comparison.to_string(index=False))
    print("\nExperiment 1 - cluster count:\n", k_results.to_string(index=False))
    print("\nExperiment 2 - scaling:\n", scaling_results.to_string(index=False))
    print("\nSegment summary:\n", summary.to_string(index=False))
    print("\n[7/7] COMPLETE. Model and results saved.")
    print(f"Model: {MODEL_DIR / 'customer_segmentation_kmeans.joblib'}")
    print(f"Results folder: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
