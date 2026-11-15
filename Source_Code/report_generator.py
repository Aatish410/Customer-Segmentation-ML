"""Create the final Markdown project report from actual training outputs."""
from pathlib import Path
import pandas as pd


def write_project_report(root: Path, raw: pd.DataFrame, cleaned: pd.DataFrame,
                         rfm: pd.DataFrame, cleaning: dict, snapshot_date,
                         artifact: dict, comparison: pd.DataFrame,
                         k_results: pd.DataFrame, scaling_results: pd.DataFrame,
                         segment_summary: pd.DataFrame):
    docs = root / "Project_Documentation"
    docs.mkdir(exist_ok=True)
    countries = cleaned["Country"].value_counts()
    top_country = str(countries.index[0]) if len(countries) else "Not available"
    top_share = (countries.iloc[0] / len(cleaned) * 100) if len(countries) else 0
    corr = rfm["Frequency"].corr(rfm["Monetary"])
    skew = rfm["Monetary"].skew()

    comp = comparison.round(4).to_string(index=False)
    kvals = k_results.round(4).to_string(index=False)
    scales = scaling_results.round(4).to_string(index=False)
    segments = segment_summary.round(2).to_string(index=False)

    report = f"""# Customer Segmentation Using Unsupervised Machine Learning

**Individual ML Project | Roll Number: 51**  
**Project type:** Unsupervised Learning – Clustering  
**Dataset:** UCI Online Retail  
**Final deployed algorithm:** K-Means

## 1. Aim

To segment customers into meaningful groups using unsupervised Machine Learning based on their purchasing behaviour.

## 2. Problem statement

An online retailer has many customers with different purchasing patterns. Analysing each customer manually is difficult. This project uses transaction history to group customers with similar recency, purchase frequency and total spending so that the business can understand broad customer behaviour patterns.

## 3. Objectives

1. Understand and analyse the Online Retail transaction dataset.
2. Clean the data and prepare customer-level features.
3. Perform exploratory data analysis using visualizations.
4. Apply K-Means and Hierarchical (Agglomerative) Clustering.
5. Compare clustering quality using appropriate internal metrics.
6. Experiment with the number of clusters and feature scaling.
7. Save a model and build a simple user interface to assign a new RFM profile to a customer segment.

## 4. Dataset information

The UCI Online Retail dataset contained **{len(raw):,} raw transaction rows** in this run. The official UCI description states that it records transactions from a UK-based online retailer between 1 December 2010 and 9 December 2011. The source includes invoice number, product code, product description, quantity, invoice date, unit price, customer ID and country.

**Source:** https://archive.ics.uci.edu/dataset/352/online+retail  
**Citation:** Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33. License: CC BY 4.0.

## 5. Problem type

This is an **Unsupervised Learning – Clustering** problem. The data does not provide predefined customer segment labels. The algorithm discovers groups based on similarities in RFM features.

## 6. Data preprocessing

The cleaning process produced **{len(cleaned):,} valid transaction rows** from {len(raw):,} raw rows. The decisions were:

- Removed duplicate transaction rows.
- Removed rows without a CustomerID because they cannot be assigned to a customer profile.
- Removed cancellation invoices, identified by invoice numbers beginning with `C`.
- Removed non-positive quantities and non-positive unit prices to focus on positive purchase transactions.
- Converted invoice dates to datetime and numeric transaction fields to numeric values.
- Calculated `LineTotal = Quantity × UnitPrice` for each transaction row.

### Cleaning summary

| Item | Value |
|---|---:|
| Raw rows | {cleaning['initial_rows']:,} |
| Duplicate rows removed | {cleaning['duplicates_removed']:,} |
| Rows with missing CustomerID | {cleaning['rows_with_missing_customer_id']:,} |
| Cancellation rows removed | {cleaning['cancelled_invoice_rows_removed']:,} |
| Rows with non-positive quantity/price removed | {cleaning['rows_with_nonpositive_quantity_or_price_removed']:,} |
| Cleaned transaction rows | {cleaning['cleaned_rows']:,} |
| Unique customers | {cleaning['unique_customers']:,} |
| Unique invoices | {cleaning['unique_invoices']:,} |

## 7. Feature and target selection

For this unsupervised task there is no labelled target variable `y`. CustomerID is used only to group transactions and is not supplied to the clustering model. The three input features are:

| Feature | Meaning |
|---|---|
| Recency | Number of days since the customer's last purchase. Lower values mean a more recent purchase. |
| Frequency | Number of distinct invoices/orders for the customer. |
| Monetary | Sum of positive line values for the customer, measured in GBP (£). |

The RFM table contains **{len(rfm):,} customers**. The reference snapshot date for Recency is **{snapshot_date.date()}** (one day after the latest transaction date in the cleaned dataset).

## 8. Train-test split

Customer-level RFM records were divided into an **80% training set and 20% testing set**, using `random_state=42` for reproducibility. In this run, **{artifact['split_summary']['training_customers']:,}** customers were used for training and **{artifact['split_summary']['testing_customers']:,}** for testing.

The scaler and 99th-percentile upper caps were learned from training customers only. `log1p` was applied to reduce right-skew, followed by StandardScaler.

## 9. Exploratory data analysis

The training script saves six figures in the `Results/` folder:

1. `01_top_countries.png` – transaction counts for the ten most represented countries.
2. `02_rfm_distributions.png` – distributions of Recency, Frequency and Monetary.
3. `03_frequency_vs_monetary.png` – relationship between invoice frequency and total spending.
4. `04_rfm_correlation.png` – correlations among the three RFM features.
5. `05_customer_clusters.png` – customer clusters by Frequency and Monetary.
6. `06_segment_counts.png` – number of customers assigned to each descriptive segment.

### Observations from this dataset run

- The country with the most cleaned transaction rows is **{top_country}**, representing approximately **{top_share:.2f}%** of cleaned rows.
- Median customer Recency is **{rfm['Recency'].median():.1f} days**.
- The correlation between Frequency and Monetary is **{corr:.3f}**.
- Monetary-value skewness is **{skew:.3f}**. A positive high value indicates that a smaller number of customers spend substantially more than the majority.

## 10. Model selection and algorithms

Two clustering algorithms were implemented:

1. **K-Means:** partitions customers into K groups by minimizing within-cluster squared distances. K was selected by comparing training silhouette scores for values from 2 through 8 (subject to available training records).
2. **Hierarchical (Agglomerative) Clustering:** builds clusters by merging observations according to linkage distance. Ward linkage was used.

### Train/test evaluation note

K-Means was fitted on the 80% training customers and then used to assign the 20% held-out test customers. Agglomerative Clustering does not provide a native `predict()` method for new observations. Therefore, it is fitted to the held-out test records themselves to assess the cluster structure present in that subset. This is a limitation and is not identical to out-of-sample prediction from a model fitted on the training subset.

## 11. Model evaluation

There are no ground-truth segment labels, so internal clustering metrics were used:

- **Silhouette Score:** closer to 1 generally indicates better separation and cohesion.
- **Davies-Bouldin Index:** lower values generally indicate better-separated clusters.
- **Calinski-Harabasz Score:** higher values generally indicate denser and better-separated clusters.

### Algorithm comparison on the held-out customer subset

```text
{comp}
```

These are internal structure scores, not classification accuracy. Interpret the Agglomerative result in light of the method limitation described above.

## 12. Model improvement experiments

### Experiment 1 – Change the number of clusters

K was varied from 2 to 8, and training silhouette scores were compared. The selected K was **{artifact['selected_k']}**, chosen by the highest training silhouette score among tested values.

```text
{kvals}
```

### Experiment 2 – Compare feature scaling methods

StandardScaler and MinMaxScaler were compared while keeping the selected K and train/test split fixed.

```text
{scales}
```

The results show how scaling affects distance-based clustering. The deployed pipeline uses StandardScaler, fitted on training data.

## 13. Final model selection

**K-Means with K = {artifact['selected_k']}** was saved as the final model. K-Means is suitable for this application because it provides a `predict()` operation for assigning new customer RFM inputs to one of the learned clusters. The selected scaler, feature caps, K-Means estimator and descriptive segment-name mapping are stored together in `Trained_Model/customer_segmentation_kmeans.joblib`.

Descriptive names are assigned after clustering by examining cluster-level RFM medians. Cluster IDs themselves are arbitrary and have no built-in business meaning.

### Customer segment summary

```text
{segments}
```

## 14. Front-end application

A Streamlit interface is provided in `Front_End/app.py`. It accepts Recency (days), Frequency (distinct invoice count) and Monetary (total spending in GBP), then displays the predicted cluster and descriptive segment name.

Run with:

```bash
streamlit run Front_End/app.py
```

## 15. Limitations and future improvements

- The dataset covers a historical period in 2010–2011; current customer behaviour may differ.
- Rows without CustomerID cannot be used for customer-level segmentation.
- Cancellations/returns are excluded from this positive-purchase RFM exercise.
- RFM uses only three features and may not capture product preferences, demographics or customer lifetime value.
- K-Means is sensitive to feature scaling, outliers and selected K.
- Hierarchical Clustering does not naturally assign future unseen customers without an additional assignment strategy.
- Future work could test additional features, other clustering algorithms, more recent data, and cluster stability over time.

## 16. Conclusion

This project demonstrates an end-to-end unsupervised learning workflow: transaction cleaning, customer-level RFM feature engineering, EDA, an 80/20 split, clustering algorithm comparison, improvement experiments, final model saving and a Streamlit prediction interface. The results in this report are generated from the dataset processed in this run.

---

**Student:** [Enter your name]  
**Roll Number:** 51  
**Class:** Third Year B.Tech. Computer Science and Engineering  
**Academic Year:** 2026–27
"""
    (docs / "Project_Report.md").write_text(report, encoding="utf-8")
    return docs / "Project_Report.md"
