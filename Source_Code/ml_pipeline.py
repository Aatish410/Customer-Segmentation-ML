"""Reusable data preparation and clustering functions for Customer Segmentation."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

FEATURES = ["Recency", "Frequency", "Monetary"]


def load_transactions(data_dir: Path) -> pd.DataFrame:
    """Load the UCI workbook or CSV downloaded from UCI."""
    xlsx = data_dir / "Online Retail.xlsx"
    csv = data_dir / "Online_Retail.csv"
    if xlsx.exists():
        df = pd.read_excel(xlsx)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(
            f"Dataset not found in {data_dir}. Run: python Source_Code/download_data.py"
        )

    # Standardize headers to avoid issues caused by spaces or capitalization.
    df.columns = [str(c).strip() for c in df.columns]
    required = {"InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean transactions and create line revenue (Quantity × UnitPrice)."""
    df = raw.copy()
    initial_rows = len(df)
    df = df.drop_duplicates().copy()
    duplicates_removed = initial_rows - len(df)

    # A CustomerID is necessary to assign transactions to a customer.
    missing_customer_rows = int(df["CustomerID"].isna().sum())
    df = df.dropna(subset=["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]).copy()

    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    cancellation_mask = df["InvoiceNo"].str.upper().str.startswith("C")
    cancelled_rows = int(cancellation_mask.sum())
    df = df.loc[~cancellation_mask].copy()

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", dayfirst=True, format="mixed")
    df["CustomerID"] = df["CustomerID"].astype(str).str.replace(r"\.0$", "", regex=True)
    df = df.dropna(subset=["Quantity", "UnitPrice", "InvoiceDate", "CustomerID"])

    invalid_value_rows = int(((df["Quantity"] <= 0) | (df["UnitPrice"] <= 0)).sum())
    df = df.loc[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    df["LineTotal"] = df["Quantity"] * df["UnitPrice"]

    if df.empty:
        raise ValueError("No valid transaction rows remain after cleaning. Check the dataset file.")

    cleaning_summary = {
        "initial_rows": initial_rows,
        "duplicates_removed": duplicates_removed,
        "rows_with_missing_customer_id": missing_customer_rows,
        "cancelled_invoice_rows_removed": cancelled_rows,
        "rows_with_nonpositive_quantity_or_price_removed": invalid_value_rows,
        "cleaned_rows": len(df),
        "unique_customers": int(df["CustomerID"].nunique()),
        "unique_invoices": int(df["InvoiceNo"].nunique()),
    }
    return df, cleaning_summary


def create_rfm(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Create one Recency-Frequency-Monetary record per customer."""
    snapshot_date = cleaned["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    rfm = cleaned.groupby("CustomerID").agg(
        LastPurchaseDate=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("LineTotal", "sum"),
    )
    rfm["Recency"] = (snapshot_date - rfm["LastPurchaseDate"].dt.normalize()).dt.days
    rfm = rfm[FEATURES].copy()
    rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna()
    rfm = rfm[(rfm[FEATURES] >= 0).all(axis=1)]
    return rfm, snapshot_date


def cap_and_log_transform(rfm: pd.DataFrame, caps: dict | None = None):
    """Cap extreme high values and apply log1p to reduce right-skew."""
    values = rfm[FEATURES].copy().astype(float)
    if caps is None:
        caps = {col: float(values[col].quantile(0.99)) for col in FEATURES}
    for col in FEATURES:
        values[col] = values[col].clip(upper=caps[col])
    transformed = np.log1p(values)
    return transformed, caps


def clustering_metrics(X, labels) -> dict:
    """Calculate internal clustering scores (higher silhouette/CH, lower DB is better)."""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return {"Silhouette Score": np.nan, "Davies-Bouldin Index": np.nan,
                "Calinski-Harabasz Score": np.nan}
    return {
        "Silhouette Score": float(silhouette_score(X, labels)),
        "Davies-Bouldin Index": float(davies_bouldin_score(X, labels)),
        "Calinski-Harabasz Score": float(calinski_harabasz_score(X, labels)),
    }


def build_segment_names(rfm_with_cluster: pd.DataFrame) -> dict:
    """Assign plain-English labels using each cluster's RFM median profile."""
    overall = rfm_with_cluster[FEATURES].median()
    cluster_medians = rfm_with_cluster.groupby("Cluster")[FEATURES].median()
    names = {}
    for cluster, row in cluster_medians.iterrows():
        recent = row["Recency"] <= overall["Recency"]
        frequent = row["Frequency"] >= overall["Frequency"]
        high_spend = row["Monetary"] >= overall["Monetary"]
        if recent and frequent and high_spend:
            name = "Loyal / High-Value Customers"
        elif not recent and not frequent:
            name = "At-Risk / Low-Engagement Customers"
        elif frequent:
            name = "Frequent Customers"
        elif high_spend:
            name = "High-Spending Customers"
        else:
            name = "Occasional Customers"
        names[int(cluster)] = name
    return names


def fit_and_compare(rfm: pd.DataFrame, random_state: int = 42):
    """Split customers 80/20, compare algorithms, and run two experiments."""
    train_rfm, test_rfm = train_test_split(
        rfm, test_size=0.20, random_state=random_state
    )

    # Learn the outlier caps and scaler from training customers only.
    train_transformed, caps = cap_and_log_transform(train_rfm)
    test_transformed, _ = cap_and_log_transform(test_rfm, caps=caps)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_transformed)
    X_test = scaler.transform(test_transformed)

    # Experiment 1: compare different K values using training-data silhouette scores.
    k_rows = []
    candidate_ks = range(2, min(9, len(train_rfm)))
    for k in candidate_ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        train_labels = km.fit_predict(X_train)
        train_scores = clustering_metrics(X_train, train_labels)
        test_labels = km.predict(X_test)
        test_scores = clustering_metrics(X_test, test_labels)
        k_rows.append({
            "K": k,
            "Train Silhouette": train_scores["Silhouette Score"],
            "Test Silhouette": test_scores["Silhouette Score"],
            "Train Davies-Bouldin": train_scores["Davies-Bouldin Index"],
            "Test Davies-Bouldin": test_scores["Davies-Bouldin Index"],
        })
    k_results = pd.DataFrame(k_rows)
    if k_results.empty or k_results["Train Silhouette"].isna().all():
        selected_k = 3
    else:
        selected_k = int(k_results.loc[k_results["Train Silhouette"].idxmax(), "K"])

    # Final K-Means model is learned only from the training split.
    kmeans = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
    train_kmeans_labels = kmeans.fit_predict(X_train)
    test_kmeans_labels = kmeans.predict(X_test)
    kmeans_test_scores = clustering_metrics(X_test, test_kmeans_labels)

    # Agglomerative clustering has no native predict() for unseen records.
    # Therefore it is fitted to the held-out customers only for a test-set
    # cluster-structure assessment; this limitation is disclosed in the report.
    agglomerative = AgglomerativeClustering(n_clusters=selected_k, linkage="ward")
    agg_test_labels = agglomerative.fit_predict(X_test)
    agg_test_scores = clustering_metrics(X_test, agg_test_labels)
    comparison = pd.DataFrame([
        {"Algorithm": "K-Means (trained on 80%; predicted on 20% test)", **kmeans_test_scores},
        {"Algorithm": "Hierarchical (fit on 20% test for structure assessment)", **agg_test_scores},
    ])

    # Experiment 2: compare StandardScaler with MinMaxScaler on the same holdout.
    minmax = MinMaxScaler()
    X_train_minmax = minmax.fit_transform(train_transformed)
    X_test_minmax = minmax.transform(test_transformed)
    minmax_model = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
    minmax_model.fit(X_train_minmax)
    minmax_test_labels = minmax_model.predict(X_test_minmax)
    standard_scores = clustering_metrics(X_test, test_kmeans_labels)
    minmax_scores = clustering_metrics(X_test_minmax, minmax_test_labels)
    scaling_results = pd.DataFrame([
        {"Scaling Method": "StandardScaler", **standard_scores},
        {"Scaling Method": "MinMaxScaler", **minmax_scores},
    ])

    # Apply the trained model to all customers for the customer-facing segment report.
    all_transformed, _ = cap_and_log_transform(rfm, caps=caps)
    X_all = scaler.transform(all_transformed)
    all_labels = kmeans.predict(X_all)
    rfm_result = rfm.copy()
    rfm_result["Cluster"] = all_labels
    segment_names = build_segment_names(rfm_result)
    rfm_result["Segment"] = rfm_result["Cluster"].map(segment_names)

    split_summary = {
        "training_customers": int(len(train_rfm)),
        "testing_customers": int(len(test_rfm)),
        "train_test_ratio": "80:20",
        "random_state": random_state,
        "note": "K-Means is fitted on training customers and predicts test customers. Agglomerative clustering has no native predict method; it is fitted on the held-out test subset only to assess the cluster structure there.",
    }
    artifact = {
        "features": FEATURES,
        "caps": caps,
        "scaler": scaler,
        "kmeans": kmeans,
        "selected_k": selected_k,
        "segment_names": segment_names,
        "snapshot_date": None,
        "transform": "upper-cap at training 99th percentile, log1p, StandardScaler",
        "split_summary": split_summary,
    }
    return artifact, rfm_result, comparison, k_results, scaling_results


def predict_one_customer(recency: float, frequency: float, monetary: float, artifact: dict):
    """Assign a new customer's RFM values to a trained K-Means cluster."""
    row = pd.DataFrame([[recency, frequency, monetary]], columns=artifact["features"])
    for col in artifact["features"]:
        row[col] = row[col].clip(upper=artifact["caps"][col])
    transformed = np.log1p(row)
    X = artifact["scaler"].transform(transformed)
    cluster = int(artifact["kmeans"].predict(X)[0])
    return cluster, artifact["segment_names"].get(cluster, f"Cluster {cluster}")
