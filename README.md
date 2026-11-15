# Customer Segmentation Using Unsupervised Machine Learning

**Individual ML Project | Roll No. 51**  
**ML Category:** Unsupervised Learning – Clustering  
**Dataset:** UCI Online Retail  
**Primary algorithm:** K-Means  
**Comparison algorithm:** Hierarchical (Agglomerative) Clustering

## 1. Project overview

This project groups customers according to their historical purchasing behaviour. Transaction-level records are cleaned and aggregated into three customer-level features known as RFM:

- **Recency:** number of days since the customer's most recent purchase.
- **Frequency:** number of distinct invoices/orders made by the customer.
- **Monetary:** total transaction value for the customer, in GBP (£).

K-Means and Hierarchical (Agglomerative) Clustering are compared. K-Means is saved for the Streamlit application because it supports assigning new customer records to an existing cluster through `predict()`.

## 2. Dataset source

The project uses the UCI Machine Learning Repository's Online Retail dataset. It contains 541,909 transaction records from a UK-based online retailer between 1 December 2010 and 9 December 2011. See the official dataset page: https://archive.ics.uci.edu/dataset/352/online+retail

**Citation:** Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33. License: CC BY 4.0.

The dataset file is not included in this ZIP. Run the downloader below, or download `Online Retail.xlsx` from the official UCI page and place it in `Dataset/`.

## 3. Project structure

```text
Customer_Segmentation_Project/
├── Dataset/                   # Place Online Retail.xlsx here
├── Notebook/
│   └── customer_segmentation.ipynb
├── Source_Code/
│   ├── download_data.py        # Downloads the official UCI dataset
│   ├── ml_pipeline.py          # Cleaning, RFM, clustering and prediction functions
│   └── train_model.py          # Complete training/evaluation workflow
├── Trained_Model/              # Saved K-Means model (.joblib)
├── Front_End/
│   └── app.py                  # Streamlit application
├── Results/                    # CSV metrics, summaries and plots (generated)
├── Project_Documentation/
│   └── Project_Report.md       # Report (generated after model training)
├── requirements.txt
└── README.md
```

## 4. Setup and execution

Python 3.10 or newer is recommended.

### Step 1 – Install dependencies

Open a terminal in this project folder and run:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2 – Download the dataset

```bash
python Source_Code/download_data.py
```

### Step 3 – Train, evaluate and save the model

```bash
python Source_Code/train_model.py
```

This creates the trained model, customer segments, algorithm comparison, two experiment tables, split details, six visualization images and the project report.

### Step 4 – Launch the frontend

```bash
streamlit run Front_End/app.py
```

The application asks for a customer's Recency, Frequency and Monetary values, then displays the predicted cluster and its descriptive segment name.

## 5. Methodology

1. **Problem definition:** discover groups of customers with similar purchasing behaviour.
2. **Dataset understanding:** inspect dimensions, data types, missing values and duplicates.
3. **Preprocessing:** remove duplicates, records without CustomerID, cancellation invoices, non-positive quantities and non-positive unit prices; parse dates and create line revenue.
4. **RFM creation:** aggregate transaction rows to one record per CustomerID.
5. **EDA:** produce country, feature distribution, frequency-versus-monetary and RFM correlation visualizations.
6. **Feature selection:** use Recency, Frequency and Monetary as customer behaviour features; CustomerID is an identifier, not a model feature.
7. **Train/test split:** split customer RFM rows into 80% training and 20% testing with `random_state=42`.
8. **Model selection:** compare K-Means and Agglomerative Clustering.
9. **Evaluation:** use Silhouette Score, Davies-Bouldin Index and Calinski-Harabasz Score. These are internal clustering metrics because the dataset has no ground-truth segment labels.
10. **Improvement experiments:** vary the number of clusters (K) and compare StandardScaler with MinMaxScaler.
11. **Final model:** save K-Means, its scaler, preprocessing caps and segment-name mapping.
12. **Frontend/deployment:** use Streamlit for prediction and prepare the folder for GitHub.

## 6. Important implementation notes

- The data records are transactions, not ready-made customer profiles. The RFM aggregation creates one row per customer.
- Cancellations are identified by invoice numbers beginning with `C`, as described by UCI, and excluded from the positive-purchase analysis.
- The source dataset is in GBP, therefore Monetary and frontend monetary input are expressed in pounds (£), not INR.
- The 99th percentile cap is learned from training data only. `log1p` reduces skew and StandardScaler puts the three features on comparable scales.
- K-Means is trained on the 80% training customers. The held-out 20% is assigned by K-Means `predict()`.
- Agglomerative Clustering does not provide a native out-of-sample `predict()` method. The code fits it to the held-out 20% to inspect cluster structure there; this is not equivalent to predicting test labels from a model trained on the 80% split. This limitation is discussed in the report.
- Cluster IDs are arbitrary. The descriptive segment names are assigned from each cluster's RFM median profile and are not predefined target labels.

## 7. Generated outputs

Check the `Results/` folder after running training:

- `dataset_overview.txt`
- `first_five_records.csv`
- `dataset_columns_missing_values.csv`
- `cleaning_summary.json`
- `run_summary.json`
- `train_test_split_summary.json`
- `rfm_before_clustering.csv`
- `customer_segments.csv`
- `segment_summary.csv`
- `model_comparison.csv`
- `experiment_1_cluster_count.csv`
- `experiment_2_scaling.csv`
- `01_top_countries.png` through `06_segment_counts.png`

The report is generated at `Project_Documentation/Project_Report.md` after a successful training run.

## 8. Academic note

This is an educational customer analytics prototype. Clusters are patterns in historical transactions; they should not be interpreted as permanent labels or as a definitive judgement about a specific customer.
