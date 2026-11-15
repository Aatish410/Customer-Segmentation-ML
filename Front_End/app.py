"""Streamlit interface for the trained Customer Segmentation model."""
from pathlib import Path
import sys
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "Trained_Model" / "customer_segmentation_kmeans.joblib"
RESULTS_DIR = ROOT / "Results"
sys.path.insert(0, str(ROOT / "Source_Code"))
from ml_pipeline import predict_one_customer

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")
st.title("🛍️ Customer Segmentation System")
st.write("Group customers by purchasing behaviour using an Unsupervised Machine Learning model.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. First run: python Source_Code/train_model.py")
    st.stop()

artifact = joblib.load(MODEL_PATH)
st.caption(f"Model: K-Means | Number of clusters: {artifact['selected_k']} | Features: Recency, Frequency, Monetary")

left, right = st.columns([1, 1])
with left:
    st.subheader("Enter customer information")
    with st.form("customer_form"):
        recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=5000, value=30, step=1)
        frequency = st.number_input("Frequency (number of distinct invoices)", min_value=1, max_value=10000, value=3, step=1)
        monetary = st.number_input("Monetary value (£ total spending)", min_value=0.01, max_value=10000000.0, value=500.0, step=50.0)
        submitted = st.form_submit_button("Predict Customer Segment", type="primary", use_container_width=True)
    st.info("Use customer-level values. Monetary value is in GBP (£), matching the source dataset.")

with right:
    st.subheader("Prediction")
    if submitted:
        cluster, segment = predict_one_customer(recency, frequency, monetary, artifact)
        st.success(f"Predicted segment: {segment}")
        st.metric("Assigned cluster", f"Cluster {cluster}")
        st.write("**Submitted RFM profile**")
        st.dataframe(pd.DataFrame({"Feature": ["Recency (days)", "Frequency (invoices)", "Monetary (£)"],
                                   "Value": [recency, frequency, monetary]}), hide_index=True, use_container_width=True)
    else:
        st.write("Enter the customer's RFM values and select **Predict Customer Segment**.")

st.divider()
st.subheader("Project results")
summary_path = RESULTS_DIR / "segment_summary.csv"
comparison_path = RESULTS_DIR / "model_comparison.csv"
if summary_path.exists():
    summary = pd.read_csv(summary_path)
    c1, c2 = st.columns(2)
    c1.metric("Customers analysed", f"{int(summary['Customers'].sum()):,}")
    c2.metric("Customer segments", f"{summary['Cluster'].nunique()}")
    st.write("**Segment profile**")
    st.dataframe(summary, hide_index=True, use_container_width=True)
if comparison_path.exists():
    st.write("**Clustering algorithm comparison**")
    st.dataframe(pd.read_csv(comparison_path).round(4), hide_index=True, use_container_width=True)

st.caption("Academic demonstration only. Customer groups are discovered from historical transaction patterns and should not be treated as fixed facts about an individual.")
