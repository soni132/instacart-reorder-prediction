from pathlib import Path
import json

import lightgbm as lgb
import pandas as pd
import streamlit as st

# -----------------------------------------------------
# Project Root
# -----------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------
# Load Trained LightGBM Model
# -----------------------------------------------------

@st.cache_resource
def load_model():
    """Load trained LightGBM model."""

    model_path = ROOT / "models" / "lgb_final_tuned.txt"

    return lgb.Booster(model_file=str(model_path))


# -----------------------------------------------------
# Load Product Lookup Table
# -----------------------------------------------------

@st.cache_data
def load_products():
    """Load product lookup."""

    return pd.read_csv(
        ROOT / "data" / "lookup" / "products.csv"
    )


# -----------------------------------------------------
# Load Activity Segment Metrics
# -----------------------------------------------------

@st.cache_data
def load_activity():
    """Load activity segment statistics."""

    return pd.read_csv(
        ROOT / "data" / "processed" / "segment_by_activity_final.csv"
    )


# -----------------------------------------------------
# Load Department Metrics
# -----------------------------------------------------

@st.cache_data
def load_department():
    """Load department performance statistics."""

    return pd.read_csv(
        ROOT / "data" / "processed" / "segment_by_department_final.csv"
    )


# -----------------------------------------------------
# Load Model Metadata
# -----------------------------------------------------

@st.cache_data
def load_meta():
    """Load model metadata."""

    with open(ROOT / "models" / "final_model_meta.json") as f:
        return json.load(f)


# -----------------------------------------------------
# Load Available Customer IDs
# -----------------------------------------------------

@st.cache_data
def load_user_ids():
    data_path = ROOT / "data" / "processed" / "deploy_sample.parquet"
    return sorted(
        pd.read_parquet(data_path, columns=["user_id"])["user_id"]
        .unique()
        .tolist()
    )
