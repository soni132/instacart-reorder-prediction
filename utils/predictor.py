from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "deploy_sample.parquet"


# -----------------------------------------------------
# Load one user's data
# -----------------------------------------------------

def load_user_data(user_id: int):
    df = pd.read_parquet(DATA_PATH)

    user_df = df[df["user_id"] == user_id].copy()

    if user_df.empty:
        return None

    return user_df


# -----------------------------------------------------
# Predict reorder probabilities
# -----------------------------------------------------

def predict_user(model, user_df, threshold=0.20):

    if user_df is None or user_df.empty:
        return None

    feature_cols = [
        "times_bought",
        "user_total_orders",
        "up_order_rate",
        "up_orders_since_last",
        "up_order_rate_since_first",
        "up_reorder_rate",
        "up_avg_cart_position",
        "product_total_orders",
        "product_reorder_rate",
        "product_avg_cart_position",
        "product_distinct_users",
        "aisle_id",
        "department_id",
        "aisle",
        "department",
        "user_avg_days_between_orders",
        "user_std_days_between_orders",
        "user_avg_basket_size",
        "user_max_basket_size",
        "user_reorder_ratio",
        "user_distinct_products",
        "user_days_since_last_order",
        "user_reorder_streak",
        "user_unique_departments",
        "product_avg_days_between_reorders",
        "product_department_reorder_rate",
        "up_order_rate_last5",
        "up_gap_vs_typical",
    ]

    X = user_df[feature_cols].copy()

    # Convert categorical columns
    X["aisle"] = X["aisle"].astype("category")
    X["department"] = X["department"].astype("category")

    # Predict
    user_df["Probability"] = model.predict(X)

    user_df["Prediction"] = (
        user_df["Probability"] >= threshold
    ).astype(int)

    user_df = (
        user_df.sort_values(
            "Probability",
            ascending=False
        )
        .reset_index(drop=True)
    )

    output_cols = [
        "user_id",
        "product_id",
        "Probability",
        "Prediction",
    ]

    if "product_name" in user_df.columns:
        output_cols.insert(2, "product_name")

    return user_df[output_cols]