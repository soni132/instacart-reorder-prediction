import streamlit as st

from utils.loader import (
    load_model,
    load_meta,
    load_user_ids,
)

from utils.predictor import (
    load_user_data,
    predict_user,
)

from utils.helper import (
    format_prediction_table,
    show_metrics,
)

# ==========================================================
# Page Configuration
# ==========================================================

st.title("🛒 Grocery Reorder Prediction")

st.markdown("""
Predict which products a customer is most likely to reorder in their next grocery purchase using a trained **LightGBM** model.
""")

# ==========================================================
# Load Resources
# ==========================================================

model = load_model()
meta = load_meta()
user_ids = load_user_ids()

best_threshold = meta["best_threshold"]

# ==========================================================
# Model Information
# ==========================================================

st.subheader("📌 Model Information")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Model",
    "LightGBM"
)

col2.metric(
    "Per-User F1",
    f"{meta['final_per_user_f1']:.3f}"
)

col3.metric(
    "Threshold",
    f"{best_threshold:.2f}"
)

col4.metric(
    "Lift",
    f"{meta['total_lift_pct']:.1f}%"
)

st.caption(
    f"""
The deployed model uses an optimized probability threshold of **{best_threshold:.2f}**
selected during validation to maximize **Per-User F1**.
"""
)

st.divider()

# ==========================================================
# Customer Selection
# ==========================================================

st.subheader("👤 Customer Selection")

with st.form("prediction_form"):

    user_id = st.selectbox(
        "Select Customer",
        user_ids,
        help="Customers are sampled from the validation dataset."
    )

    predict_btn = st.form_submit_button(
        "🔍 Predict Reorders",
        use_container_width=True
    )

# ==========================================================
# Prediction
# ==========================================================

if predict_btn:

    with st.spinner("Running LightGBM model..."):

        user_df = load_user_data(user_id)

        if user_df is None:

            st.error("Customer not found.")

        else:

            result = predict_user(
                model,
                user_df,
                best_threshold
            )

            result = format_prediction_table(result)

            show_metrics(result)

            predicted = result[
                result["Prediction"] == "✅ Reorder"
            ]

            st.divider()

            st.success(
                f"Recommended **{len(predicted)}** products out of **{len(result)}** previously purchased products."
            )

            st.subheader("🛍️ Recommended Products")

            if predicted.empty:

                st.warning(
                    "No products exceeded the prediction threshold."
                )

            else:

                st.dataframe(
                    predicted,
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander("📋 View All Predictions"):

                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True,
                )