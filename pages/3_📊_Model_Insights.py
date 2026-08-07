import streamlit as st
import pandas as pd
import os
import json


# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Model Insights",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 Model Insights")

st.markdown(
"""
Explainability, performance analysis and business impact
of the Grocery Reorder Prediction Model
"""
)


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


PLOT_DIR = os.path.join(
    BASE_DIR,
    "plots"
)


META_PATH = os.path.join(
    MODEL_DIR,
    "final_model_meta.json"
)



# -----------------------------
# Load Metadata
# -----------------------------

with open(META_PATH) as f:
    meta = json.load(f)



# -----------------------------
# KPI Cards
# -----------------------------

col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Model",
        "LightGBM"
    )


with col2:
    st.metric(
        "Model F1 Score",
        f'{meta["final_per_user_f1"]:.4f}'
    )


with col3:
    st.metric(
        "Baseline F1",
        f'{meta["baseline_f1"]:.4f}'
    )


with col4:
    st.metric(
        "Lift",
        f'{meta["total_lift_pct"]:.2f}%'
    )


st.divider()



# -----------------------------
# Feature Importance
# -----------------------------

st.header(
    "📊 Feature Importance"
)


feature_img = os.path.join(
    PLOT_DIR,
    "feature_importance.png"
)


if os.path.exists(feature_img):

    st.image(
        feature_img,
        use_container_width=True
    )

else:

    st.warning(
        "feature_importance.png not found"
    )



# -----------------------------
# Model Progression
# -----------------------------

st.header(
    "📈 Model Training Progress"
)


progress_img = os.path.join(
    PLOT_DIR,
    "model_progression.png"
)


if os.path.exists(progress_img):

    st.image(
        progress_img,
        use_container_width=True
    )



# -----------------------------
# Precision Recall Curve
# -----------------------------

st.header(
    "🎯 Precision Recall Analysis"
)


pr_img = os.path.join(
    PLOT_DIR,
    "pr_curve.png"
)


if os.path.exists(pr_img):

    st.image(
        pr_img,
        use_container_width=True
    )



# -----------------------------
# Activity Segment Analysis
# -----------------------------

st.header(
    "👥 Performance by User Activity"
)


activity_img = os.path.join(
    PLOT_DIR,
    "activity_segment.png"
)


if os.path.exists(activity_img):

    st.image(
        activity_img,
        use_container_width=True
    )



# -----------------------------
# Department Recall
# -----------------------------

st.header(
    "🏬 Recall by Department"
)


department_img = os.path.join(
    PLOT_DIR,
    "recall_by_department.png"
)


if os.path.exists(department_img):

    st.image(
        department_img,
        use_container_width=True
    )



# -----------------------------
# Feature Analysis
# -----------------------------

st.header(
    "🔎 Feature Analysis"
)


feature_table = pd.DataFrame(
{
"Feature Type":
[
"User Behaviour Features",
"Product Popularity Features",
"User-Product Interaction Features",
"Temporal Features",
"Categorical Features"
],

"Examples":
[
"user_total_orders, reorder_ratio",
"product_total_orders, product_reorder_rate",
"times_bought, up_order_rate",
"days_since_last_order, reorder_gap",
"aisle, department"
]
}
)


st.dataframe(
    feature_table,
    use_container_width=True
)



# -----------------------------
# Business Impact
# -----------------------------

st.header(
    "💼 Business Impact"
)


col1, col2 = st.columns(2)



with col1:

    st.success(
"""
✅ Improved reorder recommendations

✅ Reduced irrelevant suggestions

✅ Better customer personalization

✅ Higher prediction reliability
"""
)



with col2:

    st.info(
f"""
Baseline F1:

{meta["baseline_f1"]:.4f}


Final Model F1:

{meta["final_per_user_f1"]:.4f}


Performance Improvement:

{meta["total_lift_pct"]:.2f}%
"""
)



st.divider()


st.caption(
"Model: LightGBM | Metric: Per User F1 Score"
)