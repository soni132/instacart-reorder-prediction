import streamlit as st

from utils.loader import load_meta

# =====================================================
# Page Config
# =====================================================

st.set_page_config(
    page_title="AI-powered Grocery Reorder Prediction",
    page_icon="🛒",
    layout="wide"
)

meta = load_meta()

# =====================================================
# Hero Section
# =====================================================

st.title("🛒 AI-powered Grocery Reorder Prediction System")

st.markdown("""
### Predict which grocery products a customer is most likely to reorder using Machine Learning.

This project builds an end-to-end recommendation pipeline using the **Instacart Market Basket Analysis** dataset.
A LightGBM model learns purchasing behavior from historical orders and predicts products a customer is likely to buy again in their next shopping trip.
""")

st.divider()

# =====================================================
# Project Metrics
# =====================================================

st.subheader("📌 Project Highlights")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Per-User F1",
    f"{meta['final_per_user_f1']:.3f}"
)

c2.metric(
    "Lift over Baseline",
    f"{meta['total_lift_pct']:.1f}%"
)

c3.metric(
    "Decision Threshold",
    f"{meta['best_threshold']:.2f}"
)

c4.metric(
    "ML Model",
    "LightGBM"
)

st.divider()

# =====================================================
# Problem Statement
# =====================================================

st.subheader("🎯 Problem Statement")

st.write("""
Retail companies benefit from knowing which products customers are likely to reorder.

Instead of recommending completely new products, this project predicts **which previously purchased products**
will appear in the customer's **next grocery basket**.

These predictions can improve:

- Personalized recommendations
- Reminder notifications
- Inventory planning
- Customer retention
""")

# =====================================================
# Dataset
# =====================================================

st.subheader("📦 Dataset")

st.markdown("""
**Dataset:** Instacart Market Basket Analysis (Kaggle)

- 🛍 3.4 Million Orders
- 👥 206,000+ Customers
- 🥦 50,000+ Products
- 🧾 Multiple Departments & Aisles

Each row in the training dataset represents a **(Customer, Product)** pair,
where the model predicts whether that product will be reordered in the customer's next purchase.
""")

st.divider()

# =====================================================
# Machine Learning Pipeline
# =====================================================

st.subheader("⚙️ Machine Learning Pipeline")

st.markdown("""
1. Load raw Instacart order history

2. Build candidate (User, Product) pairs

3. Generate behavioral features

- User Features
- Product Features
- User-Product Interaction Features

4. Train LightGBM Classifier

5. Optimize Threshold

6. Evaluate using Per-User F1

7. Analyze Feature Importance

8. Deploy using Streamlit
""")

st.divider()

# =====================================================
# Features
# =====================================================

st.subheader("📊 Feature Engineering")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("### 👤 User")

    st.markdown("""
- Total Orders
- Basket Size
- Reorder Ratio
- Shopping Frequency
- Days Since Last Order
""")

with col2:

    st.markdown("### 🥦 Product")

    st.markdown("""
- Total Orders
- Reorder Rate
- Average Cart Position
- Department
- Aisle
""")

with col3:

    st.markdown("### 🔄 User-Product")

    st.markdown("""
- Times Bought
- Order Rate
- Last 5 Order Rate
- Gap vs Typical
- Average Cart Position
""")

st.divider()

# =====================================================
# Results
# =====================================================

st.subheader("🏆 Final Results")

st.success(f"""
✅ Per-User F1 : **{meta['final_per_user_f1']:.3f}**

✅ Lift over Baseline : **{meta['total_lift_pct']:.1f}%**

✅ Optimized Threshold : **{meta['best_threshold']:.2f}**
""")

st.write("""
The model significantly outperformed a simple reorder heuristic by learning
customer purchasing patterns through engineered behavioral features.

Feature engineering contributed more to performance improvements than
hyperparameter tuning, highlighting the importance of informative features.
""")

st.divider()

# =====================================================
# Navigation
# =====================================================

st.subheader("🧭 Explore the Project")

st.info("""
📌 **Predict**

Use the trained model to predict which products a customer is likely to reorder.

📌 **Model Insights**

Understand LightGBM, feature importance, decision trees, and training results.

📌 **Data Insights**

Explore customer behavior, department-wise recall, and activity analysis.

📌 **About**

Project summary, technologies used, and developer information.
""")

st.divider()

# =====================================================
# Footer
# =====================================================

st.caption(
    "Developed by Soni Kumari | AI-powered Grocery Reorder Prediction System"
)