import streamlit as st
import pandas as pd
import plotly.express as px
import os


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Grocery Reorder Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)



# -----------------------------------
# Color System
# -----------------------------------

ACCENT_USER = "#2563EB"
ACCENT_PRODUCT = "#10B981"
ACCENT_CATEGORY = "#7C3AED"
ACCENT_FEATURE = "#F59E0B"

SEQ_PALETTE_GREEN = px.colors.sequential.Greens_r
SEQ_PALETTE_PURPLE = px.colors.sequential.Purples_r
CAT_PALETTE = px.colors.qualitative.Set2

PLOTLY_TEMPLATE = "plotly_white"



# -----------------------------------
# Styling
# -----------------------------------

st.markdown(
"""
<style>


/* Sidebar */

[data-testid="stSidebar"] {

    background:#0f172a;

}


[data-testid="stSidebar"] * {

    color:white;

}


[data-testid="stSidebar"] hr {

    border-color:#334155;

}


[data-testid="stSidebar"] .stCaption {

    color:#94a3b8;

}



/* Hero */

.hero-title {

    font-size:42px;

    font-weight:900;

    background:
    linear-gradient(
    135deg,
    #2563EB,
    #7C3AED
    );

    -webkit-background-clip:text;

    -webkit-text-fill-color:transparent;

}



.hero-text {

    font-size:18px;

    color:#64748b;

}



/* KPI */

.kpi-card {

    background:white;

    padding:20px;

    border-radius:15px;

    border-left:6px solid var(--accent);

    box-shadow:
    0px 4px 12px rgba(0,0,0,0.08);

    text-align:center;

}



.kpi-icon {

    font-size:32px;

}


.kpi-title {

    color:#475569;

    font-size:16px;

}


.kpi-value {

    color:#111827;

    font-size:30px;

    font-weight:800;

}



/* Feature */

.feature-card {

    background:white;

    padding:18px;

    border-radius:15px;

    border-left:5px solid var(--accent);

    box-shadow:
    0px 3px 10px rgba(0,0,0,0.08);

}


.feature-card h4 {

    color:#111827;

}


.feature-card p {

    color:#374151;

}


</style>
""",
unsafe_allow_html=True
)



# -----------------------------------
# Sidebar
# -----------------------------------











# -----------------------------------
# Hero
# -----------------------------------

st.markdown(
"""
<h1 style="font-size:42px; font-weight:800; margin-bottom:0;">
<span style="color:#2563EB;">📊</span>
<span class="hero-title"> Grocery Reorder Data Insights</span>
</h1>
""",
unsafe_allow_html=True
)


st.markdown(
"""
<div class="hero-text">

Exploratory dashboard for customer behaviour,
product popularity, category trends and engineered features.

</div>
""",
unsafe_allow_html=True
)


st.write("")



# -----------------------------------
# Load Data
# -----------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "deploy_sample.parquet"
)

@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)

df = load_data()



st.success(
    f"Dataset Loaded: {df.shape[0]:,} rows × {df.shape[1]} features"
)



# -----------------------------------
# KPI Cards
# -----------------------------------

st.subheader("📌 Dataset Summary")


c1,c2,c3,c4 = st.columns(4)


metrics = [

("👥","Users",f"{df.user_id.nunique():,}",ACCENT_USER),

("🛒","Products",f"{df.product_id.nunique():,}",ACCENT_PRODUCT),

("🔗","User Product Pairs",f"{len(df):,}",ACCENT_CATEGORY),

("🏬","Departments",f"{df.department.nunique():,}",ACCENT_FEATURE)

]



for col,(icon,title,value,color) in zip(
    [c1,c2,c3,c4],
    metrics
):

    with col:

        st.markdown(
        f"""

        <div class="kpi-card"
        style="--accent:{color}">

        <div class="kpi-icon">
        {icon}
        </div>


        <div class="kpi-title">
        {title}
        </div>


        <div class="kpi-value">
        {value}
        </div>


        </div>

        """,
        unsafe_allow_html=True
        )



st.divider()



# -----------------------------------
# Tabs
# -----------------------------------

tab1,tab2,tab3,tab4 = st.tabs(
[
"👥 User Behaviour",
"🛒 Product Insights",
"🏬 Category Analysis",
"🔎 Feature Engineering"
]
)



# -----------------------------------
# User Behaviour
# -----------------------------------

with tab1:


    st.subheader(
        "👥 Customer Behaviour"
    )


    col1,col2 = st.columns(2)


    with col1:

        fig = px.histogram(
            df,
            x="user_total_orders",
            nbins=40,
            title="User Total Orders",
            color_discrete_sequence=[ACCENT_USER],
            template=PLOTLY_TEMPLATE
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )



    with col2:


        fig = px.histogram(
            df,
            x="user_reorder_ratio",
            nbins=40,
            title="User Reorder Ratio",
            color_discrete_sequence=[ACCENT_CATEGORY],
            template=PLOTLY_TEMPLATE
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )



# -----------------------------------
# Product
# -----------------------------------

with tab2:


    st.subheader(
        "🛒 Product Behaviour"
    )


    top_products=(

        df.groupby("product_name")
        ["product_total_orders"]
        .max()
        .sort_values(
            ascending=False
        )
        .head(15)
        .reset_index()

    )


    fig=px.bar(

        top_products,

        x="product_total_orders",

        y="product_name",

        orientation="h",

        title="Top Products",

        color="product_total_orders",

        color_continuous_scale=SEQ_PALETTE_GREEN

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



# -----------------------------------
# Category
# -----------------------------------

with tab3:


    st.subheader(
        "🏬 Category Analysis"
    )


    dept=(

        df.groupby("department")
        .size()
        .reset_index(
            name="count"
        )
        .sort_values(
            "count",
            ascending=False
        )

    )


    fig=px.bar(

        dept,

        x="department",

        y="count",

        color="department",

        title="Department Distribution",

        color_discrete_sequence=CAT_PALETTE

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



# -----------------------------------
# Feature Engineering
# -----------------------------------

with tab4:


    st.subheader(
        "🔎 Feature Engineering"
    )


    features=[

    ("👤 User Behaviour",
    "user_total_orders, reorder ratio, basket size"),

    ("🛒 Product Behaviour",
    "product popularity, reorder rate"),

    ("🔄 User Product Interaction",
    "times bought, up order rate"),

    ("⏳ Temporal Features",
    "days since last order, reorder gap"),

    ("🏷️ Category Features",
    "aisle and department information")

    ]



    for title,desc in features:

        st.markdown(
        f"""

        <div class="feature-card"
        style="--accent:#2563EB">

        <h4>{title}</h4>

        <p>{desc}</p>

        </div>

        <br>

        """,
        unsafe_allow_html=True
        )



st.divider()


st.caption(
"Dataset: Instacart Market Basket Analysis | Feature Engineering Pipeline | LightGBM Reorder Prediction"
)