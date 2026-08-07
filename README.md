# 🛒 AI-Powered Grocery Reorder Prediction System

An end-to-end Machine Learning project that predicts which grocery products a customer is most likely to reorder using the **Instacart Market Basket Analysis** dataset. The project combines feature engineering, a LightGBM model, interactive visualizations, and a Streamlit web application to deliver personalized reorder recommendations.

---

## 📌 Overview

Online grocery platforms need to recommend products customers are likely to purchase again. Instead of recommending every previously purchased product, this system predicts the probability of reorder for each product and recommends only the most relevant items.

The project includes:

- ✅ End-to-end Machine Learning pipeline
- ✅ Advanced feature engineering
- ✅ LightGBM classification model
- ✅ Interactive Streamlit dashboard
- ✅ Model explainability
- ✅ Business insights
- ✅ Customer-level reorder prediction

---

## 🚀 Features

- Predict reorder probability for each product
- Personalized customer recommendations
- 28 engineered predictive features
- LightGBM model with Optuna hyperparameter tuning
- Interactive Streamlit web application
- Model performance analysis
- Business impact dashboard
- Data exploration dashboard
- Feature importance visualization
- Precision–Recall curve analysis
- Department-wise performance evaluation

---

## 📂 Project Structure

```text
AI-powered-Grocery-Reorder-Prediction-System/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── lookup/
│   │   └── products.csv
│   └── processed/
│       ├── deploy_sample.parquet
│       ├── segment_by_activity_final.csv
│       └── segment_by_department_final.csv
│
├── models/
│   ├── lgb_final_tuned.txt
│   └── final_model_meta.json
│
├── pages/
│   ├── 1_🏠_Home.py
│   ├── 2_🛒_Predict.py
│   ├── 3_📊_Model_Insights.py
│   └── 4_📈_Data_Insights.py
│
├── plots/
│   ├── activity_segment.png
│   ├── feature_importance.png
│   ├── model_progression.png
│   ├── pr_curve.png
│   └── recall_by_department.png
│
└── utils/
    ├── helper.py
    ├── loader.py
    └── predictor.py
```

---

## 📊 Dataset

**Dataset:** Instacart Market Basket Analysis

The original dataset contains over **3 million grocery orders** from more than **200,000 customers**.

For deployment, a representative sample is used to reduce application size while preserving application functionality.

---

## ⚙️ Feature Engineering

The model uses **28 engineered features** across multiple categories.

### 👤 User Features

- Total orders
- Average basket size
- Reorder ratio
- Days between orders
- Reorder streak
- Distinct products purchased

### 🛒 Product Features

- Product popularity
- Product reorder rate
- Average cart position
- Distinct users
- Average reorder interval

### 🔄 User–Product Interaction

- Times bought
- Purchase frequency
- Orders since last purchase
- Recent purchase rate
- Purchase consistency

### 🏷️ Category Features

- Department
- Aisle
- Department reorder rate

---

## 🤖 Machine Learning Model

**Algorithm:** LightGBM Classifier

### Training Pipeline

- Feature Engineering
- Hyperparameter Optimization using Optuna
- Threshold Optimization
- Probability Prediction
- Per-user Evaluation

---

## 📈 Model Performance

| Metric | Value |
|---------|------:|
| Best Model | LightGBM |
| Per-User F1 Score | **0.3800** |
| Baseline F1 Score | **0.2660** |
| Performance Improvement | **42.84%** |

---

## 📊 Streamlit Dashboard

The application contains four interactive pages.

### 🏠 Home

- Project Overview
- Dataset Summary
- Workflow
- Model Performance
- Business Objective

### 🛒 Predict

- Select Customer ID
- Predict Reorder Probability
- View Personalized Recommendations
- Download Prediction Results

### 📊 Model Insights

- Feature Importance
- Precision–Recall Curve
- Model Progression
- Activity Segment Analysis
- Department-wise Recall
- Business Impact

### 📈 Data Insights

- User Behaviour Analysis
- Product Popularity
- Department Distribution
- Feature Engineering Overview
- Interactive Visualizations

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- Optuna
- Plotly
- Matplotlib
- Streamlit

---

## 💻 Installation

Clone the repository

```bash
git clone https://github.com/soni132/instacart-reorder-prediction.git
```

Navigate to the project directory

```bash
cd instacart-reorder-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📌 Future Improvements

- Train using the complete Instacart dataset
- Deploy prediction API using FastAPI
- Docker containerization
- Cloud database integration
- Real-time recommendation system
- User authentication
- Recommendation ranking optimization

---

## 👩‍💻 Author

**Soni Kumari**

B.Tech (Information Technology)  
Indira Gandhi Delhi Technical University for Women (IGDTUW)

GitHub: https://github.com/soni132

---

## 📄 License

This project is developed for educational, research, and portfolio purposes.
