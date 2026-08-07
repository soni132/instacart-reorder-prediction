import streamlit as st
import pandas as pd


def probability_color(prob):

    if prob >= 0.80:
        return "🟢 Very High"

    elif prob >= 0.60:
        return "🟡 High"

    elif prob >= 0.40:
        return "🟠 Medium"

    else:
        return "🔴 Low"


def prediction_label(pred):

    if pred == 1:
        return "✅ Reorder"

    return "❌ Not Likely"


def format_prediction_table(df):

    table = df.copy()

    table["Probability"] = (
        table["Probability"] * 100
    ).round(2)

    table["Confidence"] = table["Probability"].apply(probability_color)

    table["Prediction"] = table["Prediction"].apply(prediction_label)

    return table


def show_metrics(df):

    total_products = len(df)

    predicted = int(df["Prediction"].eq("✅ Reorder").sum())

    avg_prob = round(df["Probability"].mean(), 1)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Products Analysed",
        total_products
    )

    c2.metric(
        "Predicted Reorders",
        predicted
    )

    c3.metric(
        "Average Probability",
        f"{avg_prob}%"
    )