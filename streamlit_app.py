import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from PIL import Image
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Titanic Dashboard", layout="wide")

# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.title("⚓ Titanic Explorer")
page = st.sidebar.radio(
    "Go to",
    ["Home", " Predict", " EDA", "Model Insights"]
)

# ---------- HELPER: API CALL ----------
API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")  # fallback to local

def call_api(payload):
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None

# ---------- PAGE 1: HOME ----------
if page == " Home":
    st.title(" Titanic Survival Prediction")
    st.markdown("#### End‑to‑End Machine Learning Capstone")

    col1, col2, col3 = st.columns(3)
    col1.metric(" Training rows", "891")
    col2.metric(" Raw features", "11")
    col3.metric("️ Engineered features", "20")

    st.markdown("---")
    st.subheader(" Champion Model Metrics")
    metrics_df = pd.DataFrame({
        "Model": ["XGBoost (Champion)"],
        "Accuracy": [0.832],
        "F1 Score": [0.819],
        "ROC‑AUC": [0.883],
    })
    st.dataframe(metrics_df, use_container_width=True)

    st.markdown("---")
    st.caption("Built with FastAPI + Streamlit + Docker")

# ---------- PAGE 2: PREDICT ----------
elif page == " Predict":
    st.title(" Predict Survival")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            pclass = st.selectbox("Pclass", [1, 2, 3], index=2)
            sex = st.radio("Sex", ["male", "female"], horizontal=True)
            age = st.slider("Age", 0.0, 100.0, 30.0, step=1.0)
            sibsp = st.number_input("SibSp (siblings/spouses)", 0, 8, 0)
            parch = st.number_input("Parch (parents/children)", 0, 6, 0)

        with col2:
            fare = st.number_input("Fare", 0.0, 600.0, 50.0, step=1.0)
            embarked = st.selectbox("Embarked", ["C", "Q", "S"], index=2)
            title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
            deck = st.selectbox("Deck", ["A", "B", "C", "D", "E", "F", "G", "Unknown"])

        # derived features
        family_size = 1 + sibsp + parch
        is_alone = 1 if family_size == 1 else 0
        fare_per_person = fare / family_size if family_size > 0 else fare

        submitted = st.form_submit_button(" Predict")

    if submitted:
        payload = {
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": embarked,
            "Title": title,
            "FamilySize": family_size,
            "IsAlone": is_alone,
            "Deck": deck,
            "FarePerPerson": fare_per_person,
        }

        result = call_api(payload)

        if result:
            survived = result.get("survived", 0)
            prob = result.get("probability", result.get("probability_survived", 0.0))

            st.markdown("---")
            st.subheader("Prediction Result")

            # Progress bar
            st.progress(prob)
            st.write(f"**Survival Probability:** {prob:.2%}")

            # Badge
            if survived == 1:
                st.success(" **Survived**")
            else:
                st.error("**Did Not Survive**")

# ---------- PAGE 3: EDA ----------
elif page == " EDA":
    st.title(" Exploratory Data Analysis")

    # Static overview image
    if os.path.exists("reports/eda_overview.png"):
        st.image("reports/eda_overview.png", caption="EDA Overview", use_container_width=True)
    else:
        st.warning("`reports/eda_overview.png` not found. Please generate it in earlier steps.")

    # Interactive Plotly chart (example: survival by Pclass & Sex)
    st.subheader("Interactive Survival by Pclass and Sex")

    # We'll create a sample dataset based on training data (or load from CSV if available)
    try:
        df = pd.read_csv("data/train.csv")
        fig = px.histogram(
            df, x="Pclass", color="Sex", facet_col="Survived",
            barmode="group", title="Survival by Pclass and Sex"
        )
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.error("Training data not found. Cannot display interactive chart.")

# ---------- PAGE 4: MODEL INSIGHTS ----------
else:  # " Model Insights"
    st.title("Model Insights – SHAP Explanations")

    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists("reports/shap_beeswarm.png"):
            st.image("reports/shap_beeswarm.png", caption="SHAP Beeswarm", use_container_width=True)
        else:
            st.info("`reports/shap_beeswarm.png` not yet generated.")

    with col2:
        if os.path.exists("reports/shap_bar.png"):
            st.image("reports/shap_bar.png", caption="SHAP Bar Plot", use_container_width=True)
        else:
            st.info("`reports/shap_bar.png` not yet generated.")

    st.markdown("---")
    st.subheader("SHAP Insight Summary")
    st.markdown("""
    - **Sex** is the strongest predictor – being female significantly increases survival odds.
    - **Pclass** and **Fare** are also highly influential (higher class / higher fare → more likely to survive).
    - **Age** shows a non‑linear effect: children and elderly tend to have higher survival rates.
    - The model relies heavily on engineered features like **Title** (social standing) and **FamilySize**.
    """)