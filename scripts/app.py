# app.py

import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Food Waste Prediction", layout="wide")

# --- Load Model & Data ---
@st.cache_resource
def load_model():
    return joblib.load("best_food_waste_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_food_waste_data.csv")

model = load_model()
data = load_data()

# --- Sidebar ---
st.sidebar.header("ℹ️ App Info")
st.sidebar.markdown("Predict food waste in a mess kitchen and explore insights using a trained ML model.")

# --- Title ---
st.title("🥣 Mess Food Waste Prediction Dashboard")

# --- Data Overview ---
with st.expander("🔍 View Sample Data"):
    st.dataframe(data.head())

# --- Prediction ---
X = data.drop(columns=["food_waste_kg"])
y = data["food_waste_kg"]
y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
r2 = r2_score(y, y_pred)

st.subheader("📈 Model Performance")
st.metric("R² Score", f"{r2:.3f}")
st.metric("MAE", f"{mae:.2f}")
st.metric("RMSE", f"{rmse:.2f}")

# --- Actual vs Predicted Plot ---
st.subheader("📊 Actual vs Predicted Food Waste")
fig1, ax1 = plt.subplots()
sns.scatterplot(x=y, y=y_pred, alpha=0.6, ax=ax1)
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax1.set_xlabel("Actual (kg)")
ax1.set_ylabel("Predicted (kg)")
ax1.set_title("Actual vs Predicted")
st.pyplot(fig1)

# --- Residual Plot ---
st.subheader("📉 Residuals Distribution")
residuals = y - y_pred
fig2, ax2 = plt.subplots()
sns.histplot(residuals, bins=30, kde=True, ax=ax2, color="orange")
ax2.set_title("Residuals")
st.pyplot(fig2)

# --- Feature Importance ---
if hasattr(model.named_steps["regressor"], "feature_importances_"):
    st.subheader("🧠 Feature Importances")

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["regressor"].feature_importances_

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=fi_df.head(10), palette="crest", ax=ax3)
    ax3.set_title("Top 10 Features")
    st.pyplot(fig3)

# --- Footer ---
st.markdown("---")
st.markdown("🔗 Developed by *[Your Name]* | Powered by Streamlit & Scikit-Learn")
