from pathlib import Path
import math
from datetime import date, time
import joblib
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="EV Charging Demand",
    page_icon="⚡",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "Data" / "Cleaned" / "ChargingRecords_Feature_Engineered.csv"
REG_MODEL_PATH = ROOT / "Models" / "best_xgboost_regression.pkl"
CLS_MODEL_PATH = ROOT / "Models" / "best_classification_model.pkl"
REG_RESULT_PATH = ROOT / "Reports" / "regression_results.csv"
CLASSIFICATION_RESULT_PATH = ROOT / "Reports" / "classification_results.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    return joblib.load(REG_MODEL_PATH), joblib.load(CLS_MODEL_PATH)


st.title("⚡ EV Charging Demand Prediction")
st.caption("Demand prediction and charging usage analysis")

try:
    df = load_data()
    reg_model, cls_model = load_models()
except (FileNotFoundError, ImportError) as error:
    st.error(f"Unable to load project assets: {error}")
    st.stop()

feature_columns = [
    column for column in df.columns if column != "Demand"
]

if len(feature_columns) != 23:
    st.error(
        f"Expected 23 model features, found {len(feature_columns)}."
    )
    st.stop()

st.success("Dataset and both models loaded successfully.")

col1, col2, col3 = st.columns(3)

col1.metric("Charging Sessions", f"{len(df):,}")
col2.metric("Model Features", len(feature_columns))
col3.metric("Demand Threshold", f"{df['Demand'].median():.2f}")

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.divider()
st.subheader("Predict Charging Demand")

feature_columns = reg_model.get_booster().feature_names
location_columns = [
    col for col in feature_columns if col.startswith("Location_")
]

with st.form("demand_prediction"):
    session_date = st.date_input("Session date", value=date.today())
    session_time = st.time_input("Session start time", value=time(18, 0))

    charger_type = st.selectbox("Charger type", [0, 1])
    charger_company = st.selectbox("Charger company", [0, 1])

    location = st.selectbox(
        "Location",
        [col.removeprefix("Location_") for col in location_columns],
    )

    submitted = st.form_submit_button("Predict Demand")

if submitted:
    hour = session_time.hour
    weekday = session_date.weekday()
    month = session_date.month

    features = {
        "ChargerType": charger_type,
        "ChargerCompany": charger_company,
        "DayOfMonth": session_date.day,
        "HourSin": math.sin(2 * math.pi * hour / 24),
        "HourCos": math.cos(2 * math.pi * hour / 24),
        "DaySin": math.sin(2 * math.pi * weekday / 7),
        "DayCos": math.cos(2 * math.pi * weekday / 7),
        "MonthSin": math.sin(2 * math.pi * month / 12),
        "MonthCos": math.cos(2 * math.pi * month / 12),
    }

    features.update({
        col: int(col == f"Location_{location}")
        for col in location_columns
    })

    input_data = pd.DataFrame([features], columns=feature_columns)

    predicted_demand = float(reg_model.predict(input_data)[0])
    predicted_class = int(cls_model.predict(input_data)[0])
    high_demand_probability = float(
        cls_model.predict_proba(input_data)[0, 1]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Predicted Demand", f"{predicted_demand:.2f}")
    col2.metric(
        "Demand Category",
        "High Demand" if predicted_class == 1 else "Normal Demand",
    )
    col3.metric(
        "High Demand Probability",
        f"{high_demand_probability:.1%}",
    )
    
st.divider()
st.subheader("Charging Usage Patterns")

# Recover session start hour from cyclical features
df["StartHour"] = (
    np.rint(
        np.arctan2(df["HourSin"], df["HourCos"]) * 24 / (2 * np.pi)
    ).astype(int) % 24
)

hourly_usage = (
    df.groupby("StartHour")
    .size()
    .reset_index(name="Sessions")
)

fig_hourly = px.bar(
    hourly_usage,
    x="StartHour",
    y="Sessions",
    title="Charging Sessions by Start Hour",
    labels={"StartHour": "Hour of Day"},
)

st.plotly_chart(fig_hourly, use_container_width=True)

location_columns = [
    col for col in df.columns if col.startswith("Location_")
]

location_usage = pd.DataFrame({
    "Location": [
        col.removeprefix("Location_") for col in location_columns
    ],
    "AverageDemand": [
        df.loc[df[col] == 1, "Demand"].mean()
        for col in location_columns
    ],
}).sort_values("AverageDemand", ascending=False)

fig_location = px.bar(
    location_usage,
    x="Location",
    y="AverageDemand",
    title="Average Demand by Location",
)

st.plotly_chart(fig_location, use_container_width=True)   

st.subheader("Monthly Charging Trends")

df["Month"] = (
    np.rint(
        np.arctan2(df["MonthSin"], df["MonthCos"]) * 12 / (2 * np.pi)
    ).astype(int) % 12
).replace(0, 12)

monthly_usage = (
    df.groupby("Month", as_index=False)
    .agg(
        Sessions=("Demand", "size"),
        AverageDemand=("Demand", "mean"),
    )
)

fig_monthly = px.line(
    monthly_usage,
    x="Month",
    y="AverageDemand",
    markers=True,
    title="Average Demand by Month",
    labels={
        "Month": "Month",
        "AverageDemand": "Average Demand",
    },
)

fig_monthly.update_xaxes(dtick=1)
st.plotly_chart(fig_monthly, use_container_width=True) 

st.divider()
st.subheader("Model Performance")


reg_tab, cls_tab = st.tabs(["Regression", "Classification"])

with reg_tab:
    st.markdown("#### Regression Model Comparison")

    if REG_RESULT_PATH.exists():
        regression_results = pd.read_csv(REG_RESULT_PATH)
        st.dataframe(regression_results, use_container_width=True)
    else:
        st.warning("Regression results file not found.")

with cls_tab:
    st.markdown("#### Classification Model Comparison")

    if CLASSIFICATION_RESULT_PATH.exists():
        classification_results = pd.read_csv(CLASSIFICATION_RESULT_PATH)
        st.dataframe(classification_results, use_container_width=True)
    else:
        st.warning("Classification results file not found.")