import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Store Sales Dashboard",
    layout="wide",
)

# -----------------------------
# FIXED DATA PATH FROM GITHUB
# -----------------------------
DATA_URL = "https://raw.githubusercontent.com/yourusername/yourrepo/main/data/amazon_sales.csv"

# Load dataset
df = pd.read_csv(DATA_URL)

# -----------------------------
# CLEANING – REMOVE STATUS COLUMN IF EXISTS
# -----------------------------
if "Status" in df.columns:
    df = df.drop("Status", axis=1)

# Convert date column (auto-detect)
date_col = None
for c in df.columns:
    if "date" in c.lower():
        date_col = c
        break

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

# -----------------------------
# UI STYLE
# -----------------------------
st.markdown("""
<style>
    .main { background-color: #F8FAFF; }
    .metric-card { 
        background: white; padding: 20px; border-radius: 12px; 
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filters")

if date_col:
    start = df[date_col].min()
    end = df[date_col].max()

    date_range = st.sidebar.date_input("Select Date Range", [start, end])

    # filter data
    df_filtered = df[(df[date_col] >= pd.to_datetime(date_range[0])) &
                     (df[date_col] <= pd.to_datetime(date_range[1]))]
else:
    df_filtered = df.copy()

# -----------------------------
# TOP METRICS
# -----------------------------
st.title("Store Sales Dashboard")

col1, col2, col3 = st.columns(3)

sales_col = None
for c in df.columns:
    if c.lower() in ["amount", "sales", "total"] or "price" in c.lower():
        sales_col = c

units_col = None
for c in df.columns:
    if "qty" in c.lower() or "unit" in c.lower():
        units_col = c

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Total Sales")
    st.metric(label="", value=f"{df_filtered[sales_col].sum():,.0f}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Total Units Sold")
    st.metric(label="", value=f"{df_filtered[units_col].sum():,.0f}")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Average Order Value")
    st.metric(label="", value=f"{df_filtered[sales_col].mean():,.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# VISUALIZATIONS
# -----------------------------
st.header("Data Visualization")

# Sales over time
if date_col:
    st.subheader("Sales Over Time")
    daily = df_filtered.groupby(date_col)[sales_col].sum()

    fig, ax = plt.subplots(figsize=(10,4))
    sns.lineplot(x=daily.index, y=daily.values, ax=ax, linewidth=2.5, color="#0A66C2")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    st.pyplot(fig)

# Category sales
cat_col = None
for c in df.columns:
    if "category" in c.lower():
        cat_col = c

if cat_col:
    st.subheader("Sales by Category")

    fig, ax = plt.subplots(figsize=(10,4))
    df_filtered.groupby(cat_col)[sales_col].sum().sort_values().plot(
        kind="bar", ax=ax, color="#FFA726"
    )
    ax.set_ylabel("Total Sales")
    st.pyplot(fig)

# -----------------------------
# MACHINE LEARNING PREDICTION
# -----------------------------
st.header("Sales Prediction")

# Prepare ML data
df_ml = df_filtered[[sales_col, units_col]].copy()

df_ml = df_ml.dropna()

X = df_ml[[units_col]]  # Feature
y = df_ml[sales_col]     # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

pred = model.predict(X_test)

# Model metrics
colA, colB = st.columns(2)
with colA:
    st.metric("MAE", f"{mean_absolute_error(y_test, pred):.2f}")
with colB:
    st.metric("R² Score", f"{r2_score(y_test, pred):.2f}")

st.subheader("Predict Sales")

units_input = st.number_input("Enter Units Sold", min_value=0, value=10)

if st.button("Predict"):
    prediction = model.predict([[units_input]])[0]
    st.success(f"Predicted Sales: {prediction:,.2f}")

# END
