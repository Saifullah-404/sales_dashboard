# streamlit_amazon_with_sklearn.py
import os
from io import BytesIO
from datetime import timedelta

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn imports
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Amazon Sales — Streamlit + scikit-learn", layout="wide")
st.title("Amazon Sales — EDA, Visualization & Forecast (scikit-learn included)")

# Styling (custom non-default palette)
PALETTE = sns.color_palette("Dark2")
sns.set_palette(PALETTE)
plt.rcParams["figure.figsize"] = (10, 4)

# -------------------------
# Helpers
# -------------------------
def detect_columns(df):
    """Robust detection: prefer explicit 'date' column, fall back to parseable."""
    cols = df.columns.tolist()
    date_col = None
    for c in cols:
        if "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        for c in cols:
            if "order date" in c.lower() or ("order" in c.lower() and "id" not in c.lower()):
                date_col = c; break
    if date_col is None:
        for c in cols:
            try:
                pd.to_datetime(df[c].dropna().iloc[:5])
                date_col = c; break
            except Exception:
                pass

    sales_col = None
    for c in cols:
        if c.lower() in ["sales","revenue","total","order_value","amount"]:
            sales_col = c; break
    if sales_col is None:
        for c in cols:
            if "price" in c.lower() or "amount" in c.lower():
                sales_col = c; break

    units_col = None
    for c in cols:
        if any(k in c.lower() for k in ["unit","qty","quantity","count"]):
            units_col = c; break

    product_col = None
    for c in cols:
        if any(k in c.lower() for k in ["product","title","asin","sku"]):
            product_col = c; break

    category_col = None
    for c in cols:
        if "category" in c.lower():
            category_col = c; break

    return date_col, sales_col, units_col, product_col, category_col

def build_lag_features(daily_df):
    ts = daily_df.copy().set_index("date").asfreq("D").fillna(0).reset_index()
    ts["lag_1"] = ts["sales"].shift(1).fillna(0)
    ts["lag_7"] = ts["sales"].shift(7).fillna(0)
    ts["rolling_7"] = ts["sales"].rolling(7, min_periods=1).mean().shift(1).fillna(0)
    ts["dow"] = ts["date"].dt.dayofweek
    ts["month"] = ts["date"].dt.month
    return ts

def recursive_forecast(model, ts, features, days):
    recent = ts.copy().reset_index(drop=True)
    future_rows = []
    for i in range(days):
        last = recent.iloc[-1]
        next_date = last["date"] + pd.Timedelta(days=1)
        lag_1 = last["sales"]
        lag_7 = recent["sales"].iloc[-7] if len(recent) >= 7 else last["sales"]
        rolling_7 = recent["sales"].rolling(7, min_periods=1).mean().iloc[-1]
        dow = next_date.dayofweek
        month = next_date.month
        Xrow = np.array([[lag_1, lag_7, rolling_7, dow, month]])
        if hasattr(model, "predict"):
            pred = model.predict(Xrow)[0]
        else:
            # linear models from sklearn have predict; this is defensive
            pred = float(Xrow.dot(model.coef_.reshape(-1,1)) + model.intercept_)
        future_rows.append({"date": next_date, "sales": float(pred)})
        recent = pd.concat([recent, pd.DataFrame([{"date": next_date, "sales": pred}])], ignore_index=True)
    return pd.DataFrame(future_rows)

def to_csv_bytes(df):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

# -------------------------
# File upload / load
# -------------------------
uploaded = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded is None:
    default_path = "amazon_sales_data 2025.csv"
    if os.path.exists(default_path):
        st.info(f"No upload — using local file `{default_path}`")
        df = pd.read_csv(default_path, low_memory=False)
    else:
        st.warning("Upload a CSV or place `amazon_sales_data 2025.csv` in the working folder.")
        st.stop()
else:
    df = pd.read_csv(uploaded, low_memory=False)

st.write("Dataset shape:", df.shape)
st.dataframe(df.head(8))

# -------------------------
# Detect columns + parse
# -------------------------
date_col, sales_col, units_col, product_col, category_col = detect_columns(df)
st.markdown("**Detected columns:**")
st.write(f"- date: `{date_col}`")
st.write(f"- sales/price: `{sales_col}`")
st.write(f"- units: `{units_col}`")
st.write(f"- product/title: `{product_col}`")
st.write(f"- category: `{category_col}`")

if date_col is None:
    st.error("Could not detect a date column. Ensure there is a date-like column.")
    st.stop()

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).copy()

# compute numeric sales
if sales_col and sales_col in df.columns:
    df["_sales_numeric_"] = pd.to_numeric(df[sales_col], errors="coerce").fillna(0)
else:
    # fallback: price * qty
    price_col = None
    for c in df.columns:
        if "price" in c.lower():
            price_col = c; break
    if price_col and units_col and units_col in df.columns:
        df["_sales_numeric_"] = pd.to_numeric(df[price_col], errors="coerce") * pd.to_numeric(df[units_col], errors="coerce")
        df["_sales_numeric_"] = df["_sales_numeric_"].fillna(0)
    else:
        st.warning("No sales or price*quantity found — sales will be set to 0.")
        df["_sales_numeric_"] = 0

df["_units_numeric_"] = pd.to_numeric(df[units_col], errors="coerce").fillna(0) if units_col and units_col in df.columns else 1

# Aggregate to daily
daily = df.groupby(df[date_col].dt.date).agg(
    sales=("_sales_numeric_","sum"),
    units=("_units_numeric_","sum")
).reset_index().rename(columns={date_col:"date"})
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values("date").reset_index(drop=True)

if daily.empty:
    st.error("Daily series is empty after aggregation. Check input dates/sales.")
    st.stop()

# -------------------------
# Controls
# -------------------------
st.sidebar.header("Controls")
min_date = daily["date"].min().date()
max_date = daily["date"].max().date()
date_range = st.sidebar.date_input("Plot date range", [min_date, max_date], min_value=min_date, max_value=max_date)
model_choice = st.sidebar.selectbox("Model", ["LinearRegression", "Ridge", "RandomForest"])
n_forecast = st.sidebar.number_input("Forecast horizon (days)", min_value=7, max_value=365, value=30, step=1)
train_frac = st.sidebar.slider("Train fraction (time series split)", min_value=0.5, max_value=0.95, value=0.8)

# -------------------------
# KPIs and time-series plot
# -------------------------
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("KPIs")
    st.metric("Date range", f"{min_date} → {max_date}")
    st.metric("Total revenue", f"{daily['sales'].sum():,.2f}")
    st.metric("Avg daily sales", f"{daily['sales'].mean():,.2f}")

with col2:
    st.subheader("Daily sales (selected range)")
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    mask = (daily["date"] >= start) & (daily["date"] <= end)
    plot_df = daily.loc[mask]

    fig, ax = plt.subplots()
    ax.plot(plot_df["date"], plot_df["sales"], linewidth=2, marker="o")
    ax.set_xlabel("")
    ax.set_title("Daily Sales")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# -------------------------
# Top products / categories
# -------------------------
st.subheader("Top products / categories (selected range)")
left, right = st.columns(2)
with left:
    if product_col and product_col in df.columns:
        prod_agg = df.loc[mask].groupby(product_col).agg(sales=("_sales_numeric_","sum")).reset_index().sort_values("sales", ascending=False).head(15)
        figp, axp = plt.subplots()
        sns.barplot(data=prod_agg, x="sales", y=product_col, ax=axp)
        axp.set_title("Top Products")
        st.pyplot(figp)
    else:
        st.info("No product/title column detected — skip product chart.")

with right:
    if category_col and category_col in df.columns:
        cat_agg = df.loc[mask].groupby(category_col).agg(sales=("_sales_numeric_","sum")).reset_index().sort_values("sales", ascending=False).head(15)
        figc, axc = plt.subplots()
        sns.barplot(data=cat_agg, x="sales", y=category_col, ax=axc)
        axc.set_title("Top Categories")
        st.pyplot(figc)
    else:
        st.info("No category column detected — skip category chart.")

# -------------------------
# Prepare features + train
# -------------------------
st.subheader("Model training & evaluation")

ts = build_lag_features(daily)
# choose features: lag_1, lag_7, rolling_7, dow, month
feat_cols = ["lag_1", "lag_7", "rolling_7", "dow", "month"]
X = ts[feat_cols].to_numpy()
y = ts["sales"].to_numpy()

n_train = int(len(ts) * train_frac)
X_train, y_train = X[:n_train], y[:n_train]
X_test, y_test = X[n_train:], y[n_train:]
dates_test = ts["date"].iloc[n_train:].to_list()

# instantiate model
if model_choice == "LinearRegression":
    model = LinearRegression()
elif model_choice == "Ridge":
    model = Ridge(alpha=1.0)
else:
    model = RandomForestRegressor(n_estimators=200, random_state=42)

model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_test)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)

st.write(f"Trained `{model_choice}` — Test MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# plot actual vs predicted
fig_cmp, ax_cmp = plt.subplots()
ax_cmp.plot(ts["date"].iloc[n_train:], y_test, marker="o", label="Actual")
ax_cmp.plot(ts["date"].iloc[n_train:], y_pred_test, marker="o", label="Predicted")
ax_cmp.set_title("Actual vs Predicted (test window)")
ax_cmp.legend()
plt.xticks(rotation=45)
st.pyplot(fig_cmp)

# -------------------------
# Forecast next N days
# -------------------------
st.subheader("Forecast")
future_df = recursive_forecast(model, ts, feat_cols, days=int(n_forecast))
st.write(f"Next {n_forecast} days (sample):")
st.dataframe(future_df.head(10))

# plot history + forecast
fig_f, ax_f = plt.subplots()
ax_f.plot(ts["date"], ts["sales"], label="Historical")
ax_f.plot(future_df["date"], future_df["sales"], label="Forecast", linestyle="--", marker="o")
ax_f.set_title("Historical + Forecast")
ax_f.legend()
plt.xticks(rotation=45)
st.pyplot(fig_f)

# download
csv_buf = to_csv_bytes(future_df)
st.download_button("Download forecast CSV", data=csv_buf, file_name="amazon_forecast.csv", mime="text/csv")

# -------------------------
# Per-product forecast (optional)
# -------------------------
if product_col and product_col in df.columns:
    st.subheader("Per-product quick forecast (optional)")
    prod_list = sorted(df[product_col].dropna().unique().tolist())
    sel = st.selectbox("Choose product", ["(none)"] + prod_list)
    if sel and sel != "(none)":
        dfp = df[df[product_col] == sel].copy()
        dp = dfp.groupby(dfp[date_col].dt.date).agg(sales=("_sales_numeric_","sum")).reset_index().rename(columns={date_col:"date"})
        dp["date"] = pd.to_datetime(dp["date"])
        dp = dp.sort_values("date").reset_index(drop=True)
        if len(dp) < 14:
            st.warning("Not enough history for reliable product-level forecast (need >=14 days).")
        else:
            ts_p = build_lag_features(dp)
            Xp = ts_p[["lag_1","lag_7","rolling_7","dow","month"]].to_numpy()
            yp = ts_p["sales"].to_numpy()
            ntr = int(len(ts_p)*0.8)
            m_linear = LinearRegression()
            m_linear.fit(Xp[:ntr], yp[:ntr])
            future_p = recursive_forecast(m_linear, ts_p, ["lag_1","lag_7","rolling_7","dow","month"], int(n_forecast))
            st.dataframe(future_p.head(8))
            figp, axp = plt.subplots()
            axp.plot(ts_p["date"], ts_p["sales"], label="hist")
            axp.plot(future_p["date"], future_p["sales"], label="forecast", linestyle="--", marker="o")
            axp.set_title(f"{sel} — history + forecast")
            axp.legend()
            st.pyplot(figp)

st.markdown("**Notes:**\n- Model choices: LinearRegression and Ridge (fast, linear) or RandomForest (nonlinear). For production or better long-term forecasts consider dedicated time-series libraries (Prophet, ARIMA) or more engineered features.")
