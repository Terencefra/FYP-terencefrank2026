from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = FastAPI(title="FMCG AI Smart Inventory System")

# -------------------------------------------------
# CORS
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# GLOBAL DATA
# -------------------------------------------------
data_store = None
product_forecasts = []
forecast_ready = False


# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------
@app.get("/")
def home():
    file_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(file_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
@app.post("/load_data")
def load_data():
    global data_store, forecast_ready, product_forecasts

    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "raw",
        "online_retail_II.csv"
    )

    df = pd.read_csv(path, encoding="latin1")
    df = df[df["Quantity"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    data_store = df
    product_forecasts = []
    forecast_ready = False

    return {
        "message": f"Dataset loaded successfully ({len(df)} rows)"
    }


# -------------------------------------------------
# BUILD MODEL
# -------------------------------------------------
def build_model():
    model = Sequential()
    model.add(LSTM(32, input_shape=(60, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


# -------------------------------------------------
# FORECAST
# -------------------------------------------------
@app.get("/forecast")
def forecast():
    global data_store, product_forecasts, forecast_ready

    if data_store is None:
        return {"error": "Load data first"}

    # Return cached results for speed
    if forecast_ready and product_forecasts:
        return {"forecast_14_days": product_forecasts}

    df = data_store.copy()

    # Top 10 products
    top_products = df["StockCode"].value_counts().head(10).index[:10]

    forecasts = []

    for product in top_products:
        try:
            product_df = df[df["StockCode"] == product].copy()

            if len(product_df) < 20:
                continue

            description = product_df["Description"].iloc[0]

            product_df["date"] = product_df["InvoiceDate"].dt.date

            daily = (
                product_df.groupby("date")["Quantity"]
                .sum()
                .reset_index()
            )

            daily["date"] = pd.to_datetime(daily["date"])
            daily = daily.set_index("date").asfreq("D").fillna(0)

            if len(daily) < 70:
                continue

            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(
                daily[["Quantity"]]
            ).astype(np.float32)

            X = []
            for i in range(60, len(scaled)):
                X.append(scaled[i - 60:i])

            X = np.array(X)
            y = scaled[60:]

            model = build_model()
            model.fit(X, y, epochs=1, verbose=0)

            # 14-day recursive prediction
            future_days = 14
            preds = []
            last_seq = X[-1]

            for _ in range(future_days):
                pred = model.predict(
                    last_seq.reshape(1, 60, 1),
                    verbose=0
                )

                preds.append(pred[0][0])

                last_seq = np.append(
                    last_seq[1:],
                    pred,
                    axis=0
                )

            preds = np.array(preds).reshape(-1, 1)
            preds = scaler.inverse_transform(preds)

            total_14 = float(preds.sum())

            forecasts.append({
                "StockCode": product,
                "Description": description,
                "PredictedDemand_14days": round(total_14, 2)
            })

        except Exception as e:
            print("Forecast error:", product, e)
            continue

    product_forecasts = forecasts
    forecast_ready = True

    return {"forecast_14_days": forecasts}


# -------------------------------------------------
# INVENTORY
# -------------------------------------------------
@app.get("/inventory")
def inventory():
    global data_store, product_forecasts

    if data_store is None:
        return {"error": "Load data first"}

    if not product_forecasts:
        return {"error": "Run forecast first"}

    df = data_store.copy()
    df["date"] = df["InvoiceDate"].dt.date

    results = []

    for item in product_forecasts:
        try:
            code = item["StockCode"]
            desc = item["Description"]
            demand_14 = item["PredictedDemand_14days"]

            product_df = df[df["StockCode"] == code]

            if len(product_df) == 0:
                continue

            # Stock = recent 30 transactions
            last_30 = product_df.sort_values(
                "InvoiceDate"
            ).tail(30)

            stock = float(last_30["Quantity"].sum())

            # Average daily demand
            daily = (
                product_df.groupby("date")["Quantity"]
                .sum()
                .reset_index()
            )

            avg_daily = float(daily["Quantity"].mean())

            if avg_daily <= 0:
                continue

            reorder_level = avg_daily * 14
            projected = stock - demand_14
            days_left = projected / avg_daily

            # Better statuses
            if projected <= 0:
                status = "CRITICAL"
            elif projected <= reorder_level:
                status = "REORDER"
            else:
                status = "OK"

            results.append({
                "StockCode": code,
                "Description": desc,
                "CurrentStock": round(stock, 2),
                "AvgDailyDemand": round(avg_daily, 2),
                "PredictedDemand_14days": round(demand_14, 2),
                "ReorderLevel": round(reorder_level, 2),
                "ProjectedStock": round(projected, 2),
                "DaysOfStockLeft": round(days_left, 2),
                "Status": status
            })

        except Exception as e:
            print("Inventory error:", e)
            continue

    reorder_count = sum(
        1 for r in results if r["Status"] != "OK"
    )

    return {
        "inventory": results,
        "total_products": len(results),
        "reorder_count": reorder_count
    }