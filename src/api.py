from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = FastAPI(title="FMCG AI Smart Inventory System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_store = None
product_forecasts = []


@app.get("/")
def home():
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "dashboard.html"))
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        return {"error": f"File not found: {file_path}", "details": str(e)}


@app.get("/favicon.ico")
def favicon():
    return {"status": "no favicon"}



# -------------------------------
# LOAD DEFAULT DATA
# -------------------------------
@app.post("/load_data")
def load_data():
    global data_store

    path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "online_retail_II.csv")

    df = pd.read_csv(path, encoding='latin1')
    df = df[df['Quantity'] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    data_store = df

    return {"message": f"Dataset loaded with {len(df)} records"}


# -------------------------------
# BUILD LSTM MODEL
# -------------------------------
def build_model():
    model = Sequential()
    model.add(LSTM(32, input_shape=(60, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


# -------------------------------
# FORECAST (PER PRODUCT, 14 DAYS)
# -------------------------------
@app.get("/forecast")
def forecast():
    global data_store, product_forecasts

    if data_store is None:
        return {"error": "Load data first using /load_data"}

    df = data_store.copy()

    # pick top 5 products (fast + realistic)
    top_products = df['StockCode'].value_counts().head(5).index

    forecasts = []

    for product in top_products:
        product_df = df[df['StockCode'] == product].copy()

        product_df['date'] = product_df['InvoiceDate'].dt.date

        daily = product_df.groupby('date')['Quantity'].sum().reset_index()
        daily['date'] = pd.to_datetime(daily['date'])
        daily = daily.set_index('date').asfreq('D').fillna(0)

        if len(daily) < 70:
            continue

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(daily[['Quantity']]).astype(np.float32)

        # create sequences
        X = []
        for i in range(60, len(scaled)):
            X.append(scaled[i-60:i])
        X = np.array(X)

        y = scaled[60:]

        model = build_model()
        model.fit(X, y, epochs=5, verbose=0)

        # 🔥 14-DAY FORECAST
        future_days = 14
        preds = []
        last_seq = X[-1]

        for _ in range(future_days):
            pred = model.predict(last_seq.reshape(1, 60, 1), verbose=0)
            preds.append(pred[0][0])
            last_seq = np.append(last_seq[1:], pred, axis=0)

        preds = np.array(preds).reshape(-1, 1)
        preds = scaler.inverse_transform(preds)

        total_14d = float(preds.sum())

        forecasts.append({
            "StockCode": product,
            "PredictedDemand_14days": total_14d
        })

    product_forecasts = forecasts

    return {"forecast_14_days": forecasts}


# -------------------------------
# INVENTORY (14 DAY LOGIC)
# -------------------------------
@app.get("/inventory")
def inventory():
    global data_store, product_forecasts

    if data_store is None:
        return {"error": "Load data first"}

    if not product_forecasts:
        return {"error": "Run forecast first"}

    df = data_store.copy()
    df['date'] = df['InvoiceDate'].dt.date

    reorder_items = []

    for item in product_forecasts:
        product = item["StockCode"]
        forecast_14 = item["PredictedDemand_14days"]

        product_df = df[df['StockCode'] == product]

        # 🔥 CURRENT STOCK (LAST 30 DAYS)
        last_30_days = product_df.sort_values('InvoiceDate').tail(30)
        stock = last_30_days['Quantity'].sum()

        # avg daily demand
        daily = product_df.groupby('date')['Quantity'].sum().reset_index()
        avg_daily = daily['Quantity'].mean()

        if pd.isna(avg_daily) or avg_daily == 0:
            continue

        # 🔥 14 DAY LOGIC
        reorder_level = avg_daily * 14
        projected_stock = stock - forecast_14
        days_left = stock / avg_daily

        if projected_stock <= reorder_level:
            reorder_items.append({
                "StockCode": product,
                "CurrentStock_30days": float(stock),
                "AvgDailyDemand": float(avg_daily),
                "PredictedDemand_14days": float(forecast_14),
                "ReorderLevel_14days": float(reorder_level),
                "ProjectedStock": float(projected_stock),
                "DaysOfStockLeft": float(days_left)
            })

    return {"reorder_items": reorder_items}