import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# -------------------------------
# LOAD PROCESSED DATA (FOR LSTM)
# -------------------------------
def load_data():
    df = pd.read_csv("data/processed/daily_demand.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df


# -------------------------------
# LOAD RAW DATA (FOR PRODUCTS)
# -------------------------------
def load_raw_data():
    df = pd.read_csv("data/raw/online_retail_II.csv", encoding='latin1')
    df = df[df['Quantity'] > 0]  # Filter positive quantities only
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df


# -------------------------------
# PREPARE PRODUCT STOCK
# -------------------------------
def prepare_product_data(df):
    df['date'] = df['InvoiceDate'].dt.date

    # count number of days each product appears
    product_days = df.groupby(['StockCode', 'Description'])['date'].nunique().reset_index()
    product_days.columns = ['StockCode', 'Description', 'ActiveDays']

    # total sold
    stock = df.groupby(['StockCode', 'Description'])['Quantity'].sum().reset_index()
    stock.columns = ['StockCode', 'Description', 'TotalSold']

    # avg daily demand
    daily = df.groupby(['StockCode', 'Description', 'date'])['Quantity'].sum().reset_index()
    avg_daily = daily.groupby(['StockCode', 'Description'])['Quantity'].mean().reset_index()
    avg_daily.columns = ['StockCode', 'Description', 'AvgDailyDemand']

    # merge all
    product_data = pd.merge(stock, avg_daily, on=['StockCode', 'Description'])
    product_data = pd.merge(product_data, product_days, on=['StockCode', 'Description'])

    # 🔥 FILTER OUT BAD PRODUCTS
    product_data = product_data[product_data['ActiveDays'] > 10]

    return product_data


# -------------------------------
# SCALE DATA
# -------------------------------
def scale_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Quantity']])
    return scaler, scaled_data


# -------------------------------
# CREATE SEQUENCES
# -------------------------------
def create_sequences(data, window_size=60):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y)


# -------------------------------
# BUILD LSTM MODEL
# -------------------------------
def build_model(window_size):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(window_size, 1)))
    model.add(LSTM(32))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    return model


# -------------------------------
# CHECK STOCK ALERT (RETAINED FOR COMPATIBILITY, BUT EXTENDED BELOW)
# -------------------------------
def check_stock(predicted_demand, current_stock, reorder_threshold):
    """
    Checks stock levels based on predicted demand.
    Returns alert if stock after demand falls below or at reorder threshold.
    """
    projected_stock = current_stock - predicted_demand
    if projected_stock <= reorder_threshold:
        return "Low Stock - Reorder Needed"
    else:
        return "Stock OK"


# -------------------------------
# MAIN FUNCTION
# -------------------------------
def main():
    print("Loading processed data for LSTM...")
    data = load_data()

    print("Scaling data...")
    scaler, scaled_data = scale_data(data)

    print("Creating sequences...")
    X, y = create_sequences(scaled_data)

    # Train-test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("Building LSTM model...")
    window_size = X.shape[1]
    model = build_model(window_size)

    print("Training LSTM model...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)  # Added verbose=0 to reduce output

    print("Predicting with LSTM...")
    predictions = model.predict(X_test)

    # Inverse scale
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test)

    print("Evaluating LSTM...")
    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))

    print(f"\nLSTM MAE: {mae:.2f}")
    print(f"LSTM RMSE: {rmse:.2f}")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_actual.flatten(), label="Actual")
    plt.plot(predictions.flatten(), label="LSTM Forecast")
    plt.legend()
    plt.title("LSTM Demand Forecast vs Actual")
    plt.show()

    # Get latest predicted demand (aggregate)
    latest_predicted_demand = predictions[-1][0]

    # -------------------------------
    # INVENTORY MANAGEMENT PART
    # -------------------------------
    print("\nLoading raw data for inventory...")
    raw_df = load_raw_data()
    product_data = prepare_product_data(raw_df)

    # Take top 5 products by total sold (after filtering)
    product_data = product_data.sort_values(by='TotalSold', ascending=False).head(5)

    lead_time_days = 7  # Realistic lead time

    print("\n" + "="*80)
    print("REAL FMCG INVENTORY SYSTEM WITH LSTM DEMAND SIGNAL")
    print("="*80)

    reorder_list = []

    for _, row in product_data.iterrows():
        stock_code = row['StockCode']
        name = row['Description']
        stock = row['TotalSold']
        avg_demand = row['AvgDailyDemand']

        # Reorder level based on avg demand and lead time
        reorder_level = avg_demand * lead_time_days

        # Use LSTM predicted demand as a global signal (applied to all products for simplicity)
        projected_stock = stock - latest_predicted_demand

        if projected_stock <= reorder_level:
            status = "⚠️ Reorder Needed"
            reorder_list.append(name)
        else:
            status = "✅ Stock OK"

        print(f"\nProduct: {stock_code}")
        print(f"Name: {name}")
        print(f"Current Stock (Estimated): {stock}")
        print(f"Avg Daily Demand: {avg_demand:.2f}")
        print(f"Reorder Level: {reorder_level:.2f}")
        print(f"LSTM Predicted Demand (Global): {latest_predicted_demand:.2f}")
        print(f"Projected Stock: {projected_stock:.2f}")
        print(f"Status: {status}")

    print("\n" + "-"*80)
    print("SUMMARY:")
    if reorder_list:
        print("Products needing reorder:")
        for p in reorder_list:
            print(f"- {p}")
    else:
        print("All products have sufficient stock.")
    print("="*80)

    # Optional: Retain original stock alert for aggregate (if needed)
    # current_stock = 15000
    # reorder_threshold = 5000
    # alert = check_stock(latest_predicted_demand, current_stock, reorder_threshold)
    # print(f"\nAggregate Stock Alert: {alert}")


if __name__ == "__main__":
    main()