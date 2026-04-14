import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def load_data():
    df = pd.read_csv("data/processed/daily_demand.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df


def train_test_split(data):
    train_size = int(len(data) * 0.8)
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]
    return train, test


def train_arima(train):
    model = ARIMA(train['Quantity'], order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit


def forecast(model_fit, steps):
    return model_fit.forecast(steps=steps)


def evaluate(test, predictions):
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    return mae, rmse


def plot_results(test, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(test.index, test['Quantity'], label="Actual")
    plt.plot(test.index, predictions, label="ARIMA Forecast")
    plt.legend()
    plt.title("ARIMA Forecast vs Actual")
    plt.show()


def main():
    print("Loading data...")
    data = load_data()

    print("Splitting data...")
    train, test = train_test_split(data)

    print("Training ARIMA model...")
    model_fit = train_arima(train)

    print("Forecasting...")
    predictions = forecast(model_fit, len(test))

    print("Evaluating...")
    mae, rmse = evaluate(test['Quantity'], predictions)

    print(f"\nMAE: {mae}")
    print(f"RMSE: {rmse}")

    plot_results(test, predictions)


if __name__ == "__main__":
    main()