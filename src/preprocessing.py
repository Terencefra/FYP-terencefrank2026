import pandas as pd


def load_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path, encoding='latin1')
    return df


def clean_data(df):
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Remove returns (negative or zero quantities)
    df = df[df['Quantity'] > 0]

    return df


def aggregate_daily(df):
    # Extract only date (remove time)
    df['date'] = df['InvoiceDate'].dt.date

    # Group by date and sum total demand
    daily = df.groupby('date')['Quantity'].sum().reset_index()

    # Convert date column back to datetime
    daily['date'] = pd.to_datetime(daily['date'])

    # Sort by date
    daily = daily.sort_values('date')

    return daily


def fill_missing_dates(daily):
    # Set date as index and fill missing days
    daily = daily.set_index('date').asfreq('D').fillna(0)
    return daily


def save_data(daily):
    # Save processed dataset
    daily.to_csv("data/processed/daily_demand.csv")


def main():
    file_path = "data/raw/online_retail_II.csv"

    print("Step 1: Loading data...")
    df = load_data(file_path)

    print("Step 2: Cleaning data...")
    df = clean_data(df)

    print("Step 3: Aggregating daily demand...")
    daily = aggregate_daily(df)

    print("Step 4: Filling missing dates...")
    daily = fill_missing_dates(daily)

    print("Step 5: Saving processed data...")
    save_data(daily)

    print("\nPreview of processed data:")
    print(daily.head())

    print(f"\nTotal rows after processing: {len(daily)}")


if __name__ == "__main__":
    main()