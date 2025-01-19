import pandas as pd
import datetime as dt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import boto3
import io
from botocore.exceptions import ClientError
from tensorflow.keras.models import save_model, load_model
import tempfile
import h5py

from utils import WasabiClient, get_wassabi_client

wasabi = get_wassabi_client()


def check_missing_data(df):
    missing = df['Last trade price'].isna().sum() / len(df)

    all_zeros = (df['Last trade price'] == 0).all()

    return missing > 0.7 or all_zeros


def prepare_stock_data_analysis(df):
    try:
        numeric_columns = [
            'Last trade price', 'Max', 'Min', 'Avg. Price',
            '%chg.', 'Volume', 'Turnover in BEST in denars', 'Total turnover in denars'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('.', '').str.replace(',', '.').astype(float)

        return df
    except Exception as e:
        return None


def handle_missing_values(df):
    df['Last trade price'] = df['Last trade price'].bfill()
    return df


def train_model(stock_code):
    print(f"Processing stock: {stock_code}")

    df = wasabi.fetch_data(stock_code)
    if df is None or df.empty:
        print(f"No data for stock: {stock_code}")
        return

    # print(f"Original DataFrame shape: {df.shape}")

    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    df.set_index('Date', inplace=True)

    # Handle missing values
    df = handle_missing_values(df)
    if df is None:
        print(f"Data after missing value handling is empty for stock: {stock_code}")
        return

    if check_missing_data(df):
        print(f"Too much missing data for stock: {stock_code}")
        return

    df = prepare_stock_data_analysis(df)
    if df is None or df.empty:
        print(f"Data preparation failed for stock: {stock_code}")
        return

    df = df[~df.index.duplicated(keep='first')]
    # print(f"DataFrame shape after deduplication: {df.shape}")

    df = df['Last trade price']
    periods = range(-1, -6, -1)
    lag_num = len(periods)
    lags = df.shift(periods=periods)
    lags.dropna(inplace=True)
    final = pd.merge(df, lags, right_index=True, left_index=True)

    test_x, train_x, test_y, train_y = train_test_split(final.drop(columns=['Last trade price']),
                                                        final['Last trade price'], test_size=0.7, shuffle=False)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    train_y = train_y.values.reshape(-1, 1)
    test_y = test_y.values.reshape(-1, 1)

    y_scaler = MinMaxScaler(feature_range=(0, 1))
    train_y = y_scaler.fit_transform(train_y)
    test_y = y_scaler.transform(test_y)

    model = Sequential([
        Input(shape=(lag_num, 1)),
        LSTM(64, return_sequences=True),
        LSTM(32, return_sequences=True),
        LSTM(16, return_sequences=False),
        Dense(35, activation='relu'),
        BatchNormalization(),
        Dense(1)  # Linear activation for regression
    ])

    model.compile(optimizer=Adam(learning_rate=0.001, clipvalue=1.0), loss='mse', metrics=['mae'])
    model.fit(train_x, train_y, epochs=50, batch_size=32)

    wasabi.save_model_to_cloud(stock_code, model)
    print(f"Model saved to cloud for stock: {stock_code}")


def main():
    for stock in ['UNI', 'USJE', 'VARG', 'VFPM', 'VITA', 'VROS', 'VSC', 'VTKS', 'ZAS', 'ZILU', 'ZILUP', 'ZIMS', 'ZKAR',
                  'ZPKO', 'ZPOG', 'ZSIL', 'ZUAS']:
        train_model(stock)


if __name__ == "__main__":
    main()
