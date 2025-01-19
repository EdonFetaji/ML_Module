import datetime
import os
import django
from celery import shared_task
from django.conf import settings
from sklearn.preprocessing import MinMaxScaler

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Crystal.settings')
django.setup()
from ML_Module.models import StockPredictionModel
import io
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import pandas_ta as ta
import numpy as np

from keras.api.models import load_model
from utils.WassabiClient import get_wassabi_client

# Initialize the Wasabi client
wasabi = get_wassabi_client()


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
        print(f"Error preparing stock data: {e}")
        return None


def handle_missing_values(df):
    df['Last trade price'] = df['Last trade price'].bfill()

    return df


def predict_and_insert(stock_code):
    """
    Load a model from Wasabi, use it for prediction, and insert the result into the database.

    Args:
        stock_code (str): The stock code associated with the prediction model.
    """
    try:
        # Fetch the stock object

        # Fetch the associated prediction model entry
        model_entry = StockPredictionModel.objects.get(stock=stock_code)

        ml_model = wasabi.fetch_model(stock_code)

        if not ml_model:
            print(f"Model for {stock_code} could not be fetched from Wasabi.")
            return

        # Fetch stock data
        raw_data = wasabi.fetch_data(stock_code)
        if raw_data is None:
            print(f"No data available for stock {stock_code}.")
            return

        # Prepare and analyze stock data
        df = prepare_stock_data_analysis(raw_data)
        df = handle_missing_values(df)
        if df is None or df.empty:
            print(f"Stock data for {stock_code} is not valid.")
            return

        df = df[~df.index.duplicated(keep='first')]

        df = df['Last trade price']

        periods = range(-1, -5, -1)
        lags = df.shift(periods=periods)
        lags.dropna(inplace=True)
        final = pd.merge(df, lags, right_index=True, left_index=True)

        scaler = MinMaxScaler()
        scaler.fit(final['Last trade price'])

        last_entry = final.loc[0]
        last_entry = scaler.transform(last_entry.reshape(-1, 1))


        prediction = ml_model.predict(last_entry.reshape(1,-1,1))
        prediction = scaler.inverse_transform(prediction)[0][0]

        # Update the database with the new prediction
        model_entry.last_prediction = prediction
        model_entry.last_used_for_prediction = datetime.date.today()
        model_entry.save()

        print(f"Prediction for {stock_code} saved successfully: {prediction}")

        return prediction

    except StockPredictionModel.DoesNotExist:
        print(f"Prediction model for stock {stock_code} does not exist.")

    except Exception as e:
        print(f"An error occurred: {e}")


@shared_task(bind=True)
def predict_new_data():
    stocks = [x['stock_code'] for x in StockPredictionModel.objects.all().values('stock_code')]
    for stock in stocks:
        predict_and_insert(stock)

# if __name__ == "__main__":
#     # Run for all stocks
#     main()
