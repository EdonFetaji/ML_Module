# import datetime
# import os
# import django
# from celery import shared_task
# from django.conf import settings
#
# # os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Crystal.settings')
# django.setup()
# from ML_Module.models import StockPredictionModel
# import io
# import pandas as pd
# from dotenv import load_dotenv
# load_dotenv()
#
# import pandas_ta as ta
# import numpy as np
#
# from keras.api.models import load_model
# from ML_Module.utils.WassabiClient import initialize_wasabi_client
#
# # Initialize the Wasabi client
# wasabi = initialize_wasabi_client()
#
# def prepare_stock_data_analysis(df):
#     try:
#         numeric_columns = [
#             'Last trade price', 'Max', 'Min', 'Avg. Price',
#             '%chg.', 'Volume', 'Turnover in BEST in denars', 'Total turnover in denars'
#         ]
#         for col in numeric_columns:
#             if col in df.columns:
#                 df[col] = df[col].astype(str).str.replace('.', '').str.replace(',', '.').astype(float)
#
#         return df
#     except Exception as e:
#         print(f"Error preparing stock data: {e}")
#         return None
#
# def technical_indicators(df):
#     try:
#         tech_indicators = pd.DataFrame(index=df.index)
#
#         tech_indicators['garman_klass_vol'] = (
#                 (np.log(df['Max']) - np.log(df['Min'])) ** 2 / 2 -
#                 (2 * np.log(2) - 1) * (np.log(df['Last trade price']) - np.log(df['Avg. Price'])) ** 2
#         )
#
#         tech_indicators['atr'] = ta.atr(high=df['Max'], low=df['Min'], close=df['Last trade price'], length=15)
#
#         tech_indicators['RSI'] = ta.rsi(df['Last trade price'], length=15)
#         tech_indicators['EMAF'] = ta.ema(df['Last trade price'], length=15)
#
#         return tech_indicators
#     except Exception as e:
#         print(f"Error calculating technical indicators: {e}")
#         return pd.DataFrame()
#
# def predict_and_insert(stock_code):
#     """
#     Load a model from Wasabi, use it for prediction, and insert the result into the database.
#
#     Args:
#         stock_code (str): The stock code associated with the prediction model.
#     """
#     try:
#         # Fetch the stock object
#
#         # Fetch the associated prediction model entry
#         model_entry = StockPredictionModel.objects.get(stock=stock)
#
#         # Download the model from Wasabi
#         cloud_key = model_entry.cloud_key
#         model_file = wasabi.fetch_model(stock_code)
#
#         if not model_file:
#             print(f"Model for {stock_code} could not be fetched from Wasabi.")
#             return
#
#         # Load the model
#         model = load_model(model_file)
#
#         # Fetch stock data
#         raw_data = wasabi.fetch_data(stock_code)
#         if raw_data is None:
#             print(f"No data available for stock {stock_code}.")
#             return
#
#         # Prepare and analyze stock data
#         df = prepare_stock_data_analysis(raw_data)
#         if df is None or df.empty:
#             print(f"Stock data for {stock_code} is not valid.")
#             return
#
#         df = df[~df.index.duplicated(keep='first')]
#         tech_indicators = technical_indicators(df)
#
#         if tech_indicators.empty:
#             print(f"No technical indicators calculated for {stock_code}.")
#             return
#
#         # Merge data with indicators
#         final_df = pd.concat([df, tech_indicators], axis=1).dropna()
#         input_data = final_df.values  # Convert to NumPy array for prediction
#
#         # Make a prediction
#         prediction = model.predict(input_data)[-1][0]  # Get the latest prediction
#
#         # Update the database with the new prediction
#         model_entry.last_prediction = prediction
#         model_entry.last_used_for_prediction = datetime.date.today()
#         model_entry.save()
#
#         print(f"Prediction for {stock_code} saved successfully: {prediction}")
#
#         return prediction
#
#     except Stock.DoesNotExist:
#         print(f"Stock with code {stock_code} does not exist.")
#
#     except StockPredictionModel.DoesNotExist:
#         print(f"Prediction model for stock {stock_code} does not exist.")
#
#     except Exception as e:
#         print(f"An error occurred: {e}")
#
# @shared_task(bind=True)
# def predict_new_data():
#     stocks =[x['code'] for x in  Stock.objects.all().values('code')]
#     for stock in stocks:
#         predict_and_insert(stock)
#
# # if __name__ == "__main__":
# #     # Run for all stocks
# #     main()
#
#
