from datetime import datetime, date, timedelta

from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import subprocess
from rest_framework.generics import ListCreateAPIView
from sklearn.preprocessing import MinMaxScaler

from .models import StockPredictionModel
from .serializers import StockPredictionModelSerializer

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .models import StockPredictionModel
import json
import pandas as pd
from utils.WassabiClient import WasabiClient, get_wassabi_client

cloudClient: WasabiClient = get_wassabi_client()


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


@csrf_exempt
def get_stock_prediction(request, stock_code, days):
    db_model = StockPredictionModel.objects.get(stock_code=stock_code)
    if db_model is None:
        return JsonResponse({'stock_code': stock_code, 'error': f'Model for stock {stock_code} is not trained'})

    ml_model = cloudClient.fetch_model(stock_code)

    if not ml_model:
        return JsonResponse({'error': f'Model for stock {stock_code} is not trained'}, status=400)

    # Fetch stock data
    raw_data = cloudClient.fetch_data(stock_code)
    if raw_data is None:
        return JsonResponse({'error': f'NO DATA for stock {stock_code}'}, status=400)

    # Prepare and analyze stock data
    df = prepare_stock_data_analysis(raw_data)
    df = handle_missing_values(df)
    df = df[~df.index.duplicated(keep='first')]

    df = df['Last trade price']
    scaler = MinMaxScaler()
    scaler.fit(df['Last trade price'])
    #
    today = date.today()
    predictions = {}
    for i in range(1, days + 1):
        periods = range(-1, -5, -1)
        lags = df.shift(periods=periods)
        lags.dropna(inplace=True)
        final = pd.merge(df, lags, right_index=True, left_index=True)
        last_entry = final.loc[0]
        last_entry = scaler.transform(last_entry.reshape(-1, 1))
        prediction = ml_model.predict(last_entry.reshape(1, -1, 1))
        prediction = scaler.inverse_transform(prediction)[0][0]

        predictions[today.__add__(timedelta(days=i).strftime('%d.%m.%Y'))] = prediction

        new_row = pd.DataFrame({'Last trade price': prediction})
        df = pd.concat([new_row, df]).reset_index(drop=True)

    if predictions is None:
        return JsonResponse({'error': 'Prediction could not be generated'}, status=400)

    return JsonResponse({'stock_code': stock_code, 'predictions': predictions})

# # Endpoint to fetch status for a particular stock
# def model_status_view(request):
#     stock_code = request.GET.get('stock_code')
#
#     if not stock_code:
#         return JsonResponse({"error": "Missing 'stock_code' in query parameters"}, status=400)
#
#     try:
#         stock = Stock.objects.get(code=stock_code)
#         prediction_model = StockPredictionModel.objects.filter(stock=stock).order_by('-created_at').first()
#
#         if prediction_model:
#             return JsonResponse({
#                 "stock_code": stock.code,
#                 "last_prediction": prediction_model.last_prediction,
#                 "last_modified": stock.last_modified,
#                 "cloud_key": prediction_model.cloud_key
#             })
#         else:
#             return JsonResponse({"error": "No prediction model found for this stock"}, status=404)
#
#     except Stock.DoesNotExist:
#         return JsonResponse({"error": f"Stock with code {stock_code} not found."}, status=404)
#     except Exception as e:
#         return JsonResponse({"error": str(e)}, status=500)
#
#
#


#
# class StockPredictionModelAPIView(ListCreateAPIView):
#     queryset = StockPredictionModel.objects.all()
#     serializer_class = StockPredictionModelSerializer
#
# @csrf_exempt
# def populate_db_view(request):
#     if request.method == "POST":
#         try:
#             subprocess.run(["python", "path/to/populate_db.py"], check=True)
#             return JsonResponse({"message": "Database population script executed successfully."}, status=200)
#         except subprocess.CalledProcessError as e:
#             return JsonResponse({"error": f"Error executing script: {str(e)}"}, status=500)
#     return JsonResponse({"error": "Invalid request method. Use POST."}, status=400)
#
#
# @csrf_exempt
# def train_models_view(request):
#     if request.method == "POST":
#         try:
#             subprocess.run(["python", "path/to/train_models.py"], check=True)
#             return JsonResponse({"message": "Training models script executed successfully."}, status=200)
#         except subprocess.CalledProcessError as e:
#             return JsonResponse({"error": f"Error executing script: {str(e)}"}, status=500)
#     return JsonResponse({"error": "Invalid request method. Use POST."}, status=400)
#
#
# @csrf_exempt
# def predict_view(request):
#     if request.method == "POST":
#         try:
#             subprocess.run(["python", "path/to/predict.py"], check=True)
#             return JsonResponse({"message": "Prediction script executed successfully."}, status=200)
#         except subprocess.CalledProcessError as e:
#             return JsonResponse({"error": f"Error executing script: {str(e)}"}, status=500)
#     return JsonResponse({"error": "Invalid request method. Use POST."}, status=400)
