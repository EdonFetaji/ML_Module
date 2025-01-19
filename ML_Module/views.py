from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import subprocess
from rest_framework.generics import ListCreateAPIView
from .models import StockPredictionModel
from .serializers import StockPredictionModelSerializer

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .models import  StockPredictionModel
import json

#
# @csrf_exempt
# def get_stock_prediction(request, stock_code):
#     """
#     View that returns the predicted value for the given stock code,
#     and saves it to the database.
#     """
#     try:
#         # Get the predicted value for the stock and save it to the database
#         prediction = predict_and_insert(stock_code)
#
#         if prediction is None:
#             return JsonResponse({'error': 'Prediction could not be generated'}, status=400)
#
#         return JsonResponse({'stock_code': stock_code, 'prediction': prediction})
#
#     except Exception as e:
#         return JsonResponse({'error': str(e)}, status=500)
#
#
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
