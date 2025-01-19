from django.urls import path
from .views import get_stock_prediction

urlpatterns = [
    path('stock/<str:stock_code>/predict/<int:days>/', get_stock_prediction, name='get_stock_prediction')
]