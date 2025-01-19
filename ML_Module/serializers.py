from rest_framework import serializers
from .models import StockPredictionModel

class StockPredictionModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = StockPredictionModel
        fields = '__all__'
