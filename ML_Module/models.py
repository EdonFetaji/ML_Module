from django.db import models

# Create your models here.
from django.db import models



class StockPredictionModel(models.Model):
    stock_code = models.CharField(max_length=50)
    last_modified = models.DateField(null=True, blank=True)
    last_used_for_prediction = models.DateField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_prediction = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    cloud_key = models.CharField(max_length=255)

    def __str__(self):
        return f"Model for {self.stock_code} ({self.cloud_key})"

    class Meta:
        ordering = ['stock_code', 'last_modified']