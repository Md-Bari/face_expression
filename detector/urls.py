"""
URLs for the detector app.
"""
from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.index_view, name='index'),
    path('api/predict/', views.api_predict, name='api_predict'),
    path('api/history/', views.api_history, name='api_history'),
    path('api/history/delete/', views.api_history_delete, name='api_history_clear'),
    path('api/history/delete/<int:record_id>/', views.api_history_delete, name='api_history_delete_single'),
    path('api/models/', views.api_models_info, name='api_models'),
    path('api/export/', views.api_export_history, name='api_export'),
]
