from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing'),
    path('app/', views.app_page, name='app'),
    path('api/predict/', views.predict_api, name='predict_api'),
]
