from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing'),
    path('app/', views.app_page, name='app'),
    path('dashboard/', views.dashboard_page, name='dashboard'),
    path('registry/', views.registry_page, name='registry'),
    path('settings/', views.settings_page, name='settings'),
    path('api/predict/', views.predict_api, name='predict_api'),
    path('api/scan/<int:scan_id>/delete/', views.delete_scan_api, name='delete_scan'),
]
