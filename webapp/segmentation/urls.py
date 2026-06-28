from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing'),
    path('app/', views.app_page, name='app'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('about/', views.about_page, name='about'),
    path('contact/', views.contact_page, name='contact'),
    path('terms/', views.terms_page, name='terms'),
    path('dashboard/', views.dashboard_page, name='dashboard'),
    path('registry/', views.registry_page, name='registry'),
    path('settings/', views.settings_page, name='settings'),
    path('profile/', views.profile_page, name='profile'),
    path('api/predict/', views.predict_api, name='predict_api'),
    path('api/chat/<int:scan_id>/', views.chat_api, name='chat_api'),
    path('api/scan/<int:scan_id>/delete/', views.delete_scan_api, name='delete_scan'),
    path('api/scan/<int:scan_id>/recalculate_risk/', views.recalculate_risk_api, name='recalculate_risk'),
    path('api/consult/request/<int:scan_id>/', views.request_consult_api, name='request_consult'),
    path('api/consult/<int:consult_id>/messages/', views.consult_messages_api, name='consult_messages'),
    path('api/consult/<int:consult_id>/invite/', views.invite_specialist_api, name='invite_specialist'),
    path('api/consult/<int:consult_id>/signoff/', views.signoff_consult_api, name='signoff_consult'),
    path('api/patient/history/', views.patient_history_api, name='patient_history'),
]
