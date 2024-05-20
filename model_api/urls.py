from django.urls import path
from .views import PredictView, open_app, download_apk, index_page

urlpatterns = [
    
    path('', index_page, name='index'),
    path('predict/', PredictView.as_view(), name='predict'),
    path('open_app/', open_app, name='open_app'),
    path('download-apk/', download_apk, name='download_apk'),
]