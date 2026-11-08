from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/classify/', views.api_classify, name='api_classify'),
    path('api/health/', views.health, name='health'),
]
