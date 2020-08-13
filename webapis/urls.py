from django.urls import include, path
from webapis import views
from rest_framework import routers

router = routers.DefaultRouter()

urlpatterns = [
    path('', include(router.urls)),
    path('animals10', views.Animal10Predict.as_view()),
    path('vgg16fx', views.VGG16APIFX.as_view()),
]
