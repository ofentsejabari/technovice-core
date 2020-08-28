
from django.contrib import admin
from django.urls import path, include
from streamapp import views

urlpatterns = [
    path('', views.index, name='index'),
	path('livecam_feed', views.livecam_feed, name='livecam_feed'),
    path('livefeed', views.LiveFeed.as_view()),
]
