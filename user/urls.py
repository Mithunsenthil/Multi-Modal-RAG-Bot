from django.urls import path
from .views import register, editprofile, home

urlpatterns = [
    path('register/',register.as_view(),name='register'),
    path('edit_profile/',editprofile.as_view(),name='edit_profile'),
    path('home/',home.as_view(),name='home'),
]
