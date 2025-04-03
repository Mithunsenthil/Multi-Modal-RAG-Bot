
from django.db.models.base import Model as Model
from django.db.models.query import QuerySet
from django.shortcuts import render
from django.views import generic 
from .forms import register, editprofile
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django.urls import reverse_lazy

# Create your views here.

class register(generic.CreateView):
    form_class = register
    template_name = 'registration/register.html'
    success_url = reverse_lazy('login')

class editprofile(generic.UpdateView):
    form_class = editprofile
    template_name = 'registration/editprofile.html'
    success_url = reverse_lazy('home')

    def get_object(self):
        return self.request.user

class home(generic.TemplateView):
    template_name = 'registration/home.html'