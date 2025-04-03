from typing import Any
from django import forms
from django.contrib.auth.forms import UserCreationForm,UserChangeForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError

class register(UserCreationForm):
    email = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'form-control'}))
    first_name = forms.CharField(max_length=225, widget=forms.TextInput(attrs={'class': 'form-control'}))
    last_name = forms.CharField(max_length=225, widget=forms.TextInput(attrs={'class': 'form-control'}))

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')

    def __init__(self, *args, **kwargs):
        super(register, self).__init__(*args, **kwargs)

        self.fields["username"].widget.attrs['class'] = 'form-control'
        self.fields["password1"].widget.attrs['class'] = 'form-control'
        self.fields["password2"].widget.attrs['class'] = 'form-control'

    def clean_password2(self):
        # Override the password validation logic
        password1 = self.cleaned_data.get("password1")
        password2 = self.cleaned_data.get("password2")

        if password1 and password2 and password1 != password2:
            raise ValidationError("Passwords don't match.")
        
        # Remove or relax constraints here
        if len(password2) < 6:  # Lower minimum length to 6 characters
            raise ValidationError("Password must be at least 6 characters long.")

        return password2


class editprofile(UserChangeForm):
    email = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'form-control'}))
    first_name = forms.CharField(max_length=225, widget=forms.TextInput(attrs={'class': 'form-control'}))
    last_name = forms.CharField(max_length=225, widget=forms.TextInput(attrs={'class': 'form-control'}))
    username = forms.CharField(max_length=225, widget=forms.TextInput(attrs={'class': 'form-control'}))

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name' , 'email')