from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import Document

class SignupForm(UserCreationForm):
    email = forms.EmailField(required=True, help_text="")  # hide email help text

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # remove all help texts
        for field in self.fields.values():
            field.help_text = None

class DocumentUploadForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ['file', 'title']