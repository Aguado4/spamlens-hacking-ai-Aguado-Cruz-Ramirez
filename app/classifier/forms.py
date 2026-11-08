from django import forms
from django.conf import settings


class MessageForm(forms.Form):
    message = forms.CharField(
        max_length=settings.SPAMLENS_MAX_CHARS,
        strip=True,
        widget=forms.Textarea(attrs={
            'rows': 5,
            'placeholder': 'Pega aquí el mensaje de texto que recibiste...',
            'aria-describedby': 'message-help',
            'spellcheck': 'false',
        }),
        error_messages={
            'required': 'Primero pega un mensaje.',
            'max_length': f'Los mensajes están limitados a {settings.SPAMLENS_MAX_CHARS} caracteres.',
        },
    )
