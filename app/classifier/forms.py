from django import forms
from django.conf import settings


class MessageForm(forms.Form):
    message = forms.CharField(
        max_length=settings.SPAMLENS_MAX_CHARS,
        strip=True,
        widget=forms.Textarea(attrs={
            'rows': 5,
            'placeholder': 'Paste the text message you received...',
            'aria-describedby': 'message-help',
            'spellcheck': 'false',
        }),
        error_messages={
            'required': 'Paste a message first.',
            'max_length': f'Messages are limited to {settings.SPAMLENS_MAX_CHARS} characters.',
        },
    )
