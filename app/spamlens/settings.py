"""Django settings for SpamLens.

The app has no database on purpose: messages are classified in memory and never stored.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BASE_DIR.parent

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'dev-only-insecure-key-change-me')
DEBUG = os.environ.get('DJANGO_DEBUG', '1') == '1'
ALLOWED_HOSTS = os.environ.get('DJANGO_ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

INSTALLED_APPS = [
    'django.contrib.staticfiles',
    'classifier',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'spamlens.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {'context_processors': ['django.template.context_processors.request']},
    },
]

WSGI_APPLICATION = 'spamlens.wsgi.application'

DATABASES = {}

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'America/Bogota'
USE_I18N = False
USE_TZ = True

STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Model resolution: an explicit path or hub id wins, then the fine-tuned weights produced by the
# notebook, then the original pretrained model straight from the Hugging Face Hub.
SPAMLENS_MODEL = os.environ.get('SPAMLENS_MODEL', '')
SPAMLENS_FINETUNED_DIR = REPO_DIR / 'model' / 'finetuned'
SPAMLENS_PRETRAINED_ID = 'Goodmotion/spam-mail-classifier'
SPAMLENS_MAX_CHARS = 2000
