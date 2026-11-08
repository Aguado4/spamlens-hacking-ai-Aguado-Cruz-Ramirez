import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from .forms import MessageForm
from .services import get_classifier, model_card

SAMPLES = [
    ('Prize scam', 'URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word CLAIM to 81010 T&C www.dbuk.net'),
    ('Friend', "Hey, are we still on for dinner tonight? I can pick you up around 7 if that works"),
    ('Bank alert', 'Your account has been suspended due to unusual activity. Verify your details now at secure-bank-login.co to restore access'),
    ('Spanish', 'Felicidades! Ganaste un iPhone 17. Reclámalo ahora enviando GANAR al 55555'),
    ('Work', 'Running 10 min late to the standup, start without me and I will catch up on the notes'),
]

MODEL_LABELS = [
    ('pretrained_zero_shot', 'Pretrained, as published'),
    ('baseline_tfidf_logreg', 'TF-IDF baseline'),
    ('pretrained_fine_tuned', 'Pretrained, fine-tuned on SMS (what you are using)'),
]


def _evidence():
    """Test-set metrics from the notebook run, shaped for the comparison chart."""
    metrics = model_card().get('test_metrics', {})
    rows = []
    for key, label in MODEL_LABELS:
        m = metrics.get(key)
        if m:
            rows.append({
                'key': key,
                'label': label,
                'f1': m['f1'],
                'f1_pct': round(m['f1'] * 100, 1),
                'precision_pct': round(m['precision'] * 100, 1),
                'recall_pct': round(m['recall'] * 100, 1),
                'false_positives': m['false_positives'],
            })
    return rows


def _context(form, prediction=None):
    classifier = get_classifier()
    card = model_card()
    return {
        'form': form,
        'prediction': prediction,
        'samples': SAMPLES,
        'evidence': _evidence(),
        'card': card,
        'test_size': card.get('dataset', {}).get('split', {}).get('test'),
        'is_finetuned': classifier.is_finetuned,
        'threshold_pct': round(classifier.threshold * 100),
    }


@require_http_methods(['GET', 'POST'])
def index(request):
    """Works without JavaScript: the form posts here and the result renders server-side."""
    if request.method == 'POST':
        form = MessageForm(request.POST)
        prediction = get_classifier().predict(form.cleaned_data['message']).to_dict() if form.is_valid() else None
        # Progressive enhancement: the page's script asks for just the result panel.
        template = 'classifier/_result.html' if request.headers.get('X-Requested-With') == 'fetch' else 'classifier/index.html'
        return render(request, template, _context(form, prediction))
    return render(request, 'classifier/index.html', _context(MessageForm()))


# No session, no stored state, no side effects: CSRF protects nothing here, and exempting it lets
# scripts and curl call the API.
@csrf_exempt
@require_POST
def api_classify(request):
    if request.content_type == 'application/json':
        try:
            payload = json.loads(request.body or b'{}')
        except ValueError:  # JSONDecodeError or a body that is not valid UTF-8
            return JsonResponse({'error': 'Body must be valid JSON.'}, status=400)
        if not isinstance(payload, dict):
            return JsonResponse({'error': 'Body must be a JSON object.'}, status=400)
    else:
        payload = request.POST
    form = MessageForm({'message': payload.get('message', '')})
    if not form.is_valid():
        return JsonResponse({'error': form.errors['message'][0]}, status=400)
    return JsonResponse(get_classifier().predict(form.cleaned_data['message']).to_dict())


@require_GET
def health(request):
    classifier = get_classifier()
    return JsonResponse({'status': 'ok', 'model': classifier.source, 'fine_tuned': classifier.is_finetuned,
                         'threshold': classifier.threshold})
