"""Model loading and inference for SpamLens.

The model is loaded once per process, lazily, behind a lock: Django may serve requests from several
threads and a 120M-parameter model should never be loaded twice.
"""
import json
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from django.conf import settings

MODEL_CARD_PATH = Path(__file__).resolve().parent / 'model_card.json'
IG_STEPS = 16
# A word is highlighted when it carries at least this share of the message's strongest attribution.
HIGHLIGHT_SHARE = 0.25

# Same cues the notebook's EDA found to separate spam from ham (section 2.1).
SIGNAL_PATTERNS = {
    'link': (re.compile(r'(https?://|www\.|\.(com|co\.uk|net|org)\b)', re.I), 'Contiene un enlace'),
    'long_number': (re.compile(r'\b\d{5,}\b'), 'Contiene un código corto o un número de teléfono'),
    'money': (re.compile(r'[£$€]|\b\d+\s?(p|pence|gbp|usd|eur)\b', re.I), 'Menciona dinero o un precio'),
    'prize': (re.compile(r'\b(free|win|won|winner|prize|claim|reward|award|cash|bonus|gana|ganaste|premio|gratis)\b', re.I), 'Usa lenguaje de premios o sorteos'),
    'urgency': (re.compile(r'\b(urgent|now|today|immediately|expires?|final|last chance|act|ahora|urgente|hoy)\b', re.I), 'Te presiona para actuar rápido'),
}


@dataclass
class WordWeight:
    text: str
    weight: float
    strength: float = 0.0
    direction: str = ''


def word_weights(words, attributions):
    """Scale attributions to 0..1 relative to the strongest word in this message."""
    peak = max((abs(a) for a in attributions), default=0.0)
    out = []
    for word, a in zip(words, attributions):
        share = abs(a) / peak if peak > 1e-6 else 0.0
        direction = ('spam' if a > 0 else 'ham') if share >= HIGHLIGHT_SHARE else ''
        out.append(WordWeight(word, round(a, 3), round(share, 2), direction))
    return out


@dataclass
class Prediction:
    label: str
    spam_probability: float
    threshold: float
    confidence: float
    words: list = field(default_factory=list)
    signals: list = field(default_factory=list)
    latency_ms: int = 0

    def to_dict(self):
        return asdict(self)


@lru_cache(maxsize=1)
def model_card():
    if MODEL_CARD_PATH.exists():
        return json.loads(MODEL_CARD_PATH.read_text(encoding='utf-8'))
    return {}


def resolve_model_source():
    """Return (source, is_finetuned). See SPAMLENS_* settings for the order."""
    if settings.SPAMLENS_MODEL:
        return settings.SPAMLENS_MODEL, Path(settings.SPAMLENS_MODEL).resolve() == Path(settings.SPAMLENS_FINETUNED_DIR).resolve()
    if (Path(settings.SPAMLENS_FINETUNED_DIR) / 'config.json').exists():
        return str(settings.SPAMLENS_FINETUNED_DIR), True
    return settings.SPAMLENS_PRETRAINED_ID, False


def detect_signals(text):
    return [label for pattern, label in SIGNAL_PATTERNS.values() if pattern.search(text)]


class SpamClassifier:
    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self.source, self.is_finetuned = resolve_model_source()
        # The validated threshold belongs to the fine-tuned model; the raw hub model keeps 0.5.
        self.threshold = model_card().get('threshold', 0.5) if self.is_finetuned else 0.5

    def _load(self):
        with self._lock:
            if self._model is None:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.source)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.source, dtype=torch.float32).eval()
                self._model.requires_grad_(False)
        return self._tokenizer, self._model

    def spam_log_odds(self, texts):
        """log P(spam) - log P(ham) for each text."""
        import torch

        tokenizer, model = self._load()
        with torch.inference_mode():
            batch = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            logits = model(**batch).logits
            return (logits[:, 1] - logits[:, 0]).tolist()

    def explain(self, words):
        """Integrated Gradients over the input embeddings, summed per word.

        Removing one word at a time fails on spam: with "FREE", "prize", "£1000" and a short code in
        the same message, any single word is redundant and every score looks like zero. Integrated
        Gradients walks from a neutral input to the real one and shares the log-odds among all the
        words that carry the signal, even when they repeat each other.
        """
        import torch

        tokenizer, model = self._load()
        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=128, return_tensors='pt')
        embed = model.get_input_embeddings()
        with torch.no_grad():
            actual = embed(enc['input_ids'])
            special = torch.tensor(tokenizer.get_special_tokens_mask(enc['input_ids'][0].tolist(), already_has_special_tokens=True), dtype=torch.bool)
            neutral = actual.clone()
            neutral[0, ~special] = embed.weight[tokenizer.pad_token_id]
        alphas = torch.linspace(1 / IG_STEPS, 1, IG_STEPS).view(-1, 1, 1)
        path = (neutral + alphas * (actual - neutral)).requires_grad_(True)
        with torch.enable_grad():
            logits = model(inputs_embeds=path, attention_mask=enc['attention_mask'].expand(IG_STEPS, -1)).logits
            grads = torch.autograd.grad((logits[:, 1] - logits[:, 0]).sum(), path)[0]
        per_token = ((actual - neutral)[0] * grads.mean(0)).sum(-1).tolist()

        per_word = [0.0] * len(words)
        for i, word_id in enumerate(enc.word_ids(0)):
            if word_id is not None:
                per_word[word_id] += per_token[i]
        return word_weights(words, per_word)

    def predict(self, text):
        start = time.perf_counter()
        log_odds = self.spam_log_odds([text])[0]
        probability = 1 / (1 + math.exp(-log_odds))
        is_spam = probability >= self.threshold
        return Prediction(
            label='spam' if is_spam else 'ham',
            spam_probability=round(probability, 4),
            threshold=self.threshold,
            confidence=round(probability if is_spam else 1 - probability, 4),
            words=[asdict(w) for w in self.explain(text.split())],
            signals=detect_signals(text),
            latency_ms=round((time.perf_counter() - start) * 1000),
        )


_classifier = None
_classifier_lock = threading.Lock()


def get_classifier():
    global _classifier
    with _classifier_lock:
        if _classifier is None:
            _classifier = SpamClassifier()
    return _classifier
