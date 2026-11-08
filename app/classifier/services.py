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
MAX_EXPLAINED_WORDS = 60

# Same cues the notebook's EDA found to separate spam from ham (section 2.1).
SIGNAL_PATTERNS = {
    'link': (re.compile(r'(https?://|www\.|\.(com|co\.uk|net|org)\b)', re.I), 'Contains a link'),
    'long_number': (re.compile(r'\b\d{5,}\b'), 'Contains a short code or phone number'),
    'money': (re.compile(r'[£$€]|\b\d+\s?(p|pence|gbp|usd|eur)\b', re.I), 'Mentions money or a price'),
    'prize': (re.compile(r'\b(free|win|won|winner|prize|claim|reward|award|cash|bonus|gana|ganaste|premio|gratis)\b', re.I), 'Uses prize or giveaway language'),
    'urgency': (re.compile(r'\b(urgent|now|today|immediately|expires?|final|last chance|act|ahora|urgente|hoy)\b', re.I), 'Pushes you to act fast'),
}


@dataclass
class WordWeight:
    text: str
    weight: float

    @property
    def strength(self):
        # 0..1 highlight intensity; a swing of 3 log-odds from one word is already decisive.
        return round(min(1.0, abs(self.weight) / 3), 2)


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
        return self._tokenizer, self._model

    def spam_log_odds(self, texts):
        """log P(spam) - log P(ham) for each text."""
        import torch

        tokenizer, model = self._load()
        with torch.inference_mode():
            batch = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            logits = model(**batch).logits
            return (logits[:, 1] - logits[:, 0]).tolist()

    def explain(self, text, base_log_odds):
        """Word-level occlusion: how far the spam log-odds fall when each word is removed.

        Log-odds rather than probability, because a confident 0.999 barely moves in probability
        space when one word goes, which would hide which words mattered.
        """
        words = text.split()
        if len(words) < 2 or len(words) > MAX_EXPLAINED_WORDS:
            return [WordWeight(w, 0.0) for w in words]
        variants = [' '.join(words[:i] + words[i + 1:]) for i in range(len(words))]
        scores = self.spam_log_odds(variants)
        return [WordWeight(w, round(base_log_odds - s, 3)) for w, s in zip(words, scores)]

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
            words=[{**asdict(w), 'strength': w.strength} for w in self.explain(text, log_odds)],
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
