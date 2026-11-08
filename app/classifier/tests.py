"""Fast tests: the transformer is replaced by a stub so they run in milliseconds.

Run with: python manage.py test
"""
from unittest import mock

from django.test import SimpleTestCase
from django.urls import reverse

from . import services
from .services import SpamClassifier, detect_signals


def stub_classifier(threshold=0.5):
    clf = SpamClassifier.__new__(SpamClassifier)
    clf.source, clf.is_finetuned, clf.threshold = 'stub', True, threshold
    # "prize" is worth +4 log-odds of spam; everything else is neutral.
    clf.spam_log_odds = lambda texts: [2.0 if 'prize' in t.lower() else -2.0 for t in texts]
    return clf


class ClassifierLogicTests(SimpleTestCase):
    def test_threshold_decides_label(self):
        clf = stub_classifier(threshold=0.5)
        self.assertEqual(clf.predict('You won a prize').label, 'spam')
        self.assertEqual(clf.predict('See you at lunch').label, 'ham')

    def test_explanation_credits_the_decisive_word(self):
        words = stub_classifier().predict('Claim your prize today').words
        weights = {w['text']: w['weight'] for w in words}
        self.assertEqual(weights['prize'], 4.0)
        self.assertEqual(weights['Claim'], 0.0)

    def test_signals_follow_eda_cues(self):
        signals = detect_signals('WIN £500 cash now! Call 0800123456 or visit www.example.com')
        self.assertEqual(len(signals), 5)
        self.assertEqual(detect_signals('see you later'), [])


@mock.patch.object(services, '_classifier', stub_classifier())
class ViewTests(SimpleTestCase):
    def test_page_renders(self):
        response = self.client.get(reverse('index'))
        self.assertContains(response, 'Revisar mi mensaje')

    def test_form_post_without_js(self):
        response = self.client.post(reverse('index'), {'message': 'Free prize inside'})
        self.assertContains(response, 'Parece spam')

    def test_fetch_post_returns_only_the_result(self):
        response = self.client.post(reverse('index'), {'message': 'Lunch?'}, headers={'X-Requested-With': 'fetch'})
        self.assertContains(response, 'Parece legítimo')
        self.assertNotContains(response, '<html')

    def test_api_json(self):
        response = self.client.post(reverse('api_classify'), {'message': 'prize'}, content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['label'], 'spam')

    def test_api_rejects_empty_and_too_long(self):
        url = reverse('api_classify')
        self.assertEqual(self.client.post(url, {'message': ''}, content_type='application/json').status_code, 400)
        self.assertEqual(self.client.post(url, {'message': 'a' * 2001}, content_type='application/json').status_code, 400)
        self.assertEqual(self.client.post(url, 'not json', content_type='application/json').status_code, 400)
        latin1_body = '{"message": "\u00a35"}'.encode('latin-1')
        self.assertEqual(self.client.post(url, latin1_body, content_type='application/json').status_code, 400)
