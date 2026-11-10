# SpamLens: SMS spam detection with a pretrained transformer

**SpamLens** is a Django web app that tells you whether a text message is spam, explains which words drove the verdict, and never stores what you paste. Under the hood it runs [`Goodmotion/spam-mail-classifier`](https://huggingface.co/Goodmotion/spam-mail-classifier), a Multilingual MiniLM transformer, fine-tuned by us on the [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

Final project for the **Hacking AI** course, Master's program, 3rd semester.

## Team

| Name | GitHub |
|---|---|
| Juan José Aguado | [@Aguado4](https://github.com/Aguado4) |
| Juan David Cruz | [@1KVueltasAlCampo](https://github.com/1KVueltasAlCampo) |
| Juan Diego Ramírez | [@jdramirezzu](https://github.com/jdramirezzu) |

## What the project does

1. **Exploratory analysis** of 5,574 labeled SMS messages: class balance, duplicates, length, digits, links, money symbols and vocabulary of spam versus legitimate ("ham") messages.
2. **Privacy-aware statistics**: following Dwork and Roth, *The Algorithmic Foundations of Differential Privacy* (the course reading), the EDA histograms are released with the Laplace mechanism and the privacy/accuracy trade-off is measured.
3. **Modeling and evaluation** on a deduplicated, stratified 70/15/15 split:
   * a TF-IDF + logistic regression baseline,
   * the pretrained model used as published (zero-shot),
   * the same pretrained model fine-tuned on the SMS training split.
4. **A web application** that serves the fine-tuned model with a word-level explanation of every verdict and a JSON API.

## Key results

Spam-class metrics on the held-out test set (774 messages, 96 spam).

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | False positives | False negatives |
|---|---|---|---|---|---|---|---|
| Pretrained, as published (zero-shot) | 0.500 | 0.136 | 0.688 | 0.228 | 0.617 | 418 | 30 |
| TF-IDF + logistic regression baseline | 0.500 | 0.989 | 0.948 | 0.968 | 0.998 | 1 | 5 |
| Pretrained, fine-tuned on SMS (deployed) | 0.479 | 0.968 | 0.958 | 0.963 | 0.999 | 3 | 4 |

**Main findings**

* The pretrained model was trained on *email subjects*. Used as published, it flags most ordinary SMS as spam: a domain shift that no threshold can fix.
* Fine-tuning the same network on the SMS training split (about 8 minutes on a laptop CPU) lifts spam F1 from about 0.23 to about 0.96.
* A well-tuned TF-IDF + logistic regression model is slightly better on English SMS in F1 (0.968 vs 0.963), precision (0.989 vs 0.968) and false positives (1 vs 3), though the gap is one or two messages out of 774 and not conclusive. The transformer only edges it out in ROC-AUC and average precision, and still catches typical Spanish scams, which the TF-IDF model misses entirely. We ship the transformer because the assignment asks for the pretrained model and for its partial multilingual coverage, not for a better F1. English-only fine-tuning did cost some multilingual ability: a French prize scam the original model caught now slips through.

The full reasoning, with charts, is in the notebook.

## Repository structure

```
.
├── app/                         Django web application
│   ├── manage.py
│   ├── spamlens/                project settings and URLs
│   └── classifier/              the app: model service, views, API, templates, static files, tests
│       ├── services.py          model loading, inference, word-level explanations
│       ├── views.py             page (works without JavaScript) and JSON API
│       ├── model_card.json      threshold and test metrics written by the notebook
│       └── management/commands/download_model.py
├── notebook/
│   └── spam_detection.ipynb     step-by-step data science process, executed with outputs
├── data/
│   └── sms+spam+collection.zip  SMS Spam Collection (UCI), read directly from the zip
├── model/                       model weights (git-ignored, see model/README.md)
├── requirements.txt
└── README.md
```

## How to run it

Requirements: Python 3.12 and about 2 GB of free RAM. Everything runs on CPU; no GPU needed.

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/Aguado4/spamlens-hacking-ai-Aguado-Cruz-Ramirez.git
cd spamlens-hacking-ai-Aguado-Cruz-Ramirez
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate

# 2. Install dependencies (the first line installs the small CPU-only build of PyTorch)
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Get the fine-tuned model (about 240 MB, from this repo's GitHub release)
python app/manage.py download_model

# 4. Start the web app
python app/manage.py runserver
```

Open http://127.0.0.1:8000. The first check takes a few seconds while the model loads; after that each verdict takes well under a second.

If you skip step 3 the app still works: it falls back to the original `Goodmotion/spam-mail-classifier` from the Hugging Face Hub and shows a notice, but expect many false alarms on SMS (see the results above). To point the app to any other model folder or Hub id, set `SPAMLENS_MODEL`.

### Reproducing the notebook

```bash
cd notebook
jupyter notebook spam_detection.ipynb
```

Run all cells. It downloads the pretrained model from the Hub (or uses `model/` if you placed a local copy there), fine-tunes it on CPU (about 10 to 20 minutes on a laptop), writes the weights to `model/finetuned/` and refreshes `app/classifier/model_card.json`, so the numbers on the website always match the latest run.

### API

```bash
curl -X POST http://127.0.0.1:8000/api/classify/ \
     -H "Content-Type: application/json" \
     -d '{"message": "URGENT! You have won a £1000 prize. Call 09061701461 to claim"}'
```

Returns the label, the spam probability, the decision threshold, the per-word weights and the spam cues detected. `GET /api/health/` reports which model is loaded.

### Tests

```bash
cd app
python manage.py test
```

The tests replace the transformer with a stub, so they run in milliseconds and do not need the weights.

## Design notes

* **No database.** Messages are classified in memory and never logged or stored. This is the data-minimization lesson from the privacy reading applied to the product.
* **Threshold chosen on validation, never on test.** We compared a default 0.5, max-F1 and precision-first rules on the validation split. The fine-tuned model's probabilities are very peaked, so with only 97 validation spam messages the precision-first rules pushed the threshold above 0.99 and gave up recall for noise; max-F1 was the stable choice. The test set is used once, at the end.
* **Explanations with Integrated Gradients.** The model's spam log-odds are attributed to each word by integrating gradients along a path from a neutral input to the real message. Removing one word at a time does not work for spam, because cues like "FREE", a prize amount and a short code back each other up and every single removal looks harmless. Words are highlighted in red (toward spam) or green (toward legitimate), scaled relative to the strongest word in the message.
* **Progressive enhancement.** The page works with JavaScript disabled (plain form post); with JavaScript it updates in place, animates the result and supports Ctrl+Enter.

## Data and model credits

* Almeida, T.A., Gómez Hidalgo, J.M. (2011). *SMS Spam Collection*. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84
* Goodmotion. *spam-mail-classifier*, fine-tuned from `microsoft/Multilingual-MiniLM-L12-H384`. Apache 2.0. https://huggingface.co/Goodmotion/spam-mail-classifier
* Dwork, C., Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy*. Foundations and Trends in Theoretical Computer Science, 9(3-4), 211-407.
