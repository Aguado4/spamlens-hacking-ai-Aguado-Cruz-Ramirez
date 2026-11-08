# Model weights

This folder is ignored by git because the weights are large (about 470 MB for the original model, 240 MB for the fine-tuned fp16 copy).

| Folder | What it is | How to get it |
|---|---|---|
| `model/finetuned/` | `Goodmotion/spam-mail-classifier` fine-tuned on the SMS training split. **Used by the web app.** | `python app/manage.py download_model`, or run the notebook end to end |
| `model/` (root files) | Optional local copy of the original [`Goodmotion/spam-mail-classifier`](https://huggingface.co/Goodmotion/spam-mail-classifier) | `huggingface-cli download Goodmotion/spam-mail-classifier --local-dir model` (otherwise the notebook downloads it from the Hub automatically) |
