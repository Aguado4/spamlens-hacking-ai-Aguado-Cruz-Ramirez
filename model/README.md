# Pesos del modelo

Esta carpeta es ignorada por git porque los pesos son pesados (cerca de 470 MB para el modelo original y 240 MB para la copia fine-tuned en fp16).

| Carpeta | Qué es | Cómo obtenerla |
|---|---|---|
| `model/finetuned/` | `Goodmotion/spam-mail-classifier` con fine-tuning sobre la partición de entrenamiento de SMS. **Es el que usa la aplicación web.** | `python app/manage.py download_model`, o ejecutar el notebook completo |
| `model/` (archivos en la raíz) | Copia local opcional del [`Goodmotion/spam-mail-classifier`](https://huggingface.co/Goodmotion/spam-mail-classifier) original | `huggingface-cli download Goodmotion/spam-mail-classifier --local-dir model` (de lo contrario, el notebook lo descarga automáticamente desde el Hub) |
