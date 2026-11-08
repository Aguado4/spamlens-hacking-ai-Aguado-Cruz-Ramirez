# SpamLens: detección de spam en SMS con un transformer preentrenado

**SpamLens** es una aplicación web en Django que indica si un mensaje de texto es spam, explica qué palabras determinaron el veredicto y nunca almacena lo que se pega. Por dentro ejecuta [`Goodmotion/spam-mail-classifier`](https://huggingface.co/Goodmotion/spam-mail-classifier), un transformer Multilingual MiniLM, al que le hicimos fine-tuning con el [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

Proyecto final del curso **Hacking AI**, Maestría, 3.er semestre.

## Equipo

| Nombre | GitHub |
|---|---|
| Juan José Aguado | [@Aguado4](https://github.com/Aguado4) |
| Juan David Cruz | [@1KVueltasAlCampo](https://github.com/1KVueltasAlCampo) |
| Juan Diego Ramírez | [@jdramirezzu](https://github.com/jdramirezzu) |

## Qué hace el proyecto

1. **Análisis exploratorio** de 5,574 mensajes SMS etiquetados: balance de clases, duplicados, longitud, dígitos, enlaces, símbolos de dinero y vocabulario de los mensajes spam frente a los legítimos ("ham").
2. **Estadísticas con privacidad**: siguiendo a Dwork y Roth, *The Algorithmic Foundations of Differential Privacy* (la lectura del curso), los histogramas del EDA se publican con el mecanismo de Laplace y se mide el equilibrio entre privacidad y exactitud.
3. **Modelado y evaluación** sobre una partición 70/15/15 estratificada y sin duplicados:
   * un baseline de TF-IDF + regresión logística,
   * el modelo preentrenado tal como fue publicado (zero-shot),
   * el mismo modelo preentrenado con fine-tuning sobre la partición de entrenamiento de SMS.
4. **Una aplicación web** que sirve el modelo con fine-tuning, con una explicación a nivel de palabra de cada veredicto y una API JSON.

## Resultados principales

Métricas de la clase spam sobre el conjunto de prueba reservado (774 mensajes, 96 spam).

| Modelo | Umbral | Precisión | Recall | F1 | ROC-AUC | Falsos positivos | Falsos negativos |
|---|---|---|---|---|---|---|---|
| Preentrenado, tal como fue publicado (zero-shot) | 0.500 | 0.136 | 0.688 | 0.228 | 0.617 | 418 | 30 |
| Baseline de TF-IDF + regresión logística | 0.500 | 0.989 | 0.948 | 0.968 | 0.998 | 1 | 5 |
| Preentrenado, con fine-tuning en SMS (desplegado) | 0.479 | 0.968 | 0.958 | 0.963 | 0.999 | 3 | 4 |

**Hallazgos principales**

* El modelo preentrenado fue entrenado con *asuntos de correos electrónicos*. Usado tal como fue publicado, marca como spam la mayoría de los SMS ordinarios: un cambio de dominio que ningún umbral puede corregir.
* Hacer fine-tuning de la misma red sobre la partición de entrenamiento de SMS (cerca de 8 minutos en la CPU de un portátil) eleva el F1 de spam de alrededor de 0.23 a alrededor de 0.96.
* Un modelo de TF-IDF + regresión logística bien ajustado queda prácticamente empatado en F1 para SMS en inglés. El transformer ordena mejor los mensajes (mayor ROC-AUC y precisión promedio) y además detecta estafas típicas en español, que el modelo TF-IDF no detecta en absoluto. El fine-tuning solo en inglés sí costó algo de capacidad multilingüe: una estafa de premios en francés que el modelo original detectaba ahora pasa sin ser detectada.

El razonamiento completo, con gráficos, está en el notebook.

## Estructura del repositorio

```
.
├── app/                         Aplicación web en Django
│   ├── manage.py
│   ├── spamlens/                configuración del proyecto y URLs
│   └── classifier/              la app: servicio del modelo, vistas, API, plantillas, archivos estáticos, pruebas
│       ├── services.py          carga del modelo, inferencia, explicaciones a nivel de palabra
│       ├── views.py             página (funciona sin JavaScript) y API JSON
│       ├── model_card.json      umbral y métricas de prueba escritos por el notebook
│       └── management/commands/download_model.py
├── notebook/
│   └── spam_detection.ipynb     proceso de ciencia de datos paso a paso, ejecutado con salidas
├── data/
│   └── sms+spam+collection.zip  SMS Spam Collection (UCI), leído directamente desde el zip
├── model/                       pesos del modelo (ignorados por git, ver model/README.md)
├── requirements.txt
└── README.md
```

## Cómo ejecutarlo

Requisitos: Python 3.12 y cerca de 2 GB de RAM libre. Todo se ejecuta en CPU; no se necesita GPU.

```bash
# 1. Clonar y crear un entorno virtual
git clone https://github.com/Aguado4/spamlens-hacking-ai-Aguado-Cruz-Ramirez.git
cd spamlens-hacking-ai-Aguado-Cruz-Ramirez
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate

# 2. Instalar dependencias (la primera línea instala la versión pequeña de PyTorch solo para CPU)
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Obtener el modelo con fine-tuning (cerca de 240 MB, desde el release de GitHub de este repositorio)
python app/manage.py download_model

# 4. Iniciar la aplicación web
python app/manage.py runserver
```

Abre http://127.0.0.1:8000. La primera consulta tarda unos segundos mientras se carga el modelo; después, cada veredicto toma bastante menos de un segundo.

Si omites el paso 3 la aplicación sigue funcionando: recurre al `Goodmotion/spam-mail-classifier` original desde el Hugging Face Hub y muestra un aviso, pero espera muchas falsas alarmas en SMS (ver los resultados anteriores). Para apuntar la aplicación a cualquier otra carpeta de modelo o id del Hub, define `SPAMLENS_MODEL`.

### Reproducir el notebook

```bash
cd notebook
jupyter notebook spam_detection.ipynb
```

Ejecuta todas las celdas. Descarga el modelo preentrenado desde el Hub (o usa `model/` si colocaste allí una copia local), le hace fine-tuning en CPU (de 10 a 20 minutos en un portátil), escribe los pesos en `model/finetuned/` y actualiza `app/classifier/model_card.json`, de modo que los números del sitio web siempre coinciden con la última ejecución.

### API

```bash
curl -X POST http://127.0.0.1:8000/api/classify/ \
     -H "Content-Type: application/json" \
     -d '{"message": "URGENT! You have won a £1000 prize. Call 09061701461 to claim"}'
```

Devuelve la etiqueta, la probabilidad de spam, el umbral de decisión, los pesos por palabra y las señales de spam detectadas. `GET /api/health/` informa qué modelo está cargado.

### Pruebas

```bash
cd app
python manage.py test
```

Las pruebas reemplazan el transformer por un stub, por lo que se ejecutan en milisegundos y no necesitan los pesos.

## Notas de diseño

* **Sin base de datos.** Los mensajes se clasifican en memoria y nunca se registran ni se almacenan. Es la lección de minimización de datos de la lectura sobre privacidad aplicada al producto.
* **Umbral elegido en validación, nunca en prueba.** Comparamos un umbral por defecto de 0.5, el de F1 máximo y reglas que priorizan la precisión sobre la partición de validación. Las probabilidades del modelo con fine-tuning son muy extremas, así que con solo 97 mensajes spam de validación las reglas que priorizan la precisión llevaron el umbral por encima de 0.99 y sacrificaron recall por ruido; el de F1 máximo fue la opción estable. El conjunto de prueba se usa una sola vez, al final.
* **Explicaciones por oclusión.** Se elimina cada palabra por turno y el cambio en la probabilidad de spam se muestra como un resaltado rojo (hacia spam) o verde (hacia legítimo).
* **Mejora progresiva.** La página funciona con JavaScript desactivado (envío de formulario simple); con JavaScript se actualiza en el lugar, anima el resultado y admite Ctrl+Enter.

## Créditos de datos y modelo

* Almeida, T.A., Gómez Hidalgo, J.M. (2011). *SMS Spam Collection*. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84
* Goodmotion. *spam-mail-classifier*, con fine-tuning a partir de `microsoft/Multilingual-MiniLM-L12-H384`. Apache 2.0. https://huggingface.co/Goodmotion/spam-mail-classifier
* Dwork, C., Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy*. Foundations and Trends in Theoretical Computer Science, 9(3-4), 211-407.
