# NeuroWrite AI — Intelligent Handwritten Digit Recognition System

**Infynix Tech · Machine Learning Internship · Task 3 — Handwritten Character Recognition**

NeuroWrite AI recognises handwritten digits (0–9) with a Convolutional Neural Network (CNN) trained on the
MNIST dataset. It contains a reproducible training and evaluation pipeline, a shared image-preprocessing
module for real-world uploads, automated tests, and a Streamlit dashboard for exploring the model.

> **No results are hardcoded.** Every metric, chart and sample shown by the app is produced by running
> `train_model.py` (or `evaluate.py`) on your machine and read from the `outputs/` folder.

---

## 1. Objectives

1. Train a CNN on the genuine MNIST training set to classify digits 0–9.
2. Evaluate it honestly on the untouched MNIST test set (accuracy, loss, precision, recall, F1, confusion matrix).
3. Let users test their own handwritten digit images, with robust preprocessing and clear error messages.
4. Keep the project clean, testable and reproducible.

## 2. Features

| Area | What you get |
|---|---|
| Training | Stratified 90/10 train/validation split, seeded runs, early stopping, learning-rate reduction, light augmentation |
| Evaluation | Test accuracy & loss, macro/weighted precision/recall/F1, classification report, confusion matrix, per-class chart, correct and incorrect examples, training curves |
| Preprocessing | Format/size validation, grayscale, automatic black-on-white / white-on-black detection, transparent-PNG support, noise removal, digit cropping, aspect-ratio-preserving resize, centre-of-mass centring, tensor validation |
| App | Overview dashboard, Recognition Studio (upload), Sample Predictions (real MNIST), Model Performance, About & Methodology |
| Safety | Missing-model handling, corrupt-file handling, no retraining on app start, no secrets, no absolute paths, no external API |
| Tests | Preprocessing, data validation, model shapes/outputs, saved-model checks, and app smoke tests |

## 3. Architecture and workflow

```
MNIST (60k train / 10k test)
   │  validate shapes, dtypes, labels
   ├── 90 % train ──┐
   ├── 10 % validation (early stopping, LR schedule)        test set is held out
   ▼                                                          until the final evaluation
 normalise /255 → (N,28,28,1) ──► CNN ──► softmax(10)
                                      │
                    models/neurowrite_cnn.keras  +  outputs/ (metrics, figures, samples)
                                      │
 upload ─► utils/preprocessing.py ────┴──► app.py (Streamlit)
```

CNN (`utils/model.py`): `Input 28×28×1 → [random rotate/shift/zoom, training only] → Conv32(3×3,ReLU) →
MaxPool → Conv64(3×3,ReLU) → MaxPool → Dropout(0.25) → Flatten → Dense128(ReLU) → Dropout(0.5) → Dense10(softmax)`,
about 422 thousand parameters, Adam optimiser, sparse categorical cross-entropy loss.

## 4. Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) — 70,000 grayscale 28×28 handwritten-digit images
(60,000 training, 10,000 test). `train_model.py` loads it via Keras. If that download is blocked, it falls
back to public mirrors and **verifies the official MD5 checksums**. For offline use, place the four official
`*-ubyte.gz` files in `data/` and they will be used directly.

## 5. Project structure

```
NeuroWrite-AI/
├── app.py                 Streamlit application
├── train_model.py         Train, save model, evaluate, write artefacts
├── evaluate.py            Re-run evaluation on a saved model
├── requirements.txt
├── pytest.ini
├── .streamlit/config.toml Light navy/blue theme
├── utils/
│   ├── config.py          Paths and hyper-parameters (no absolute paths)
│   ├── data.py            MNIST loading, validation, split, normalisation
│   ├── model.py           CNN, loading, validated inference
│   ├── preprocessing.py   The single upload-preprocessing implementation
│   ├── inference.py       preprocess + predict (used by the app and the tests)
│   ├── evaluation.py      Metrics, figures, pipeline stress test
│   └── plots.py           Matplotlib/Seaborn figures
├── tests/                 pytest suite
├── models/                neurowrite_cnn.keras (created by training)
├── outputs/               metrics.json, figures/, reports (created by training)
├── data/                  MNIST archives (downloaded automatically)
└── docs/                  Abstract, workflow, requirement mapping, LinkedIn post, demo script, viva Q&A
```

## 6. Setup

Python 3.10–3.12 is recommended (check which versions your TensorFlow release supports).

```bash
python -m venv .venv

# Windows (PowerShell / cmd)
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 7. Train the model

```bash
python train_model.py
```

This downloads MNIST (first run only), trains the CNN (up to 20 epochs; early stopping usually ends sooner),
saves `models/neurowrite_cnn.keras`, evaluates on the test set, and writes everything into `outputs/`.
Expect a few minutes on a laptop CPU. Useful options:

```bash
python train_model.py --epochs 5            # quick run
python train_model.py --no-augmentation     # disable rotate/shift/zoom
python train_model.py --deterministic       # slower, reproducible op order on one machine
python evaluate.py                          # re-run evaluation of an existing model
```

Seeds are fixed (default 42) for the data split and weight initialisation. Exact numbers can still differ
slightly between machines, TensorFlow versions and CPU/GPU hardware; use `--deterministic` for the closest match.

## 8. Launch the app

```bash
streamlit run app.py
```

If the model has not been trained, the app shows instructions instead of failing; it never trains on start-up.

## 9. Test your own digit

1. Open **Recognition Studio**.
2. Upload a PNG/JPG/JPEG with **one** digit on a plain background (dark pen on white paper works well).
3. Check the **Model input (28×28)** preview: it should show a clear white digit on black.
4. Read the predicted digit, its confidence and the probability chart. Confidence is the model's softmax
   output, not a guarantee. If it looks wrong, switch the *Ink / background* option or use a thicker pen.

## 10. Run the tests

```bash
python -m pytest
```

Tests that need the trained model are skipped automatically until you have run `python train_model.py`.

## 11. Internship Task 3 requirement mapping

| Requirement | Implementation |
|---|---|
| Image processing | `utils/preprocessing.py` (grayscale, polarity, denoise, crop, resize, centre, normalise); shown in Recognition Studio |
| MNIST dataset | `utils/data.py` loads, validates and splits the genuine 60k/10k dataset; `train_model.py` |
| Deep learning | TensorFlow/Keras training with callbacks in `train_model.py` |
| CNN architecture | `build_cnn()` in `utils/model.py`; explained in the app and in this README |
| Image classification | `utils/inference.py` + Streamlit pages; ten-class softmax output |
| Model training and evaluation | `train_model.py`, `evaluate.py`, `utils/evaluation.py`; results in `outputs/` |

A longer version is in [`docs/REQUIREMENT_MAPPING.md`](docs/REQUIREMENT_MAPPING.md).

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `No trained model found` | Run `python train_model.py` first. |
| MNIST download fails | The loader tries Keras, then mirrors. If your network blocks both, download the four MNIST `*-ubyte.gz` files manually into `data/`. |
| `ModuleNotFoundError: utils` | Run commands from the project root (the folder containing `app.py`). |
| Model fails to load after upgrading TensorFlow/Keras | Retrain with the same environment you use for the app. |
| Prediction is wrong on a photo | Use even lighting, a plain background, a thick dark pen, one digit per image, and check the 28×28 preview. |
| TensorFlow will not install | Use a supported Python version (3.10–3.12) in a fresh virtual environment. |
| `streamlit` not found | Activate the virtual environment, then `pip install -r requirements.txt`. |

## 13. Limitations and future work

- Recognises one isolated digit; letters, words and multi-digit numbers are out of scope.
- MNIST digits are clean, centred and US-style; accuracy on your own handwriting or on photos can be lower than the MNIST test accuracy.
- Softmax confidence can be high for wrong answers; the app flags low-certainty cases but cannot detect every error.
- Future work: a drawing canvas, EMNIST letters, segmenting multi-digit numbers, probability calibration, out-of-distribution detection, ensembles, and a deeper network with batch normalisation.

## 14. Verified status

See the "Verification" section of the delivery message for exactly what was executed when this project was built.
What you will measure yourself is whatever `outputs/metrics.json` contains after you run training.
