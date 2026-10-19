# Infynix Tech — Task 3 Requirement Mapping

| # | Requirement | How NeuroWrite AI satisfies it | Where |
|---|---|---|---|
| 1 | **Image processing** | Uploaded photos/scans are validated, converted to grayscale (transparent PNGs use the alpha channel), polarity-detected (black-on-white vs white-on-black via Otsu threshold and border brightness), background-subtracted, contrast-stretched, denoised (small connected components removed), cropped to the digit, resized to fit a 20×20 box **with the aspect ratio preserved**, centred by centre of mass in a 28×28 frame, and normalised to [0, 1]. The tensor is validated (shape, dtype, finite, range) before inference. | `utils/preprocessing.py`; Recognition Studio page |
| 2 | **MNIST dataset usage** | The genuine 60,000 / 10,000 MNIST split is loaded (Keras, or checksum-verified official files), validated (shape, dtype, label range, class coverage), analysed (class counts, sample grid) and split 90/10 stratified into train/validation. The test set is only used for the final evaluation. | `utils/data.py`, `train_model.py` |
| 3 | **Deep learning** | A neural network is trained end-to-end with Keras/TensorFlow: Adam optimiser, cross-entropy loss, mini-batches, dropout, augmentation, early stopping, learning-rate scheduling, seeded runs. | `train_model.py` |
| 4 | **CNN architecture** | Conv(32) → MaxPool → Conv(64) → MaxPool → Dropout → Flatten → Dense(128) → Dropout → Dense(10, softmax). Each component's role is explained in the app (About & Methodology) and README. | `utils/model.py` → `build_cnn()` |
| 5 | **Image classification** | The ten-class softmax output gives the predicted digit, confidence and full probability distribution for uploaded images and dataset samples. | `utils/inference.py`, Recognition Studio, Sample Predictions |
| 6 | **Model training and evaluation** | Training history (accuracy/loss curves), test accuracy and loss, macro/weighted precision/recall/F1, per-class report, confusion matrix, correct/incorrect examples, and a preprocessing stress test. All saved to `outputs/` and displayed in the app. | `train_model.py`, `evaluate.py`, `utils/evaluation.py`, Model Performance page |

## Evidence the requirements are met

- `outputs/metrics.json`, `outputs/classification_report.txt`, `outputs/figures/*.png` are produced by actually running the code.
- `python -m pytest` verifies preprocessing, data validation, model shapes/outputs, saved-metric consistency and app rendering.
