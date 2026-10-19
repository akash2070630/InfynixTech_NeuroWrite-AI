# NeuroWrite AI — System Workflow

## A. Training and evaluation workflow (`python train_model.py`)

1. **Load** MNIST (local archives → Keras → checksum-verified mirrors).
2. **Validate** shapes `(60000,28,28)` / `(10000,28,28)`, `uint8` dtype, labels 0–9, all ten classes present.
3. **Analyse** class distribution and save a sample grid and distribution chart.
4. **Split** the 60,000 training images into 54,000 train / 6,000 validation (stratified, seeded).
5. **Prepare** pixels → `float32 / 255`, add channel axis; labels → `int32`.
6. **Build** the CNN and save its summary.
7. **Train** with Adam; EarlyStopping and ReduceLROnPlateau monitor *validation* loss only.
8. **Save** `models/neurowrite_cnn.keras` and the training history / curves.
9. **Evaluate once** on the 10,000-image test set → metrics, report, confusion matrix, example figures.
10. **Export** a class-balanced pool of real test images for the app's Sample Predictions page.
11. **Stress-test** the upload pipeline using re-rendered *validation* digits (never the test set).

## B. Recognition workflow (Recognition Studio)

```
Upload (PNG/JPG/JPEG)
  → validate format, size, integrity
  → grayscale (alpha-aware)
  → detect polarity (Otsu + border brightness) and invert if dark-on-light
  → remove background, stretch contrast, drop noise specks
  → crop to digit, resize (aspect ratio kept) to fit 20×20, centre in 28×28
  → normalise to [0, 1]; validate tensor (1, 28, 28, 1)
  → CNN → softmax probabilities
  → show original, model input, predicted digit, confidence, probability chart
```

## C. Data discipline

| Partition | Size | Used for |
|---|---|---|
| Train | 54,000 | Updating weights |
| Validation | 6,000 | Early stopping, learning-rate schedule, pipeline stress test |
| Test | 10,000 | Final evaluation only |
