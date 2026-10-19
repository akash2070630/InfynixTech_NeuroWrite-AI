"""Train the NeuroWrite CNN on MNIST and write every artefact to models/ and outputs/.

Usage:
    python train_model.py                       # default settings
    python train_model.py --epochs 5            # quick run
    python train_model.py --deterministic       # slower, bit-for-bit reproducible on one machine

Data discipline
    * the 60,000 training images are split 90 / 10 into train / validation
      (stratified, seeded);
    * early stopping and learning-rate reduction watch the VALIDATION loss only;
    * the 10,000-image test set is touched exactly once, after training, for the
      final evaluation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

from utils import plots  # noqa: E402
from utils.config import (  # noqa: E402
    BATCH_SIZE, EARLY_STOP_PATIENCE, FIGURES_DIR, LEARNING_RATE, LR_REDUCE_FACTOR,
    LR_REDUCE_PATIENCE, MAX_EPOCHS, MIN_LEARNING_RATE, MODEL_PATH, MODELS_DIR, OUTPUTS_DIR,
    RUN_INFO_PATH, SAMPLES_PATH, SEED, SUMMARY_PATH, VAL_FRACTION,
)
from utils.data import (  # noqa: E402
    class_distribution, load_mnist, normalize_images, prepare_labels, select_sample_pool,
    split_train_validation, validate_dataset,
)
from utils.evaluation import evaluate_and_save, run_pipeline_check, save_training_curves  # noqa: E402
from utils.model import build_cnn  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the NeuroWrite AI digit-recognition CNN.")
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="maximum epochs (early stopping may end sooner)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-augmentation", action="store_true", help="disable random rotation/shift/zoom")
    p.add_argument("--deterministic", action="store_true", help="enable TensorFlow op determinism (slower)")
    p.add_argument("--skip-pipeline-check", action="store_true", help="skip the upload-pipeline stress test")
    return p.parse_args()


def set_seeds(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed, args.deterministic)
    for folder in (MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    print("== 1/5  Loading and validating MNIST ==")
    raw = load_mnist()
    validate_dataset(raw.x_train, raw.y_train, raw.x_test, raw.y_test)
    train_counts, test_counts = class_distribution(raw.y_train), class_distribution(raw.y_test)
    print(f"source        : {raw.source}")
    print(f"train images  : {raw.x_train.shape}  labels: {raw.y_train.shape}")
    print(f"test images   : {raw.x_test.shape}  labels: {raw.y_test.shape}")
    print(f"pixel range   : {raw.x_train.min()}-{raw.x_train.max()} (uint8)")
    print(f"train classes : {train_counts}")
    print(f"test classes  : {test_counts}")
    plots.save_figure(plots.plot_sample_grid(raw.x_train, raw.y_train, seed=args.seed), FIGURES_DIR / "sample_digits.png")
    plots.save_figure(plots.plot_class_distribution(train_counts, test_counts), FIGURES_DIR / "class_distribution.png")

    x_tr_raw, x_val_raw, y_tr_raw, y_val_raw = split_train_validation(
        raw.x_train, raw.y_train, VAL_FRACTION, args.seed)
    x_tr, x_val = normalize_images(x_tr_raw), normalize_images(x_val_raw)
    y_tr, y_val = prepare_labels(y_tr_raw), prepare_labels(y_val_raw)
    print(f"split         : {len(x_tr):,} train / {len(x_val):,} validation / {len(raw.x_test):,} test (held out)")

    # ----------------------------------------------------------------- model
    print("\n== 2/5  Building the CNN ==")
    model = build_cnn(augment=not args.no_augmentation, learning_rate=args.learning_rate)
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    SUMMARY_PATH.write_text("\n".join(lines))
    print("\n".join(lines))

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=LR_REDUCE_FACTOR,
                                          patience=LR_REDUCE_PATIENCE, min_lr=MIN_LEARNING_RATE, verbose=1),
    ]

    # ----------------------------------------------------------------- train
    print("\n== 3/5  Training ==")
    started = time.time()
    history = model.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs=args.epochs,
                        batch_size=args.batch_size, callbacks=callbacks, verbose=2)
    train_seconds = time.time() - started
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    best_epoch = int(np.argmin(hist["val_loss"])) + 1
    model.save(MODEL_PATH)
    save_training_curves(hist)
    print(f"model saved   : {MODEL_PATH}  (best epoch {best_epoch}, {train_seconds / 60:.1f} min)")

    # -------------------------------------------------------------- evaluate
    print("\n== 4/5  Evaluating on the untouched test set ==")
    run_info = {
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_source": raw.source,
        "train_samples": int(len(x_tr)), "validation_samples": int(len(x_val)),
        "test_samples": int(len(raw.x_test)),
        "epochs_requested": args.epochs, "epochs_run": len(hist["loss"]), "best_epoch": best_epoch,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate, "seed": args.seed,
        "augmentation": not args.no_augmentation, "deterministic": args.deterministic,
        "parameters": int(model.count_params()),
        "training_seconds": round(train_seconds, 1),
        "tensorflow_version": tf.__version__, "keras_version": keras.__version__,
        "final_learning_rate": float(hist.get("learning_rate", [args.learning_rate])[-1]),
    }
    metrics = evaluate_and_save(model, raw.x_test, raw.y_test)
    RUN_INFO_PATH.write_text(json.dumps(run_info, indent=2))
    print(f"test accuracy : {metrics['test_accuracy']:.4f}   test loss: {metrics['test_loss']:.4f}")
    print(f"macro P/R/F1  : {metrics['precision_macro']:.4f} / {metrics['recall_macro']:.4f} / {metrics['f1_macro']:.4f}")
    print(f"errors        : {metrics['n_errors']} of {metrics['n_samples']}")

    # --------------------------------------------------- app assets and checks
    print("\n== 5/5  Saving demo samples and running the pipeline check ==")
    sx, sy, _ = select_sample_pool(raw.x_test, raw.y_test, per_class=20, seed=args.seed)
    np.savez_compressed(SAMPLES_PATH, images=sx, labels=sy)
    if not args.skip_pipeline_check:
        check = run_pipeline_check(model, x_val_raw, y_val_raw, seed=args.seed)
        print(f"direct 28x28 baseline (validation subset): {check['baseline_direct_28x28_accuracy']:.4f}")
        for name, res in check["conditions"].items():
            print(f"  {name:<20} accuracy {res['accuracy']:.4f}  rejected {res['rejected_images']}")

    print("\nDone. Launch the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
