"""Re-run the full evaluation for an already-trained model.

Usage:
    python evaluate.py                    # test-set metrics, figures, pipeline check
    python evaluate.py --skip-pipeline-check

Reads models/neurowrite_cnn.keras, evaluates it on the MNIST test set and
regenerates everything in outputs/ (metrics.json, classification report,
confusion matrix, example figures). Training curves are re-drawn from
outputs/training_history.json when that file exists.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

from utils.config import HISTORY_PATH, MODEL_PATH, SAMPLES_PATH, SEED, VAL_FRACTION  # noqa: E402
from utils.data import load_mnist, select_sample_pool, split_train_validation, validate_dataset  # noqa: E402
from utils.evaluation import evaluate_and_save, run_pipeline_check, save_training_curves  # noqa: E402
from utils.model import ModelNotFoundError, load_trained_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained NeuroWrite CNN.")
    parser.add_argument("--seed", type=int, default=SEED, help="seed used when the model was trained")
    parser.add_argument("--skip-pipeline-check", action="store_true")
    args = parser.parse_args()

    try:
        model = load_trained_model(MODEL_PATH)
    except ModelNotFoundError as exc:
        raise SystemExit(f"Error: {exc}")

    raw = load_mnist()
    validate_dataset(raw.x_train, raw.y_train, raw.x_test, raw.y_test)

    metrics = evaluate_and_save(model, raw.x_test, raw.y_test)
    print(f"test accuracy : {metrics['test_accuracy']:.4f}")
    print(f"test loss     : {metrics['test_loss']:.4f}")
    print(f"macro P/R/F1  : {metrics['precision_macro']:.4f} / {metrics['recall_macro']:.4f} / {metrics['f1_macro']:.4f}")
    print(f"errors        : {metrics['n_errors']} of {metrics['n_samples']}")

    sx, sy, _ = select_sample_pool(raw.x_test, raw.y_test, per_class=20, seed=args.seed)
    np.savez_compressed(SAMPLES_PATH, images=sx, labels=sy)

    if HISTORY_PATH.exists():
        save_training_curves(json.loads(HISTORY_PATH.read_text()))

    if not args.skip_pipeline_check:
        _, x_val, _, y_val = split_train_validation(raw.x_train, raw.y_train, VAL_FRACTION, args.seed)
        check = run_pipeline_check(model, x_val, y_val, seed=args.seed)
        print(f"direct 28x28 baseline: {check['baseline_direct_28x28_accuracy']:.4f}")
        for name, res in check["conditions"].items():
            print(f"  {name:<20} accuracy {res['accuracy']:.4f}  rejected {res['rejected_images']}")
    print("Artefacts written to outputs/.")


if __name__ == "__main__":
    main()
