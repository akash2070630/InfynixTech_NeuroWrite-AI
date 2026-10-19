"""NeuroWrite AI - handwritten digit recognition dashboard.

Run with:  streamlit run app.py
The app only *loads* the trained model (models/neurowrite_cnn.keras); it never
trains on start-up. Every number shown on the performance pages is read from
files written by train_model.py / evaluate.py.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from utils.config import (
    ALLOWED_EXTENSIONS, CLASS_NAMES, FIGURES_DIR, METRICS_PATH, MODEL_PATH, PIPELINE_CHECK_PATH,
    RUN_INFO_PATH, SAMPLES_PATH,
)
from utils.data import normalize_images
from utils.inference import LOW_CONFIDENCE, LOW_MARGIN, recognise
from utils.model import (
    ModelInputError, build_cnn, describe_architecture, load_trained_model, model_exists,
    predict_probabilities,
)
from utils.plots import plot_probabilities
from utils.preprocessing import (
    POLARITY_AUTO, POLARITY_DARK_ON_LIGHT, POLARITY_LIGHT_ON_DARK, ImageValidationError,
    PreprocessingError, load_image,
)

st.set_page_config(page_title="NeuroWrite AI", page_icon="✍️", layout="wide",
                   initial_sidebar_state="expanded")

PAGES = ["Overview", "Recognition Studio", "Sample Predictions", "Model Performance", "About & Methodology"]

CSS = """
<style>
:root { --navy:#0B2545; --blue:#1D4ED8; --sky:#60A5FA; --pale:#DBEAFE; --slate:#475569;
        --line:#E2E8F0; --good:#15803D; --bad:#B91C1C; --warn:#B45309; }
.block-container { padding-top: 2rem; max-width: 1180px; }
h1, h2, h3, h4 { color: var(--navy); letter-spacing: -0.01em; }
.nw-hero { background:#FFFFFF; border:1px solid var(--line); border-left:5px solid var(--blue);
           border-radius:12px; padding:1.4rem 1.6rem; margin-bottom:1.2rem; }
.nw-hero h1 { margin:0 0 .25rem 0; font-size:2rem; }
.nw-hero p { margin:0; color:var(--slate); font-size:1.02rem; }
.nw-eyebrow { color:var(--blue); font-weight:700; font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; }
.nw-card { background:#FFFFFF; border:1px solid var(--line); border-radius:12px; padding:1rem 1.15rem; height:100%; }
.nw-label { color:var(--slate); font-size:.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.05em; }
.nw-value { color:var(--navy); font-size:1.7rem; font-weight:700; line-height:1.25; }
.nw-help { color:var(--slate); font-size:.8rem; margin-top:.15rem; }
.nw-pills { display:flex; flex-wrap:wrap; gap:.5rem; margin:.4rem 0 1rem 0; }
.nw-pill { width:2.6rem; height:2.6rem; border-radius:10px; background:var(--pale); color:var(--navy);
           font-weight:700; font-size:1.25rem; display:flex; align-items:center; justify-content:center;
           border:1px solid #BFDBFE; }
.nw-step { background:#FFFFFF; border:1px solid var(--line); border-radius:12px; padding:.9rem 1rem; height:100%; }
.nw-step b { color:var(--blue); }
.nw-result { background:#FFFFFF; border:1px solid var(--line); border-radius:12px; padding:1rem 1.2rem; text-align:center; }
.nw-digit { font-size:4.6rem; font-weight:800; color:var(--blue); line-height:1; }
.nw-conf { font-size:1.15rem; color:var(--navy); font-weight:600; }
.nw-note { color:var(--slate); font-size:.82rem; }
.nw-badge { display:inline-block; padding:.15rem .6rem; border-radius:999px; font-size:.78rem; font-weight:600; }
.nw-ok { background:#DCFCE7; color:var(--good); } .nw-warn { background:#FEF3C7; color:var(--warn); }
.nw-bad { background:#FEE2E2; color:var(--bad); }
.nw-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(112px, 1fr)); gap:.75rem; }
.nw-sample { background:#FFFFFF; border:1px solid var(--line); border-radius:10px; padding:.5rem; text-align:center;
             font-size:.82rem; color:var(--navy); }
.nw-sample img { width:100%; image-rendering:pixelated; border-radius:6px; }
.nw-sample .ok { color:var(--good); font-weight:600; } .nw-sample .bad { color:var(--bad); font-weight:600; }
[data-testid="stSidebar"] { border-right:1px solid var(--line); }
@media (max-width: 640px) { .nw-hero h1 { font-size:1.5rem; } .nw-digit { font-size:3.4rem; } .nw-value { font-size:1.4rem; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached loaders (keyed on file modification time, so retraining refreshes them)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading trained model...")
def _cached_model(mtime: float):
    return load_trained_model(MODEL_PATH)


def get_model():
    """Return the trained model, or None (with a visible message) if unavailable."""
    if not model_exists(MODEL_PATH):
        return None
    try:
        return _cached_model(MODEL_PATH.stat().st_mtime)
    except Exception as exc:   # corrupted file / incompatible Keras version: tell the user
        st.error(f"The model file exists but could not be loaded: {type(exc).__name__}: {exc}\n\n"
                 "Re-train it with `python train_model.py` using the same environment as the app.")
        return None


@st.cache_data(show_spinner=False)
def _cached_json(path: str, mtime: float):
    return json.loads(Path(path).read_text())


def read_json(path: Path):
    if not Path(path).exists():
        return None
    return _cached_json(str(path), Path(path).stat().st_mtime)


@st.cache_data(show_spinner=False)
def _cached_samples(mtime: float):
    with np.load(SAMPLES_PATH) as data:
        return data["images"], data["labels"]


def read_samples():
    if not SAMPLES_PATH.exists():
        return None
    return _cached_samples(SAMPLES_PATH.stat().st_mtime)


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def hero(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="nw-hero"><div class="nw-eyebrow">{eyebrow}</div>'
                f'<h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(f'<div class="nw-card"><div class="nw-label">{label}</div>'
                f'<div class="nw-value">{value}</div><div class="nw-help">{help_text}</div></div>',
                unsafe_allow_html=True)


def to_data_uri(arr: np.ndarray, size: int = 112) -> str:
    img = Image.fromarray(arr.astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def show_figure(name: str, caption: str = "") -> None:
    path = FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption or None)
    else:
        st.info(f"`{name}` has not been generated yet. Run `python train_model.py`.")


def training_instructions() -> None:
    st.warning("**No trained model found yet.** Train it once, then refresh this page:")
    st.code("python train_model.py", language="bash")
    st.caption("Training downloads MNIST (about 11 MB) on first use and takes a few minutes on a CPU. "
               "The app never trains automatically.")


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    hero("Machine Learning · Computer Vision · Task 3",
         "NeuroWrite AI",
         "An intelligent handwritten digit recognition system built on a Convolutional Neural Network "
         "trained on the MNIST dataset.")
    model = get_model()
    metrics = read_json(METRICS_PATH)
    info = read_json(RUN_INFO_PATH)

    cols = st.columns(4)
    with cols[0]:
        if model is not None:
            metric_card("Model status", "Trained", f"Loaded from <code>{MODEL_PATH.parent.name}/{MODEL_PATH.name}</code>")
        else:
            metric_card("Model status", "Not trained", "Run <code>python train_model.py</code>")
    with cols[1]:
        metric_card("Test accuracy", pct(metrics["test_accuracy"]) if metrics else "n/a",
                    f"on {metrics['n_samples']:,} unseen MNIST test images" if metrics else "Available after training")
    with cols[2]:
        metric_card("Macro F1-score", f"{metrics['f1_macro']:.4f}" if metrics else "n/a",
                    "Average over the 10 digit classes" if metrics else "Available after training")
    with cols[3]:
        metric_card("Parameters", f"{info['parameters']:,}" if info else "n/a",
                    f"Trained {info['trained_at'][:10]}, {info['epochs_run']} epochs" if info else "Available after training")
    if model is None:
        st.write("")
        training_instructions()

    st.write("")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("What is handwritten digit recognition?")
        st.write(
            "Handwritten digit recognition is an image-classification problem: given a picture of a single "
            "hand-drawn digit, decide which of the ten digits (0-9) it shows. Handwriting varies enormously "
            "between people, so instead of hand-writing rules, a Convolutional Neural Network learns visual "
            "patterns (strokes, loops, corners) directly from 60,000 labelled examples. It is the classic "
            "benchmark for introducing deep learning and has real uses such as reading postal codes, "
            "cheques and forms.")
    with right:
        st.subheader("Supported classes")
        pills = "".join(f'<div class="nw-pill">{c}</div>' for c in CLASS_NAMES)
        st.markdown(f'<div class="nw-pills">{pills}</div>', unsafe_allow_html=True)
        st.caption("Ten digit classes, 0 through 9. The model does not recognise letters or multi-digit numbers.")

    st.subheader("How NeuroWrite AI works")
    steps = [
        ("1 · Upload", "You provide a PNG or JPEG of one handwritten digit."),
        ("2 · Preprocess", "Grayscale, polarity detection, digit cropping, 28×28 centring, normalisation."),
        ("3 · Classify", "The CNN turns the image into ten softmax probabilities."),
        ("4 · Explain", "The app shows the prediction, its confidence and the full distribution."),
    ]
    for col, (title, text) in zip(st.columns(4), steps):
        with col:
            st.markdown(f'<div class="nw-step"><b>{title}</b><br>{text}</div>', unsafe_allow_html=True)

    if metrics:
        st.caption(f"Metrics above were measured by evaluating the saved model on the MNIST test set "
                   f"({metrics['generated_at'].replace('T', ' ')}). See the Model Performance page for details.")


def page_studio() -> None:
    hero("Recognition Studio", "Test a handwritten digit",
         "Upload a picture of one digit and see how the model reads it.")
    model = get_model()
    if model is None:
        training_instructions()
        return

    left, right = st.columns([3, 2])
    with left:
        uploaded = st.file_uploader(
            "Upload an image (PNG, JPG or JPEG, up to 10 MB)", type=list(ALLOWED_EXTENSIONS),
            help="Use a single digit, clearly written, with a plain background.")
    with right:
        polarity_label = st.selectbox(
            "Ink / background", ["Detect automatically", "Dark digit on light background",
                                 "Light digit on dark background"],
            help="Automatic detection compares the image border with the Otsu threshold. "
                 "Override it if the processed preview looks inverted.")
        auto_crop = st.checkbox("Crop and centre the digit automatically", value=True,
                                help="Finds the digit, scales it to 20×20 and centres it inside 28×28, like MNIST.")
    polarity = {"Detect automatically": POLARITY_AUTO,
                "Dark digit on light background": POLARITY_DARK_ON_LIGHT,
                "Light digit on dark background": POLARITY_LIGHT_ON_DARK}[polarity_label]

    with st.expander("When can automatic image processing fail?"):
        st.markdown(
            "- **Several digits or text in one image**: the pipeline expects exactly one digit.\n"
            "- **Uneven lighting, shadows or textured paper**: shadows can be mistaken for ink.\n"
            "- **Very thin or faint pen strokes**, or strokes that nearly touch the image border.\n"
            "- **Coloured backgrounds or grids/lines** that are as dark as the ink.\n"
            "- **Styles far from MNIST** (for example a crossed 7 or a serif 1) can be misread.\n\n"
            "If the *Model input* preview does not clearly show a white digit on black, change the "
            "ink/background option, retake the photo with even light, or use a thicker pen.")

    if uploaded is None:
        st.info("Upload an image to begin. You can also try the **Sample Predictions** page for real MNIST digits.")
        return

    data = uploaded.getvalue()
    try:
        original = load_image(data)
        result = recognise(model, data, polarity=polarity, auto_crop=auto_crop)
    except (ImageValidationError, PreprocessingError) as exc:
        st.error(f"**This image can't be used.** {exc}")
        return
    except ModelInputError as exc:
        st.error(f"Internal input validation failed: {exc}")
        return

    pred, proc = result.prediction, result.processed
    st.subheader("Result")
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        st.markdown("**Original image**")
        st.image(original.convert("RGBA") if original.mode in ("P", "LA") else original, width=240)
        st.caption(f"{proc.info['original_size'][0]}×{proc.info['original_size'][1]} px")
    with c2:
        st.markdown("**Model input (28×28)**")
        st.image(Image.fromarray(proc.preview).resize((240, 240), Image.Resampling.NEAREST), width=240)
        st.caption("Exactly what the CNN receives.")
    with c3:
        badge = ('<span class="nw-badge nw-warn">Low certainty - double-check</span>' if result.is_uncertain
                 else '<span class="nw-badge nw-ok">Model is fairly sure</span>')
        st.markdown(
            f'<div class="nw-result"><div class="nw-label">Predicted digit</div>'
            f'<div class="nw-digit">{pred.digit}</div>'
            f'<div class="nw-conf">Confidence {pct(pred.confidence, 1)}</div>{badge}'
            f'<div class="nw-note" style="margin-top:.6rem">Confidence is the model\'s softmax output. '
            f'It is <b>not</b> a guarantee that the answer is correct.</div></div>', unsafe_allow_html=True)

    if result.is_uncertain:
        st.warning(f"The top probability is {pct(pred.confidence, 1)} and the gap to the runner-up is "
                   f"{pct(pred.margin, 1)} (thresholds: {LOW_CONFIDENCE:.0%} / {LOW_MARGIN:.0%}). "
                   "Check the Model input preview, try another polarity setting, or use a clearer image.")

    st.subheader("Probability distribution")
    chart_col, table_col = st.columns([3, 2])
    with chart_col:
        fig = plot_probabilities(pred.probabilities, pred.digit)
        st.pyplot(fig)
    with table_col:
        order = np.argsort(pred.probabilities)[::-1]
        df = pd.DataFrame({"Digit": [CLASS_NAMES[i] for i in order],
                           "Probability": [float(pred.probabilities[i]) * 100 for i in order]})
        st.dataframe(df, hide_index=True, column_config={
            "Probability": st.column_config.ProgressColumn("Probability", min_value=0, max_value=100, format="%.2f%%")})

    with st.expander("Preprocessing details"):
        info = proc.info
        st.json({
            "polarity detected": info["polarity_detected"],
            "polarity applied": info["polarity_used"],
            "transparent PNG alpha used as ink": info["alpha_channel_used"],
            "digit crop box (x0, y0, x1, y1)": info["crop_box_xyxy"],
            "Otsu threshold": info["otsu_threshold"],
            "model-input shape": list(proc.tensor.shape),
            "model-input value range": [float(proc.tensor.min()), float(proc.tensor.max())],
        })


def page_samples() -> None:
    hero("Sample Predictions", "Try the model on real MNIST digits",
         "These images come from the MNIST test set, so the true label is known.")
    model = get_model()
    if model is None:
        training_instructions()
        return
    samples = read_samples()
    if samples is None:
        st.warning("The sample file `outputs/sample_images.npz` is missing. Run `python train_model.py` "
                   "(or `python evaluate.py`) to create it.")
        return
    images, labels = samples

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        digit_filter = st.selectbox("Digit to show", ["All digits"] + CLASS_NAMES)
    with c2:
        count = st.slider("Number of images", 6, 36, 18, step=6)
    with c3:
        st.write("")
        if st.button("Show a different random selection"):
            st.session_state["sample_seed"] = st.session_state.get("sample_seed", 0) + 1

    rng = np.random.default_rng(st.session_state.get("sample_seed", 0))
    pool = np.arange(len(labels)) if digit_filter == "All digits" else np.flatnonzero(labels == int(digit_filter))
    chosen = rng.choice(pool, size=min(count, len(pool)), replace=False)
    probs = predict_probabilities(model, normalize_images(images[chosen]))
    preds = probs.argmax(axis=1)
    n_correct = int((preds == labels[chosen]).sum())

    cards = []
    for k, idx in enumerate(chosen):
        ok = preds[k] == labels[idx]
        cards.append(
            f'<div class="nw-sample"><img src="{to_data_uri(images[idx])}" alt="digit sample">'
            f'True label: <b>{labels[idx]}</b><br>'
            f'<span class="{"ok" if ok else "bad"}">Predicted {preds[k]} · {probs[k].max() * 100:.1f}%</span></div>')
    st.markdown(f'<div class="nw-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
    st.caption(f"{n_correct} of {len(chosen)} displayed samples predicted correctly. This is a small random "
               "selection for demonstration, not a performance estimate; see Model Performance for the "
               "full 10,000-image evaluation.")

    st.subheader("Inspect one sample")
    pick = st.selectbox("Sample", range(len(chosen)),
                        format_func=lambda i: f"#{i + 1} · true label {labels[chosen[i]]}")
    a, b = st.columns([1, 2])
    with a:
        st.image(Image.fromarray(images[chosen[pick]]).resize((224, 224), Image.Resampling.NEAREST), width=224)
        st.markdown(f"**True:** {labels[chosen[pick]]}  \n**Predicted:** {preds[pick]}  \n"
                    f"**Confidence:** {probs[pick].max() * 100:.1f}%")
    with b:
        st.pyplot(plot_probabilities(probs[pick], int(preds[pick])))


def page_performance() -> None:
    hero("Model Performance", "Measured results",
         "Everything on this page is read from files written by the training and evaluation scripts.")
    metrics = read_json(METRICS_PATH)
    if metrics is None:
        st.warning("No evaluation results yet. Train and evaluate the model first:")
        st.code("python train_model.py", language="bash")
        return
    info = read_json(RUN_INFO_PATH)

    cols = st.columns(5)
    cards = [("Test accuracy", pct(metrics["test_accuracy"]), f"{metrics['n_errors']} errors in {metrics['n_samples']:,}"),
             ("Test loss", f"{metrics['test_loss']:.4f}", "Cross-entropy"),
             ("Precision (macro)", f"{metrics['precision_macro']:.4f}", "Mean over classes"),
             ("Recall (macro)", f"{metrics['recall_macro']:.4f}", "Mean over classes"),
             ("F1-score (macro)", f"{metrics['f1_macro']:.4f}", "Mean over classes")]
    for col, (label, value, helper) in zip(cols, cards):
        with col:
            metric_card(label, value, helper)
    st.caption(f"Evaluated {metrics['generated_at'].replace('T', ' ')} on the untouched MNIST test set.")

    tabs = st.tabs(["Report", "Confusion matrix", "Training curves", "Per-class", "Examples", "Dataset", "Pipeline check"])
    with tabs[0]:
        rows = [{"Class": c, **{k.title(): v for k, v in metrics["per_class"][c].items()}} for c in CLASS_NAMES]
        rows.append({"Class": "accuracy", "Precision": None, "Recall": None,
                     "F1-Score": metrics["test_accuracy"], "Support": metrics["n_samples"]})
        rows.append({"Class": "macro avg", "Precision": metrics["precision_macro"], "Recall": metrics["recall_macro"],
                     "F1-Score": metrics["f1_macro"], "Support": metrics["n_samples"]})
        rows.append({"Class": "weighted avg", "Precision": metrics["precision_weighted"],
                     "Recall": metrics["recall_weighted"], "F1-Score": metrics["f1_weighted"],
                     "Support": metrics["n_samples"]})
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, column_config={
            "Precision": st.column_config.NumberColumn(format="%.4f"),
            "Recall": st.column_config.NumberColumn(format="%.4f"),
            "F1-Score": st.column_config.NumberColumn(format="%.4f"),
            "Support": st.column_config.NumberColumn(format="%d")})
        with st.expander("How to read these metrics"):
            st.markdown(
                "- **Accuracy**: share of test images classified correctly.\n"
                "- **Loss**: cross-entropy; penalises confident wrong answers; lower is better.\n"
                "- **Precision (digit d)**: of the images predicted as d, how many really are d.\n"
                "- **Recall (digit d)**: of the images that really are d, how many were found.\n"
                "- **F1**: harmonic mean of precision and recall.\n"
                "- **Macro average**: plain mean over the 10 classes (every digit counts equally). "
                "**Weighted average**: mean weighted by how many test images each digit has.")
    with tabs[1]:
        show_figure("confusion_matrix.png", "Rows = true digit, columns = predicted digit. Off-diagonal cells are mistakes.")
    with tabs[2]:
        show_figure("training_curves.png", "Training vs. validation accuracy and loss for each epoch.")
        if info:
            st.caption(f"{info['epochs_run']} epochs run (best validation loss at epoch {info['best_epoch']}), "
                       f"batch size {info['batch_size']}, seed {info['seed']}, augmentation "
                       f"{'on' if info['augmentation'] else 'off'}. Training accuracy can sit below validation "
                       "accuracy early on because dropout and augmentation make training harder than validation.")
    with tabs[3]:
        show_figure("per_class_metrics.png")
    with tabs[4]:
        show_figure("correct_examples.png")
        show_figure("misclassified_examples.png")
    with tabs[5]:
        show_figure("sample_digits.png")
        show_figure("class_distribution.png")
    with tabs[6]:
        check = read_json(PIPELINE_CHECK_PATH)
        if check is None:
            st.info("Pipeline check results are not available. Run `python evaluate.py` to generate them.")
        else:
            st.write("Real **validation-set** digits were re-rendered as uploads (enlarged, shifted, recoloured, "
                     "noisy, JPEG-compressed, transparent) and sent through the same preprocessing pipeline used "
                     "by the Recognition Studio.")
            rows = [{"Condition": v["description"], "Accuracy": v["accuracy"], "Rejected images": v["rejected_images"]}
                    for v in check["conditions"].values()]
            rows.insert(0, {"Condition": "Baseline: original 28×28 images, no pipeline",
                            "Accuracy": check["baseline_direct_28x28_accuracy"], "Rejected images": 0})
            st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
                "Accuracy": st.column_config.NumberColumn(format="%.4f")})
            st.caption(f"{check['n_images_per_condition']} images per condition. {check['note']}")


def page_about() -> None:
    hero("About & Methodology", "How the system is built",
         "Dataset, preprocessing, network design, training, evaluation and limitations.")
    info = read_json(RUN_INFO_PATH)
    tabs = st.tabs(["Dataset", "Preprocessing", "CNN architecture", "Training", "Evaluation", "Limitations & use", "Task 3 mapping"])
    with tabs[0]:
        st.markdown(
            "**MNIST** (Modified National Institute of Standards and Technology) contains 70,000 grayscale "
            "images of handwritten digits, each 28×28 pixels with values 0-255. The standard split is "
            "**60,000 training** and **10,000 test** images, written by different people. "
            "NeuroWrite AI validates shapes, dtypes, label range and class coverage on every training run, "
            "holds out 10 % of the training images (stratified) as a validation set, and uses the test set "
            "only once for the final evaluation.")
    with tabs[1]:
        st.markdown(
            "**Training data**: pixels are divided by 255 so every value lies in [0, 1]; images get a channel "
            "axis → (N, 28, 28, 1); labels stay integers 0-9 (sparse categorical cross-entropy).\n\n"
            "**Uploaded images** pass through one shared pipeline (`utils/preprocessing.py`):\n"
            "1. Validate format, size and integrity (PNG/JPEG, ≤ 10 MB, 8-8000 px).\n"
            "2. Convert to grayscale (transparent PNGs use the alpha channel as the drawing).\n"
            "3. Detect polarity: if the image border is brighter than the Otsu threshold the digit is dark on light, "
            "and the image is inverted so the digit is white on black like MNIST.\n"
            "4. Remove background level, stretch contrast, and drop small noise blobs.\n"
            "5. Crop to the digit, scale it to fit 20×20 **keeping the aspect ratio**, centre it by its centre of mass in 28×28.\n"
            "6. Normalise to [0, 1] and validate the tensor (shape, dtype, finite, range) before inference.")
    with tabs[2]:
        model = get_model() if model_exists(MODEL_PATH) else None
        rows = describe_architecture(model if model is not None else build_cnn())
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.markdown(
            "- **Conv2D + ReLU** layers learn small visual detectors (edges, curves, loops) with shared weights.\n"
            "- **MaxPooling** halves the resolution, keeping the strongest responses, which gives tolerance to small shifts.\n"
            "- **Dropout** randomly silences neurons during training to reduce over-fitting.\n"
            "- **Flatten + Dense(128)** combines the detected features into a decision.\n"
            "- **Dense(10, softmax)** outputs ten probabilities that sum to 1.\n"
            "- **Augmentation layers** (small random rotation, shift and zoom) are active only during training.")
    with tabs[3]:
        if info:
            st.markdown(
                f"Last training run: **{info['epochs_run']} epochs** (best validation loss at epoch {info['best_epoch']}), "
                f"batch size {info['batch_size']}, Adam optimiser (initial learning rate {info['learning_rate']}), "
                f"seed {info['seed']}, {info['train_samples']:,} training / {info['validation_samples']:,} validation / "
                f"{info['test_samples']:,} test images, TensorFlow {info['tensorflow_version']}.")
        else:
            st.info("No training run recorded yet. Run `python train_model.py`.")
        st.markdown(
            "Loss: sparse categorical cross-entropy. **Early stopping** ends training when validation loss stops "
            "improving and restores the best weights; **ReduceLROnPlateau** halves the learning rate when "
            "validation loss stalls. Neither callback ever sees the test set.")
    with tabs[4]:
        st.markdown(
            "The saved model is evaluated once on the 10,000 test images: accuracy, cross-entropy loss, "
            "per-class and macro/weighted precision, recall and F1, a confusion matrix, and example "
            "correct/incorrect predictions. A separate *pipeline check* re-renders validation digits as "
            "uploads to confirm the preprocessing does not damage recognisable digits.")
    with tabs[5]:
        st.markdown(
            "**Limitations**\n"
            "- Recognises one isolated digit (0-9); not letters, words or multi-digit numbers.\n"
            "- MNIST digits are clean, centred and US-style; unusual styles, heavy noise or photos with "
            "shadows can reduce accuracy.\n"
            "- Softmax confidence can be high even when the answer is wrong.\n"
            "- High MNIST accuracy does not guarantee the same accuracy on your own handwriting.\n\n"
            "**Educational use**: teaching convolutional networks, demonstrating evaluation metrics, "
            "exploring how preprocessing affects model input, and discussing model confidence.")
    with tabs[6]:
        st.dataframe(pd.DataFrame([
            {"Requirement": "Image processing", "Where": "utils/preprocessing.py; Recognition Studio"},
            {"Requirement": "MNIST dataset", "Where": "utils/data.py; train_model.py"},
            {"Requirement": "Deep learning", "Where": "TensorFlow/Keras training in train_model.py"},
            {"Requirement": "CNN architecture", "Where": "utils/model.py (build_cnn)"},
            {"Requirement": "Image classification", "Where": "utils/inference.py; Studio and Samples pages"},
            {"Requirement": "Training and evaluation", "Where": "train_model.py, evaluate.py, utils/evaluation.py"},
        ]), hide_index=True)


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ✍️ NeuroWrite AI")
    st.caption("Handwritten digit recognition · CNN · MNIST")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed", key="page")
    st.divider()
    if model_exists(MODEL_PATH):
        st.success("Model ready")
    else:
        st.error("Model not trained")
    st.caption("Infynix Tech · Machine Learning Internship · Task 3")

{
    "Overview": page_overview,
    "Recognition Studio": page_studio,
    "Sample Predictions": page_samples,
    "Model Performance": page_performance,
    "About & Methodology": page_about,
}[page]()
