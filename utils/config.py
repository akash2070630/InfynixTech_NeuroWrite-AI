"""Central configuration: paths, hyper-parameters and the colour palette shared by every module.

All paths are relative to the project root, so nothing depends on a local absolute path.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "neurowrite_cnn.keras"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
REPORT_PATH = OUTPUTS_DIR / "classification_report.txt"
CONFUSION_CSV_PATH = OUTPUTS_DIR / "confusion_matrix.csv"
HISTORY_PATH = OUTPUTS_DIR / "training_history.json"
RUN_INFO_PATH = OUTPUTS_DIR / "run_info.json"
SUMMARY_PATH = OUTPUTS_DIR / "model_summary.txt"
SAMPLES_PATH = OUTPUTS_DIR / "sample_images.npz"
PIPELINE_CHECK_PATH = OUTPUTS_DIR / "pipeline_check.json"

IMAGE_SIZE = 28
NUM_CLASSES = 10
CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]

# Training settings
SEED = 42
VAL_FRACTION = 0.10
BATCH_SIZE = 128
MAX_EPOCHS = 20
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-5
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 2
EARLY_STOP_PATIENCE = 5

# Upload limits
ALLOWED_EXTENSIONS = ("png", "jpg", "jpeg")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MIN_IMAGE_SIDE = 8
MAX_IMAGE_SIDE = 8000

# Palette (matches the Streamlit UI)
NAVY = "#0B2545"
BLUE = "#1D4ED8"
SKY = "#60A5FA"
PALE = "#DBEAFE"
SLATE = "#475569"
GRID = "#E2E8F0"
GOOD = "#15803D"
BAD = "#B91C1C"
