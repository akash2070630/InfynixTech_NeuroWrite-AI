"""Smoke tests: every page of the Streamlit app renders without raising."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["Overview", "Recognition Studio", "Sample Predictions", "Model Performance", "About & Methodology"]


def render(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio(key="page").set_value(page).run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_with_current_artefacts(page):
    at = render(page)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_when_model_is_missing(page, monkeypatch, tmp_path):
    import utils.config as config

    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "missing.keras")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "missing_metrics.json")
    monkeypatch.setattr(config, "RUN_INFO_PATH", tmp_path / "missing_info.json")
    monkeypatch.setattr(config, "SAMPLES_PATH", tmp_path / "missing_samples.npz")
    at = render(page)
    assert not at.exception, [e.value for e in at.exception]
    if page in ("Overview", "Recognition Studio", "Sample Predictions"):
        text = " ".join(w.value for w in at.warning)
        assert "train_model.py" in text or "No trained model" in text
