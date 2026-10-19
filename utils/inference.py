"""End-to-end recognition: raw image -> preprocessing -> CNN -> prediction."""
from __future__ import annotations

from dataclasses import dataclass

from utils.model import Prediction, predict_digit
from utils.preprocessing import POLARITY_AUTO, PreprocessResult, preprocess_image

LOW_CONFIDENCE = 0.70
LOW_MARGIN = 0.20


@dataclass
class Recognition:
    prediction: Prediction
    processed: PreprocessResult

    @property
    def is_uncertain(self) -> bool:
        """True when the model is not clearly sure (low top-1 or a close runner-up)."""
        return (self.prediction.confidence < LOW_CONFIDENCE) or (self.prediction.margin < LOW_MARGIN)


def recognise(model, source, polarity: str = POLARITY_AUTO, auto_crop: bool = True) -> Recognition:
    """Run the single shared preprocessing pipeline and the model on ``source``.

    Raises ``ImageValidationError`` / ``PreprocessingError`` with user-readable
    messages when the image cannot be used.
    """
    processed = preprocess_image(source, polarity=polarity, auto_crop=auto_crop)
    return Recognition(prediction=predict_digit(model, processed.tensor), processed=processed)
