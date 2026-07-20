from __future__ import annotations

import base64
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from .config import CLASSES, MODEL_PATH


_interpreter = None
_input_details = None
_output_details = None
_tflite_module = None


def preprocess_image(image: Image.Image) -> np.ndarray:
    resized_image = image.resize((150, 150))
    image_array = np.array(resized_image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


@lru_cache(maxsize=1)
def get_interpreter():
    global _interpreter, _input_details, _output_details, _tflite_module

    if _tflite_module is None:
        try:
            import tflite_runtime.interpreter as tflite_module
        except ImportError:  # pragma: no cover - fallback for environments with TensorFlow only
            try:
                import tensorflow.lite as tflite_module
            except ImportError as exc:  # pragma: no cover - deployment issue
                raise RuntimeError("TensorFlow Lite runtime is not installed") from exc

        _tflite_module = tflite_module

    if _interpreter is None:
        _interpreter = _tflite_module.Interpreter(model_path=str(MODEL_PATH))
        _interpreter.allocate_tensors()
        _input_details = _interpreter.get_input_details()
        _output_details = _interpreter.get_output_details()

    return _interpreter


def get_risk_level(predicted_class: str, confidence: float) -> str:
    if predicted_class == "No Tumor":
        return "Low"
    if confidence >= 85:
        return "High"
    if confidence >= 60:
        return "Medium"
    return "Low"


def generate_heatmap(image: Image.Image) -> str:
    rgb_image = np.array(image.convert("RGB"))
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(rgb_image, 0.6, heatmap, 0.4, 0)
    _, buffer = cv2.imencode(".jpg", overlay)
    return base64.b64encode(buffer).decode("utf-8")


def run_prediction(image: Image.Image) -> dict:
    processed_image = preprocess_image(image)
    interpreter = get_interpreter()

    input_details = _input_details or interpreter.get_input_details()
    output_details = _output_details or interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], processed_image)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])[0]
    probabilities = prediction * 100.0
    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASSES[predicted_index]
    confidence = float(probabilities[predicted_index])
    heatmap_base64 = generate_heatmap(image)

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "risk_level": get_risk_level(predicted_class, confidence),
        "all_probabilities": {
            CLASSES[index]: round(float(probabilities[index]), 2)
            for index in range(len(CLASSES))
        },
        "heatmap_base64": heatmap_base64,
    }
