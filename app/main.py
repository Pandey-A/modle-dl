import base64
import io
import json
import os
from typing import List

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

MODEL_PATH = os.getenv("MODEL_PATH", "lung_cancer_model_final2.h5")
CLASS_NAMES_ENV = os.getenv("CLASS_NAMES", "")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "class_names.json")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))

app = FastAPI(
    title="Lung Cancer Model API",
    version="1.0.0",
    description="Inference API for the lung cancer image classification model.",
)

model = None
class_names: List[str] = []


class Base64PredictRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 image string. Data URL is also accepted.")


def _get_output_dim(loaded_model) -> int:
    output_shape = loaded_model.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]

    if output_shape is None or len(output_shape) == 0 or output_shape[-1] is None:
        raise RuntimeError("Unable to determine model output dimension.")

    return int(output_shape[-1])


def _resolve_class_names(output_dim: int) -> List[str]:
    if CLASS_NAMES_ENV.strip():
        labels = [label.strip() for label in CLASS_NAMES_ENV.split(",") if label.strip()]
        if len(labels) != output_dim:
            raise RuntimeError(
                f"CLASS_NAMES count ({len(labels)}) does not match model output_dim ({output_dim})."
            )
        return labels

    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise RuntimeError("class_names.json must be a JSON array of strings.")

        if len(data) != output_dim:
            raise RuntimeError(
                f"class_names.json count ({len(data)}) does not match model output_dim ({output_dim})."
            )

        return data

    return [f"class_{i}" for i in range(output_dim)]


def _decode_base64_image(raw_value: str) -> bytes:
    value = raw_value.strip()
    if not value:
        raise HTTPException(status_code=400, detail="image_base64 is empty")

    if "," in value and value.lower().startswith("data:image"):
        value = value.split(",", 1)[1]

    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image data") from exc


def _preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def _predict_from_array(image_array: np.ndarray) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    raw_pred = model.predict(image_array, verbose=0)
    probabilities = np.asarray(raw_pred[0], dtype=np.float32)

    predicted_index = int(np.argmax(probabilities))
    predicted_label = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    per_class = [
        {
            "class_index": idx,
            "label": class_names[idx],
            "probability": float(probabilities[idx]),
        }
        for idx in range(len(probabilities))
    ]

    return {
        "predicted_index": predicted_index,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": per_class,
    }


@app.on_event("startup")
def startup_event() -> None:
    global model
    global class_names

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")

    try:
        import tf_keras
        model = tf_keras.models.load_model(MODEL_PATH, compile=False)
    except ImportError:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    output_dim = _get_output_dim(model)
    class_names = _resolve_class_names(output_dim)

    dummy = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)


@app.get("/")
def root() -> dict:
    return {"message": "Lung Cancer Model API is running"}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "image_size": IMAGE_SIZE,
        "class_count": len(class_names),
        "class_names": class_names,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    image_array = _preprocess_image(image_bytes)
    return _predict_from_array(image_array)


@app.post("/predict/base64")
def predict_base64(payload: Base64PredictRequest) -> dict:
    image_bytes = _decode_base64_image(payload.image_base64)
    image_array = _preprocess_image(image_bytes)
    return _predict_from_array(image_array)
