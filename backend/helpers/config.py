from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"
UPLOAD_DIR = STATIC_DIR / "uploads"
HEATMAP_DIR = STATIC_DIR / "heatmaps"
IMAGE_DIR = STATIC_DIR / "images"
MODEL_PATH = BASE_DIR / "model" / "brain_tumor_model.tflite"

APP_NAME = "Brain Tumor Detection"
SECRET_KEY = os.environ.get("SECRET_KEY", "brain_tumor_project_2026_secure_key")
DATABASE_URL = os.environ.get("DATABASE_URL")

PORT = int(os.environ.get("PORT", "10000"))
MAX_CONTENT_LENGTH = 12 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASSES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
