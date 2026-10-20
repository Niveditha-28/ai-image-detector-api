
from pathlib import Path
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Image Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://niveditha-ai-notebook.kindcamel13.chatgpt.site",
    ],
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Load the trained CNN once when the server starts
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "best_ai_image_detector.keras"
)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported"
        )

    contents = await file.read(MAX_FILE_SIZE + 1)

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 5 MB"
        )

    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
        image = image.resize((256, 256))
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    # Same preprocessing as the Colab model
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Sigmoid output: probability of AI-generated class
    probability_ai = float(model.predict(image_array, verbose=0)[0][0])

    class_id = 1 if probability_ai >= 0.5 else 0
    prediction = "AI-Generated" if class_id == 1 else "Real"
    confidence = probability_ai if class_id == 1 else 1 - probability_ai

    return {
        "prediction": prediction,
        "class_id": class_id,
        "ai_probability": round(probability_ai, 4),
        "confidence": round(confidence, 4)
    }
