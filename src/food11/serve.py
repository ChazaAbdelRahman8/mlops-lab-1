import io
import os
from contextlib import asynccontextmanager

import mlflow
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image



MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MODEL_URI = "models:/food11@champion"

IMAGE_SIZE = (128, 128)

CATEGORIES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]




model = None




@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    print("=" * 60)
    print("Starting Food-11 API")
    print("=" * 60)

    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Loading model: {MODEL_URI}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model = mlflow.pyfunc.load_model(
        MODEL_URI
    )

    print("Model loaded successfully.")

    yield

    print("Stopping Food-11 API")




app = FastAPI(
    title="Food-11 Classification API",
    description="Classify food images using the champion Food-11 model.",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# Image preprocessing
# ============================================================

def preprocess_image(image: Image.Image):
    """
    Apply the same preprocessing used during model training.

    Output shape:
        (1, 3, 128, 128)
    """

    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS,
    )

    # Convert PIL image to numpy
    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    # Scale from [0, 255] to [0, 1]
    image_array /= 255.0

    # ImageNet normalization used during training
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    image_array = (
        image_array - mean
    ) / std

    # HWC -> CHW
    image_array = np.transpose(
        image_array,
        (2, 0, 1),
    )

    # Add batch dimension:
    # (3, 128, 128) -> (1, 3, 128, 128)
    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    return image_array.astype(
        np.float32
    )




def softmax(logits):
    """
    Convert model logits into probabilities.
    """

    logits = np.asarray(logits)

    logits = logits - np.max(
        logits,
        axis=1,
        keepdims=True,
    )

    exp_values = np.exp(logits)

    probabilities = (
        exp_values
        / np.sum(
            exp_values,
            axis=1,
            keepdims=True,
        )
    )

    return probabilities




@app.get("/health")
def health():
    return {
        "status": "ok"
    }




@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

   

    if not file.content_type or not file.content_type.startswith(
        "image/"
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    try:
        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        )

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read uploaded image.",
        )

   

    input_batch = preprocess_image(
        image
    )

  

    try:
        predictions = model.predict(
            input_batch
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        )



    probabilities = softmax(
        predictions
    )

    predicted_index = int(
        np.argmax(probabilities[0])
    )

    confidence = float(
        probabilities[0][predicted_index]
    )

    predicted_category = CATEGORIES[
        predicted_index
    ]


    return {
        "category": predicted_category,
        "confidence": confidence,
    }