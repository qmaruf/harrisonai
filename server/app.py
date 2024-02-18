import base64
import os
import shutil
import uuid
from typing import Dict

from fastapi import FastAPI, File, UploadFile
from loguru import logger

from src.config import config
from src.inference import inference_img

app = FastAPI()


@app.get("/")
def ping() -> str:
    """
    Ping server
    """
    return "pong"


@app.post("/inference")
async def inference(file: UploadFile = File(...)) -> Dict:
    """
    Inference on single image received as a file upload.
    """

    image_id = f"{str(uuid.uuid4())}.jpg"
    file_path = os.path.join((config.storage_path), image_id)
    logger.info(f"Saving image to {file_path}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    predicted_labels, masked_img_byte = inference_img(file_path)
    masked_img_base64 = base64.b64encode(masked_img_byte).decode()

    return {
        "predicted_labels": predicted_labels,
        "masked_img_base64": masked_img_base64,
    }
