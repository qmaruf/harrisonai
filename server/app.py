from typing import Dict

from fastapi import FastAPI, UploadFile, File

from loguru import logger
import os
import shutil
import uuid

from src.inference import inference_img
from src.config import config

app = FastAPI()


@app.get("/")
def ping() -> str:
    """
    Ping server
    """
    return "pong"


# @app.get("/inference")
# def inference(image_path: str) -> Dict:
#     """
#     Inference on single image
#     """
#     logger.info(f"Received request for image: {image_path}")
#     predicted_labels, mask_path = inference_img(image_path)
#     return {"predicted_labels": predicted_labels, "mask_path": mask_path}


@app.get("/inference")
async def inference(file: UploadFile = File(...)) -> Dict:
    """
    Inference on single image received as a file upload.
    """
    # Define the path where you want to save the file
    image_id = f"{str(uuid.uuid4())}.jpg"
    file_path = os.path.join((config.storage_path), image_id)

    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Once the file is saved, you can perform inference on it
    predicted_labels, mask_path = inference_img(file_path)

    # Log and return results
    logger.info(f"Received and processed file: {file.filename}")
    return {"predicted_labels": predicted_labels, "mask_path": mask_path}
