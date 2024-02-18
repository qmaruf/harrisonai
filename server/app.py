from typing import Dict

from fastapi import FastAPI
from loguru import logger

from src.inference import inference_img

app = FastAPI()


@app.get("/")
def ping() -> str:
    """
    Ping server
    """
    return "pong"


@app.get("/inference")
def inference(image_path: str) -> Dict:
    """
    Inference on single image
    """
    logger.info(f"Received request for image: {image_path}")
    predicted_labels, mask_path = inference_img(image_path)
    return {"predicted_labels": predicted_labels, "mask_path": mask_path}
