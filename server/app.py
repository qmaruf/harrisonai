from typing import Dict

from fastapi import FastAPI

from src.inference import inference_img

app = FastAPI()


@app.get("/inference")
def inference(image_path: str) -> Dict:
    """
    Inference on single image
    """
    predicted_labels, mask_path = inference_img(image_path)
    return {"predicted_labels": predicted_labels, "mask_path": mask_path}
