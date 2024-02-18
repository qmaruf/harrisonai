import sys

sys.path.append("src")
import base64
from pathlib import Path
from typing import Dict

import requests
import streamlit as st
from config import config
from loguru import logger

url = config.api_endpoint
storage_path = Path(config.storage_path)

st.title("Pet Detection")

uploaded_file = st.file_uploader("Choose a jpg file")


def show_response(predicted_labels: Dict, mask_path: str) -> None:
    """
    Show response from server in streamlit ui
    args:
        predicted_labels: Dict
        mask_path: str
    returns:
        None
    """
    total_pet = predicted_labels["cat"] + predicted_labels["dog"]
    st.info(f"Total pets: {total_pet}")
    if predicted_labels["cat"] > 0:
        st.info(f"Cat Breed: {predicted_labels['cat_breed']}")
    if predicted_labels["dog"] > 0:
        st.info(f"Dog Breed: {predicted_labels['dog_breed']}")
    st.image(mask_path)


def verify(uploaded_file) -> bool:
    """
    Verify if uploaded file is jpg
    args:
        uploaded_file: file
    """
    return uploaded_file.name.endswith(".jpg")


from io import BytesIO

if uploaded_file is not None:
    if verify(uploaded_file):
        image_bytes = BytesIO(uploaded_file.getvalue())

        response = requests.post(
            url, files={"file": ("filename.jpg", image_bytes, "image/jpeg")}
        )
        logger.info(f"Received response {response}")

        response = response.json()

        masked_img_base64 = response["masked_img_base64"]
        img_data = base64.b64decode(masked_img_base64)

        with open("masked_img.jpg", "wb") as hndl:
            hndl.write(img_data)

        predicted_labels = response["predicted_labels"]

        show_response(predicted_labels, "masked_img.jpg")
        logger.info(f"Received response {response}")
    else:
        st.error("Please upload a valid jpg file")
