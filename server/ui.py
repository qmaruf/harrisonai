import sys

sys.path.append("src")
import os
import uuid
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


def save_image(uploaded_file) -> str:
    """
    Save image to storage
    args:
        uploaded_file: file
    returns:
        image_path: str
    """
    image_id = f"{str(uuid.uuid4())}.jpg"
    image_path = os.path.join(str(storage_path), image_id)

    logger.info(f"Saving image to {image_path}")

    with open(image_path, "wb") as hndl:
        hndl.write(uploaded_file.getbuffer())

    return image_path


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


if uploaded_file is not None:
    if verify(uploaded_file):
        saved_image_path = save_image(uploaded_file)
        response = requests.get(url, params={"image_path": saved_image_path})

        response = response.json()
        mask_path = response["mask_path"]
        predicted_labels = response["predicted_labels"]

        show_response(predicted_labels, mask_path)
        logger.info(f"Received response {response}")
    else:
        st.error("Please upload a valid jpg file")
