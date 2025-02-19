import os
import streamlit as st
import torch
import zipfile
import gdown
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

# Model download settings
MODEL_URL = "https://drive.google.com/uc?export=download&id=10yPukQO7DvURZJ4ZqfWdaZHYEpYm_-4f"
MODEL_DIR = "saved_model"
MODEL_ZIP = "saved_model.zip"

# Download and extract the model if not present
if not os.path.exists(MODEL_DIR):
    st.info("Downloading model... Please wait ⏳")
    gdown.download(MODEL_URL, MODEL_ZIP, quiet=False)

    # Extract model
    with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    os.remove(MODEL_ZIP)
    st.success("Model downloaded successfully ✅")

# 🔴 **DEBUG: Print extracted files**
st.write("Extracted Files:", os.listdir(MODEL_DIR))

# 🔴 **DEBUG: Check if model.safetensors exists**
if not os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
    st.error("❌ model.safetensors is missing! Check your ZIP file.")

# Load the model and tokenizer
try:
    model = GPT2ForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model.config.pad_token_id = tokenizer.pad_token_id
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error(f"❌ Model failed to load: {str(e)}")
