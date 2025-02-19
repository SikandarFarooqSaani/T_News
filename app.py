import streamlit as st
import torch
import os
import zipfile
import gdown  # Use gdown to correctly download from Google Drive
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

# Model download settings
MODEL_URL = "https://drive.google.com/uc?export=download&id=1RmZK1LAu2ufN35pouL1pAty7dCYja8mB"
MODEL_DIR = "saved_model"
MODEL_ZIP = "saved_model.zip"

# Download and extract the model if not present
if not os.path.exists(MODEL_DIR):
    st.info("Downloading model... Please wait ⏳")
    gdown.download(MODEL_URL, MODEL_ZIP, quiet=False)  # Use gdown to download correctly

    # Extract model
    with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    os.remove(MODEL_ZIP)
    st.success("Model downloaded successfully ✅")

# Load the model and tokenizer
model = GPT2ForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model.config.pad_token_id = tokenizer.pad_token_id

# Prediction function
def predict_news(news_text):
    inputs = tokenizer(news_text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return "Real" if prediction == 1 else "Fake"

# Streamlit App
st.title("Fake News Detection")
st.write("Enter news text to check if it's Fake or Real.")

news_input = st.text_area("News Text", height=200)

if st.button("Predict"):
    if news_input:
        prediction = predict_news(news_input)
        st.subheader("Prediction:")
        if prediction == "Real":
            st.success("This news is likely Real.")
        else:
            st.error("This news is likely Fake.")
    else:
        st.warning("Please enter news text to get a prediction.")
