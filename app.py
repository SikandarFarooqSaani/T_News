import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

# Load the model from Hugging Face
MODEL_NAME = "SikandarFarooqSaani/transformers"

@st.cache_resource
def load_model():
    st.info("Loading model from Hugging Face... ⏳")
    model = GPT2ForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id
    st.success("Model loaded successfully ✅")
    return model, tokenizer

# Load model once
model, tokenizer = load_model()

# Prediction function
def predict_news(news_text):
    inputs = tokenizer(news_text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return "Real" if prediction == 1 else "Fake"

# Streamlit UI
st.title("Fake News Detection")
st.write("Enter news text to check if it's Fake or Real.")

news_input = st.text_area("News Text", height=200)

if st.button("Predict"):
    if news_input:
        prediction = predict_news(news_input)
        st.subheader("Prediction:")
        st.success("This news is likely Real.") if prediction == "Real" else st.error("This news is likely Fake.")
    else:
        st.warning("Please enter news text to get a prediction.")
