import os
import torch
import pickle
import pandas as pd
import streamlit as st

import ai_related_topic_classifier as classifier_module

from transformers import BertTokenizer, BertModel
from config import MODEL_NAME, TOKENIZER_NAME, OUTPUT_DIR_MODEL

# =============================================================================
# Helper Functions for Inference
# =============================================================================


def load_model(model_path):
    """Load the saved classifier model from disk."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def get_bert_embedding(bert_model, tokenizer, texts, batch_size=32):
    """
    Compute BERT embeddings for a list of texts.

    Uses the [CLS] token from the last hidden state as the sentence embedding.
    """
    if isinstance(texts, str):
        texts = [texts]
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = bert_model(**inputs)
        # Use the [CLS] token embedding
        batch_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(batch_embeddings)
    return torch.cat(all_embeddings, dim=0)


def classify_texts(texts, classifier, bert_model, tokenizer):
    """
    Classify texts as 'AI-related' or 'Not AI-related'.

    Returns a list of tuples (label, confidence) where confidence is
    derived from the model's probability (if available).
    """
    embeddings = get_bert_embedding(bert_model, tokenizer, texts)
    df_embeddings = pd.DataFrame(
        embeddings.numpy(), columns=[f"col{i}" for i in range(embeddings.shape[1])]
    )
    predictions = classifier.predict(df_embeddings)

    # Get prediction probabilities if available
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(df_embeddings)
        confidences = proba.max(axis=1)
    else:
        confidences = [None] * len(predictions)

    results = []
    for pred, conf in zip(predictions, confidences):
        label = "AI-related" if pred == 1 else "Not AI-related"
        results.append((label, conf))
    return results


# =============================================================================
# Resource Loading (Cached)
# =============================================================================


# st.cache_resource is now recommended for caching expensive resources like models.
@st.cache_resource
def load_resources():
    """
    Cache and load heavy resources:
    - The saved classifier model,
    - BERT tokenizer,
    - BERT model.

    Raises a FileNotFoundError if the model file is missing.
    """
    model_path = f"{OUTPUT_DIR_MODEL}/ai_classifier_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found.")
    classifier = load_model(model_path)
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
    bert_model = BertModel.from_pretrained(MODEL_NAME)
    return classifier, tokenizer, bert_model


# =============================================================================
# Streamlit App Layout
# =============================================================================

st.markdown(
    "<h1 style='text-align: center; color: #1F8ECD;'>🤖 AI Related Topic Classifier</h1>",
    unsafe_allow_html=True,
)
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose App Mode", ["Inference", "Training"])

if app_mode == "Inference":
    st.header("Text Classification")

    # Check if model exists; if not, show warning and instruct the user to run training.
    model_file = f"{OUTPUT_DIR_MODEL}/ai_classifier_model.pkl"
    if not os.path.exists(model_file):
        st.warning(
            "No trained model found. Please run the training pipeline first in the 'Training' mode."
        )
        st.stop()

    text_input = st.text_area("Enter text to classify", None)
    if st.button("Classify"):
        try:
            classifier, tokenizer, bert_model = load_resources()
            results = classify_texts([text_input], classifier, bert_model, tokenizer)
            for label, confidence in results:
                if confidence is not None:
                    st.success(f"Prediction: {label} (Confidence: {confidence:.2f})")
                else:
                    st.success(f"Prediction: {label}")
        except Exception as e:
            st.error(f"An error occurred during inference: {e}")

elif app_mode == "Training":
    st.header("Training Pipeline")
    st.warning(
        "The training pipeline uses web scraping and Selenium. It might take a long time and requires an appropriate runtime environment."
    )
    if st.button("Run Training Pipeline"):
        try:
            import ai_related_topic_classifier as classifier_module

            classifier_module.main()
            st.success("Training completed successfully!")
        except Exception as e:
            st.error(f"An error occurred during training: {e}")
