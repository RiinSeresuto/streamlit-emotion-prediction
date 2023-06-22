import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

st.set_page_config(
    page_title="Emotion Prediction",
    layout="centered",
)

st.markdown(
    """
    <style>
        .header{
            text-align: center;
            text-transform: uppercase;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="header">Emotion Prediction</h2>', unsafe_allow_html=True)

# Load tokenizer
def load_tokenizer():
    with open('tokenizer.json', 'r') as file:
        tokenizer_json = file.read()

    tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer

# Loading model
def load_model():
    model = tf.keras.models.load_model('emotion-model.h5')
    return model


textInput = st.text_input("Describe what you feel right now",
    key="text-input",
    placeholder="i feel a little mellow today")

def predict_emotion(text):
    tokenizer = load_tokenizer()
    model = load_model()

    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, padding='post', maxlen=100)

    prediction = model.predict(sequence)
    emotion_label = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
    predicted_label = emotion_label[np.argmax(prediction[0])]

    return predicted_label

# Predict Emotion
if textInput:
    prediction = predict_emotion(textInput)

    st.markdown("<h3>Predicted</h3>")
    st.header(prediction)