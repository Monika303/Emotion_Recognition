from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import streamlit as st
import os

# Load the model
model = load_model(r"C:\Users\user\OneDrive\Desktop\S3\best_emotion_model.keras")

# FER2013 emotion labels
emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

# Map emotions to songs
emotion_songs = {
    'anger': r"C:\Users\user\OneDrive\Desktop\S3\angry.mp3",
    'disgust': r"C:\Users\user\OneDrive\Desktop\S3\disgust.mp3",
    'fear': r"C:\Users\user\OneDrive\Desktop\S3\fear.mp3",
    'happiness': r"C:\Users\user\OneDrive\Desktop\S3\happy.mp3",
    'sadness': r"C:\Users\user\OneDrive\Desktop\S3\sad.mp3",
    'surprise': r"C:\Users\user\OneDrive\Desktop\S3\surprise.mp3",
    'neutral': r"C:\Users\user\OneDrive\Desktop\S3\surprise.mp3"
}

# Fun emojis/effects for each emotion
emotion_effects = {
    'anger': "⚡💥",
    'disgust': "🤢🟢",
    'fear': "😱👻",
    'happiness': "🎈🎉",
    'sadness': "😢☔",
    'surprise': "😮✨",
    'neutral': "😐🌀"
}

# --- Function to predict emotion ---
def predict_emotion(img):
    img_resized = img.resize((48, 48)).convert('L')
    img_array = np.array(img_resized).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))
    prediction = model.predict(img_array)
    return emotion_labels[np.argmax(prediction)]

# --- Streamlit UI ---
st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("Emotion Recognition System 🎉")
st.markdown("Upload a face image and see the predicted emotion with a matching song!")

# File uploader with unique key
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"], 
    key="unique_image_upload"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display image in normal size
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Predict emotion
    emotion = predict_emotion(image)
    
    # Display emotion in big fun text with emoji/effect
    effect = emotion_effects.get(emotion, "")
    st.markdown(
        f"<h1 style='color: #ff4b4b; font-size: 60px; text-align:center'>{emotion.upper()} {effect}</h1>",
        unsafe_allow_html=True
    )
    
    # Play corresponding song if file exists
    song_file = emotion_songs.get(emotion)
    if song_file and os.path.exists(song_file):
        audio_file = open(song_file, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')
    else:
        st.warning(f"Song file for {emotion} not found!")
