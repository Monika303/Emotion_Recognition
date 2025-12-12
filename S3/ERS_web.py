import numpy as np
from PIL import Image
import streamlit as st
import os
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="best_emotion_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# FER2013 emotion labels
emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

# Map emotions to songs
emotion_songs = {
    'anger': r"C:\Users\user\OneDrive\Desktop\S3\Songs\angry.mp3",
    'disgust': r"C:\Users\user\OneDrive\Desktop\S3\Songs\disgust.mp3",
    'fear': r"C:\Users\user\OneDrive\Desktop\S3\Songs\fear.mp3",
    'happiness': r"C:\Users\user\OneDrive\Desktop\S3\Songs\happy.mp3",
    'sadness': r"C:\Users\user\OneDrive\Desktop\S3\Songs\sad.mp3",
    'surprise': r"C:\Users\user\OneDrive\Desktop\S3\Songs\surprise.mp3",
    'neutral': r"C:\Users\user\OneDrive\Desktop\S3\Songs\neutral.mp3"
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

def predict_emotion(image):
    # Convert image to 48x48 grayscale
    img = image.resize((48, 48)).convert('L')
    img_array = np.array(img, dtype=np.float32)

    # Add batch and channel dimensions
    input_data = np.expand_dims(img_array, axis=0)   # batch
    input_data = np.expand_dims(input_data, axis=-1) # channel

    # Set the tensor and invoke interpreter
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output and return predicted class index
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data)

# --- Streamlit UI ---
st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("Emotion Recognition System 🎉")
st.markdown("Upload a face image and see the predicted emotion with a matching song!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="unique_image_upload")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Predict emotion
    emotion_index = predict_emotion(image)
    emotion_label = emotion_labels[emotion_index]
    
    # Get emoji/effect
    effect = emotion_effects.get(emotion_label, "")
    
    # Display emotion
    st.markdown(
        f"<h1 style='color: #ff4b4b; font-size: 60px; text-align:center'>{emotion_label.upper()} {effect}</h1>",
        unsafe_allow_html=True
    )
    
    # Play corresponding song
    song_file = emotion_songs.get(emotion_label)
    if song_file and os.path.exists(song_file):
        with open(song_file, 'rb') as audio_file:
            st.audio(audio_file.read(), format='audio/mp3')
    else:
        st.warning(f"Song file for {emotion_label} not found!")
