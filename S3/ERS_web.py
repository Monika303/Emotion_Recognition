import os
from typing import Optional

import numpy as np
from PIL import Image
import streamlit as st

# Try to import TensorFlow Lite interpreter (assumes tensorflow or tflite-runtime is installed)
try:
    import tflite_runtime.interpreter as tflite_interpreter  # type: ignore
    Interpreter = tflite_interpreter.Interpreter
except Exception:
    import tensorflow as tf  # type: ignore
    Interpreter = tf.lite.Interpreter

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Look for model in common locations (adjust if your model is elsewhere)
MODEL_CANDIDATES = [
    os.path.join(script_dir, "best_emotion_model.tflite"),
    os.path.join(script_dir, os.pardir, "best_emotion_model.tflite"),
    os.path.join(os.getcwd(), "best_emotion_model.tflite"),
]
MODEL_PATH: Optional[str] = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)

# Labels and songs (use repo-relative S3/Songs/)
emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
songs_dir = os.path.join(script_dir, "Songs")
emotion_songs = {
    'anger': os.path.join(songs_dir, "angry.mp3"),
    'disgust': os.path.join(songs_dir, "disgust.mp3"),
    'fear': os.path.join(songs_dir, "fear.mp3"),
    'happiness': os.path.join(songs_dir, "happy.mp3"),
    'sadness': os.path.join(songs_dir, "sad.mp3"),
    'surprise': os.path.join(songs_dir, "surprise.mp3"),
    'neutral': os.path.join(songs_dir, "neutral.mp3"),
}
emotion_effects = {
    'anger': "⚡💥",
    'disgust': "🤢🟢",
    'fear': "😱👻",
    'happiness': "🎈🎉",
    'sadness': "😢☔",
    'surprise': "😮✨",
    'neutral': "😐🌀"
}

# --- Load interpreter ---
interpreter = None
input_details = output_details = None
if MODEL_PATH:
    try:
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        interpreter = None
        st.error(f"Failed to load TFLite model: {e}")
else:
    st.warning("TFLite model not found. Place 'best_emotion_model.tflite' in the repo root or the S3 folder.")

def predict_emotion(image: Image.Image) -> Optional[int]:
    if interpreter is None or input_details is None or output_details is None:
        return None

    img = image.resize((48, 48)).convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0  # normalize 0-1
    inp = np.expand_dims(arr, axis=0)  # batch
    if inp.ndim == 3:
        inp = np.expand_dims(inp, axis=-1)  # channel

    expected_dtype = input_details[0]['dtype']
    inp = inp.astype(expected_dtype)

    # reshape if necessary
    try:
        inp = inp.reshape(tuple(input_details[0]['shape']))
    except Exception:
        pass

    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    return int(np.argmax(out))

# --- Streamlit UI ---
st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("Emotion Recognition System")
st.caption("Upload a face image and see the predicted emotion with a matching song.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Unable to open image: {e}")
        image = None

    if image:
        st.image(image, use_container_width=True)
        idx = predict_emotion(image)
        if idx is None:
            st.info("Prediction unavailable (model not loaded). Ensure 'best_emotion_model.tflite' is in the repo and requirements include TensorFlow or tflite-runtime.")
        else:
            label = emotion_labels[idx] if 0 <= idx < len(emotion_labels) else "unknown"
            effect = emotion_effects.get(label, "")
            st.markdown(f"## {label.upper()} {effect}")

            song_path = emotion_songs.get(label)
            if song_path and os.path.exists(song_path):
                with open(song_path, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")
            else:
                st.info("No matching song found for this emotion.")