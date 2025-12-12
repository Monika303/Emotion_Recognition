import numpy as np
from PIL import Image
import streamlit as st
import os

# Try to import a TFLite interpreter implementation
Interpreter = None
try:
    import tflite_runtime.interpreter as tflite_interpreter  # type: ignore
    Interpreter = tflite_interpreter.Interpreter
    _INTERP_BACKEND = "tflite-runtime"
except Exception:
    try:
        import tensorflow as tf  # type: ignore
        Interpreter = tf.lite.Interpreter
        _INTERP_BACKEND = "tensorflow.lite"
    except Exception:
        Interpreter = None
        _INTERP_BACKEND = None

# Helper: find model in a few likely locations
def find_model(filename="best_emotion_model.tflite"):
    candidates = []
    # same dir as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(script_dir, filename))
    # repo root (one level up from S3/)
    candidates.append(os.path.join(script_dir, os.pardir, filename))
    # current working directory
    candidates.append(os.path.join(os.getcwd(), filename))
    # explicit S3 folder
    candidates.append(os.path.join(script_dir, "best_emotion_model.tflite"))
    candidates = [os.path.normpath(p) for p in candidates]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# FER2013 emotion labels
emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

# Map emotions to songs (use repo-relative paths)
script_dir = os.path.dirname(os.path.abspath(__file__))
songs_dir = os.path.join(script_dir, "Songs")  # expects S3/Songs/
emotion_songs = {
    'anger': os.path.join(songs_dir, "angry.mp3"),
    'disgust': os.path.join(songs_dir, "disgust.mp3"),
    'fear': os.path.join(songs_dir, "fear.mp3"),
    'happiness': os.path.join(songs_dir, "happy.mp3"),
    'sadness': os.path.join(songs_dir, "sad.mp3"),
    'surprise': os.path.join(songs_dir, "surprise.mp3"),
    'neutral': os.path.join(songs_dir, "neutral.mp3")
}

# effects
emotion_effects = {
    'anger': "⚡💥",
    'disgust': "🤢🟢",
    'fear': "😱👻",
    'happiness': "🎈🎉",
    'sadness': "😢☔",
    'surprise': "😮✨",
    'neutral': "😐🌀"
}

# Attempt to locate and load the model
MODEL_FILENAME = "best_emotion_model.tflite"
MODEL_PATH = find_model(MODEL_FILENAME)

interpreter = None
input_details = output_details = None

if Interpreter is None:
    st.error("No TFLite interpreter available: neither tflite-runtime nor tensorflow.lite could be imported. Check requirements.")
else:
    if MODEL_PATH is None:
        st.warning(f"TFLite model '{MODEL_FILENAME}' not found. Expected locations checked. Please add the model to the repo (e.g. repo root or S3/) and redeploy.")
    else:
        try:
            interpreter = Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            st.success(f"Loaded model from {MODEL_PATH} using {_INTERP_BACKEND}")
        except Exception as e:
            st.error(f"Failed to initialize TFLite Interpreter with model at {MODEL_PATH}. See details below.")
            st.exception(e)
            interpreter = None

def predict_emotion(image: Image.Image):
    if interpreter is None or input_details is None or output_details is None:
        return None

    img = image.resize((48, 48)).convert('L')
    img_array = np.array(img, dtype=np.float32) / 255.0
    input_data = np.expand_dims(img_array, axis=0)
    if input_data.ndim == 3:
        input_data = np.expand_dims(input_data, axis=-1)

    expected_dtype = input_details[0]['dtype']
    try:
        input_data = input_data.astype(expected_dtype)
    except Exception:
        input_data = input_data.astype(np.float32)

    try:
        # reshape if necessary
        input_shape = tuple(input_details[0]['shape'])
        input_data = input_data.reshape(input_shape)
    except Exception:
        pass

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return int(np.argmax(output_data))

# --- Streamlit UI ---
st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("Emotion Recognition System 🎉")
st.markdown("Upload a face image and see the predicted emotion with a matching song!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="unique_image_upload")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Unable to open image: {e}")
        image = None

    if image:
        st.image(image, caption='Uploaded Image', use_container_width=True)

        emotion_index = predict_emotion(image)
        if emotion_index is None:
            st.info("Prediction unavailable (model not loaded).")
        else:
            if 0 <= emotion_index < len(emotion_labels):
                emotion_label = emotion_labels[emotion_index]
            else:
                emotion_label = "unknown"

            effect = emotion_effects.get(emotion_label, "")
            st.markdown(
                f"<h1 style='color: #ff4b4b; font-size: 60px; text-align:center'>{emotion_label.upper()} {effect}</h1>",
                unsafe_allow_html=True
            )

            song_file = emotion_songs.get(emotion_label)
            if song_file and os.path.exists(song_file):
                with open(song_file, 'rb') as audio_file:
                    st.audio(audio_file.read(), format='audio/mp3')
            else:
                st.warning(f"Song file for {emotion_label} not found at expected path: {song_file}")