import tensorflow as tf

# Load your existing Keras model
model = tf.keras.models.load_model("C:/Users/user/OneDrive/Desktop/S3/best_emotion_model.keras")

# Create a TFLiteConverter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (Optional) Enable optimization to reduce size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the converted TFLite model
with open("C:/Users/user/OneDrive/Desktop/S3/best_emotion_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Conversion complete!")
