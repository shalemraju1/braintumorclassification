import tensorflow as tf
import os

model_path = os.path.join("model", "brain_tumor_cnn_final.h5")

model = tf.keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

output_path = os.path.join("model", "brain_tumor_model.tflite")

with open(output_path, "wb") as f:
    f.write(tflite_model)

print("TFLite model created successfully")