from flask_cors import CORS
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import base64
from io import BytesIO
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app)

app.secret_key = "brain_tumor_project_2026_secure_key"

# =========================
# Database Connection
# =========================

db = None
cursor = None

try:
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="brain_tumor"
    )
    cursor = db.cursor()
    print("Database connected successfully")
except Exception as e:
    print("Database not available, running without DB:", e)

# =========================
# Load Model
# =========================

model_path = os.path.join("model", "brain_tumor_cnn_final.h5")
model = tf.keras.models.load_model(model_path)

dummy_input = tf.zeros((1,150,150,3))
model(dummy_input)

# =========================
# Class Labels
# =========================

classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# =========================
# Image Preprocessing
# =========================

def preprocess_image(image):
    image = image.resize((150,150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =========================
# Grad-CAM
# =========================

def generate_gradcam(model, image_array):

    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        return None

    conv_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=last_conv_layer.output
    )

    conv_output = conv_model(image_array)

    final_dense = model.layers[-1]
    weights = final_dense.get_weights()[0]

    prediction = model.predict(image_array)
    class_index = np.argmax(prediction)
    class_weights = weights[:,class_index]

    conv_output = conv_output[0].numpy()

    heatmap = np.zeros(shape=conv_output.shape[:2],dtype=np.float32)

    for i in range(conv_output.shape[-1]):
        heatmap += conv_output[:,:,i] * class_weights[i]

    heatmap = np.maximum(heatmap,0)

    if np.max(heatmap) == 0:
        return None

    heatmap /= np.max(heatmap)

    return heatmap

# =========================
# Routes
# =========================

@app.route("/")
def home():

    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template("index.html")

# =========================
# Register
# =========================

@app.route("/register", methods=["GET", "POST"])
def register():

    if cursor is None:
        flash("Registration disabled: database not available.", "danger")
        return redirect(url_for("login"))

    if request.method == "POST":

        name = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        hashed_password = generate_password_hash(password)

        cursor.execute("SELECT * FROM users WHERE email=%s",(email,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash("Email already registered","danger")
            return redirect(url_for("register"))

        cursor.execute(
            "INSERT INTO users (name,email,password) VALUES (%s,%s,%s)",
            (name,email,hashed_password)
        )

        db.commit()

        flash("Registration successful! Please login.","success")

        return redirect(url_for("login"))

    return render_template("register.html")

# =========================
# Login
# =========================

@app.route("/login", methods=["GET","POST"])
def login():

    if cursor is None:
        return render_template("login.html")

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        cursor.execute("SELECT * FROM users WHERE email=%s",(email,))
        user = cursor.fetchone()

        if user and check_password_hash(user[3],password):

            session["user_id"] = user[0]
            session["username"] = user[1]

            flash("Login successful!","success")

            return redirect(url_for("home"))

        else:

            flash("Invalid email or password","danger")

            return redirect(url_for("login"))

    return render_template("login.html")

# =========================
# Logout
# =========================

@app.route("/logout")
def logout():

    session.clear()

    flash("Logged out successfully","info")

    return redirect(url_for("login"))

# =========================
# Web Prediction
# =========================

@app.route("/predict_web", methods=["POST"])
def predict_web():

    if "user_id" not in session:
        return redirect(url_for("login"))

    file = request.files["file"]

    original_image = Image.open(file).convert("RGB")

    image_array = preprocess_image(original_image)

    prediction = model.predict(image_array)

    predicted_class = classes[np.argmax(prediction)]

    confidence = float(np.max(prediction))

    heatmap = generate_gradcam(model,image_array)

    heatmap_image_base64 = None

    if heatmap is not None:

        heatmap = cv2.resize(heatmap,original_image.size)

        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

        original_array = np.array(original_image)

        superimposed_img = cv2.addWeighted(original_array,0.6,heatmap,0.4,0)

        result_image = Image.fromarray(superimposed_img)

        buffered = BytesIO()

        result_image.save(buffered,format="PNG")

        heatmap_image_base64 = base64.b64encode(buffered.getvalue()).decode()

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=round(confidence*100,2),
        heatmap_image=heatmap_image_base64
    )

# =========================
# API Prediction (Flutter)
# =========================

@app.route("/api/predict", methods=["POST"])
def predict_api():

    if "file" not in request.files:
        return jsonify({"error":"No file uploaded"}),400

    file = request.files["file"]

    original_image = Image.open(file).convert("RGB")

    image_array = preprocess_image(original_image)

    prediction = model.predict(image_array)

    probabilities = prediction[0] * 100

    predicted_index = np.argmax(probabilities)

    predicted_class = classes[predicted_index]

    confidence = float(probabilities[predicted_index])

    all_probs = {
        classes[i]: round(float(probabilities[i]),2)
        for i in range(len(classes))
    }

    return jsonify({
        "prediction": predicted_class,
        "confidence": round(confidence,2),
        "all_probabilities": all_probs
    })

# =========================
# Run App
# =========================

if __name__ == "__main__":

    port = int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0",port=port)