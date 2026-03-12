from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import psycopg
from werkzeug.security import generate_password_hash, check_password_hash
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =========================
# Flask Setup
# =========================

app = Flask(__name__)
CORS(app)

app.secret_key = "brain_tumor_project_2026_secure_key"

# =========================
# PostgreSQL Database
# =========================

db = None
cursor = None

try:
    DATABASE_URL = os.environ.get("DATABASE_URL")

    db = psycopg.connect(DATABASE_URL)
    cursor = db.cursor()

    print("PostgreSQL Connected Successfully")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id SERIAL PRIMARY KEY,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)

    db.commit()

except Exception as e:
    print("Database connection failed:", e)
    db = None
    cursor = None


# =========================
# Load ML Model
# =========================

model_path = os.path.join("model", "brain_tumor_cnn_final.h5")

model = tf.keras.models.load_model(model_path)

dummy_input = tf.zeros((1,150,150,3))
model(dummy_input)

print("Model Loaded Successfully")

# =========================
# Class Labels
# =========================

classes = ["Glioma","Meningioma","Pituitary","No Tumor"]


# =========================
# Image Preprocessing
# =========================

def preprocess_image(image):

    image = image.resize((150,150))
    image = np.array(image)/255.0
    image = np.expand_dims(image,axis=0)

    return image


# =========================
# Home Page
# =========================

@app.route("/")
def home():

    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template("index.html")


# =========================
# Register
# =========================

@app.route("/register", methods=["GET","POST"])
def register():

    if cursor is None:
        return "Database not connected"

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

        cursor.execute(
            "SELECT * FROM users WHERE email=%s",
            (email,)
        )

        user = cursor.fetchone()

        if user and check_password_hash(user[3],password):

            session["user_id"] = user[0]
            session["username"] = user[1]

            flash("Login successful","success")

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

    if "file" not in request.files:

        flash("No file uploaded","danger")
        return redirect(url_for("home"))

    file = request.files["file"]

    image = Image.open(file).convert("RGB")

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)

    probabilities = prediction[0] * 100

    predicted_index = np.argmax(probabilities)

    predicted_class = classes[predicted_index]

    confidence = float(probabilities[predicted_index])

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=round(confidence,2)
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