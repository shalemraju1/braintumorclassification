import os
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_cors import CORS
import numpy as np
from PIL import Image
import psycopg
import tflite_runtime.interpreter as tflite
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import base64

app = Flask(__name__)
CORS(app)

app.secret_key = "brain_tumor_project_2026_secure_key"

db = None
cursor = None

try:

    DATABASE_URL = os.environ.get("DATABASE_URL")

    db = psycopg.connect(DATABASE_URL)

    cursor = db.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id SERIAL PRIMARY KEY,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        result TEXT,
        confidence FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

    db.commit()

except Exception as e:

    print("Database connection failed:", e)

    db = None
    cursor = None


interpreter = None
input_details = None
output_details = None

model_path = os.path.join("model", "brain_tumor_model.tflite")


def get_interpreter():

    global interpreter, input_details, output_details

    if interpreter is None:

        interpreter = tflite.Interpreter(model_path=model_path)

        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()

        output_details = interpreter.get_output_details()

    return interpreter


classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]


def preprocess_image(image):

    image = image.resize((150,150))

    image = np.array(image) / 255.0

    image = np.expand_dims(image, axis=0).astype(np.float32)

    return image


def generate_heatmap(image):

    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    _, buffer = cv2.imencode(".jpg", overlay)

    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    return heatmap_base64


@app.route("/")
def home():

    if "user_id" not in session:

        return redirect(url_for("login"))

    return render_template("index.html")


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

            flash("Login successful","success")

            return redirect(url_for("home"))

        else:

            flash("Invalid email or password","danger")

            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/api/register", methods=["POST"])
def register_api():

    data = request.get_json()

    name = data["name"]

    email = data["email"]

    password = data["password"]

    hashed_password = generate_password_hash(password)

    cursor.execute("SELECT * FROM users WHERE email=%s",(email,))

    user = cursor.fetchone()

    if user:

        return jsonify({"error":"Email already exists"}),400

    cursor.execute(
        "INSERT INTO users (name,email,password) VALUES (%s,%s,%s)",
        (name,email,hashed_password)
    )

    db.commit()

    return jsonify({"message":"Registration successful"})


@app.route("/api/login", methods=["POST"])
def login_api():

    data = request.get_json()

    email = data["email"]

    password = data["password"]

    cursor.execute("SELECT * FROM users WHERE email=%s",(email,))

    user = cursor.fetchone()

    if user and check_password_hash(user[3],password):

        return jsonify({
            "message":"Login successful",
            "user_id":user[0],
            "username":user[1]
        })

    return jsonify({"error":"Invalid email or password"}),401


@app.route("/logout")
def logout():

    session.clear()

    flash("Logged out successfully","info")

    return redirect(url_for("login"))


@app.route("/predict_web", methods=["POST"])
def predict_web():

    if "file" not in request.files:

        flash("No file uploaded","danger")

        return redirect(url_for("home"))

    file = request.files["file"]

    image = Image.open(file).convert("RGB")

    processed_image = preprocess_image(image)

    interpreter = get_interpreter()

    interpreter.set_tensor(input_details[0]["index"], processed_image)

    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])[0]

    probabilities = prediction * 100

    predicted_index = np.argmax(probabilities)

    predicted_class = classes[predicted_index]

    confidence = float(probabilities[predicted_index])

    if cursor is not None and "user_id" in session:

        cursor.execute(
            """
            INSERT INTO predictions (user_id, result, confidence)
            VALUES (%s,%s,%s)
            """,
            (session["user_id"], predicted_class, confidence)
        )

        db.commit()

    return render_template(
        "index.html",
        prediction = predicted_class,
        confidence = round(confidence,2)
    )


@app.route("/api/predict", methods=["POST"])
def predict_api():

    if "file" not in request.files:

        return jsonify({"error":"No file uploaded"}),400

    file = request.files["file"]

    image = Image.open(file).convert("RGB")

    image_array = preprocess_image(image)

    interpreter = get_interpreter()

    interpreter.set_tensor(input_details[0]["index"], image_array)

    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])[0]

    probabilities = prediction * 100

    predicted_index = np.argmax(probabilities)

    predicted_class = classes[predicted_index]

    confidence = float(probabilities[predicted_index])

    all_probs = {
        classes[i]: round(float(probabilities[i]),2)
        for i in range(len(classes))
    }

    heatmap = generate_heatmap(image)

    return jsonify({
        "prediction": predicted_class,
        "confidence": round(confidence,2),
        "all_probabilities": all_probs,
        "heatmap": heatmap
    })


@app.route("/api/history")
def get_history():

    if cursor is None:

        return jsonify([])

    cursor.execute(
        """
        SELECT result, confidence, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT 20
        """
    )

    rows = cursor.fetchall()

    history = []

    for r in rows:

        history.append({
            "result": r[0],
            "confidence": r[1],
            "date": str(r[2])
        })

    return jsonify(history)


if __name__ == "__main__":

    port = int(os.environ.get("PORT",10000))

    app.run(host="0.0.0.0", port=port)