from __future__ import annotations

from flask import Blueprint, jsonify, request
from werkzeug.security import check_password_hash

from helpers.database import ensure_database_schema, get_db_connection
from helpers.prediction import run_prediction
from helpers.storage import fetch_history, save_heatmap, save_report, save_uploaded_image

from PIL import Image

api_bp = Blueprint("api", __name__)


@api_bp.route("/api/init-users", methods=["POST"])
def init_users_table_api():
    try:
        with get_db_connection() as connection:
            ensure_database_schema(connection)
        return jsonify({"message": "Users table initialized"})
    except Exception as exc:  # pragma: no cover - operational endpoint
        print("Error in /api/init-users:", exc)
        return jsonify({"error": "Internal server error"}), 500


@api_bp.route("/api/init-db", methods=["GET"])
def init_db_api():
    try:
        with get_db_connection() as connection:
            ensure_database_schema(connection)
        return jsonify({"message": "Reports table is ready"}), 200
    except Exception as exc:  # pragma: no cover - operational endpoint
        print("Error in /api/init-db:", exc)
        return jsonify({"error": "Database initialization failed"}), 500


@api_bp.route("/api/login", methods=["POST"])
def login_api():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    try:
        with get_db_connection() as connection:
            ensure_database_schema(connection)
            with connection.cursor() as cursor:
                cursor.execute("SELECT id, name, password FROM users WHERE email = %s", (email,))
                user = cursor.fetchone()

        if user and check_password_hash(user[2], password):
            return jsonify({"message": "Login successful", "user_id": user[0], "username": user[1]}), 200

        return jsonify({"error": "Invalid email or password"}), 401
    except Exception as exc:  # pragma: no cover - database failures are environment specific
        print("API login error:", exc)
        return jsonify({"error": "Database connection failed"}), 500


@api_bp.route("/api/predict", methods=["POST"])
def predict_api():
    uploaded_file = request.files.get("file")
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({"error": "No file uploaded"}), 400

    user_id = request.form.get("user_id", type=int)
    if user_id is None:
        return jsonify({"error": "Missing required field: user_id"}), 400

    try:
        uploaded_path, image_db_path = save_uploaded_image(uploaded_file)
        image = Image.open(uploaded_path).convert("RGB")
        prediction_result = run_prediction(image)
        heatmap_path, heatmap_db_path = save_heatmap(
    prediction_result["heatmap_base64"]
)

        save_report(
            user_id=user_id,
            prediction=prediction_result["prediction"],
            confidence=round(prediction_result["confidence"], 2),
            risk_level=prediction_result["risk_level"],
            image_path=image_db_path,
            heatmap_path=heatmap_db_path,
        )
    except Exception as exc:  # pragma: no cover - image/model/database failures are environment specific
        print("API prediction error:", exc)
        return jsonify({"error": "Unable to process prediction"}), 500

    return jsonify(
        {
            "prediction": prediction_result["prediction"],
            "confidence": round(prediction_result["confidence"], 2),
            "risk_level": prediction_result["risk_level"],
            "all_probabilities": prediction_result["all_probabilities"],
            "heatmap": prediction_result["heatmap_base64"],
        }
    )


@api_bp.route("/api/history")
def history_api():
    user_id = request.args.get("user_id", type=int)
    if user_id is None:
        return jsonify({"error": "Missing required query parameter: user_id"}), 400

    try:
        history = fetch_history(user_id)
    except Exception as exc:  # pragma: no cover - database failures are environment specific
        print("API history error:", exc)
        return jsonify({"error": "Database error while fetching history"}), 500

    return jsonify(
        [
            {
                "id": item["id"],
                "prediction": item["prediction"],
                "confidence": item["confidence"],
                "risk_level": item["risk_level"],
                "created_at": item["created_at"].isoformat() if item["created_at"] else None,
            }
            for item in history
        ]
    ), 200
