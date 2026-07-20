from __future__ import annotations

import base64
import uuid
from pathlib import Path

from werkzeug.utils import secure_filename

from .config import HEATMAP_DIR, UPLOAD_DIR
from .database import ensure_database_schema, get_db_connection


def to_public_static_url(stored_path: str | None) -> str | None:
    if not stored_path:
        return None

    stored_path = stored_path.replace("\\", "/").lstrip("/")
    return f"/static/{stored_path}"


def save_uploaded_image(uploaded_file) -> tuple[str, str]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    original_name = secure_filename(uploaded_file.filename or "scan.jpg")
    suffix = Path(original_name).suffix.lower() or ".jpg"

    saved_name = f"{uuid.uuid4().hex}{suffix}"
    saved_path = UPLOAD_DIR / saved_name

    uploaded_file.save(saved_path)

    return saved_path.as_posix(), f"uploads/{saved_name}"


def save_heatmap(heatmap_base64: str) -> tuple[str, str]:
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    saved_name = f"heatmap_{uuid.uuid4().hex}.jpg"
    saved_path = HEATMAP_DIR / saved_name

    saved_path.write_bytes(base64.b64decode(heatmap_base64))

    return saved_path.as_posix(), f"heatmaps/{saved_name}"


def save_report(
    user_id: int,
    prediction: str,
    confidence: float,
    risk_level: str,
    image_path: str | None,
    heatmap_path: str | None,
) -> int:
    with get_db_connection() as connection:
        ensure_database_schema(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO reports (
                    user_id,
                    prediction,
                    confidence,
                    risk_level,
                    image_path,
                    heatmap_path
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (user_id, prediction, confidence, risk_level, image_path, heatmap_path),
            )
            report_id = cursor.fetchone()[0]
        connection.commit()
    return report_id


def fetch_history(user_id: int, limit: int | None = None) -> list[dict]:
    query = """
        SELECT
            reports.id,
            users.name,
            reports.prediction,
            reports.confidence,
            reports.risk_level,
            reports.image_path,
            reports.heatmap_path,
            reports.created_at
        FROM reports
        LEFT JOIN users ON users.id = reports.user_id
        WHERE reports.user_id = %s
        ORDER BY reports.created_at DESC, reports.id DESC
    """
    params = [user_id]
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    with get_db_connection() as connection:
        ensure_database_schema(connection)
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    history = []
    for row in rows:
        history.append(
            {
                "id": row[0],
                "patient_name": row[1],
                "prediction": row[2],
                "confidence": float(row[3]),
                "risk_level": row[4],
                "image_path": row[5],
                "heatmap_path": row[6],
                "image_url": to_public_static_url(row[5]),
                "heatmap_url": to_public_static_url(row[6]),
                "created_at": row[7],
                "report_url": f"/report/{row[0]}",
            }
        )
    return history


def fetch_report(report_id: int, user_id: int | None = None) -> dict | None:
    query = """
        SELECT
            reports.id,
            reports.user_id,
            users.name,
            users.email,
            reports.prediction,
            reports.confidence,
            reports.risk_level,
            reports.image_path,
            reports.heatmap_path,
            reports.created_at
        FROM reports
        LEFT JOIN users ON users.id = reports.user_id
        WHERE reports.id = %s
    """
    params = [report_id]

    if user_id is not None:
        query += " AND reports.user_id = %s"
        params.append(user_id)

    with get_db_connection() as connection:
        ensure_database_schema(connection)
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()

    if not row:
        return None

    return {
        "id": row[0],
        "user_id": row[1],
        "patient_name": row[2],
        "patient_email": row[3],
        "prediction": row[4],
        "confidence": float(row[5]),
        "risk_level": row[6],
        "image_path": row[7],
        "heatmap_path": row[8],
        "image_url": to_public_static_url(row[7]),
        "heatmap_url": to_public_static_url(row[8]),
        "created_at": row[9],
        "report_url": f"/report/{row[0]}",
        "pdf_url": f"/report/{row[0]}/pdf",
    }
