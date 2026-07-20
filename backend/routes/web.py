from __future__ import annotations

from io import BytesIO
from pathlib import Path
from flask import current_app
from flask import Response


from flask import Blueprint, abort, flash, redirect, render_template, request, session, url_for
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash

from helpers.database import ensure_database_schema, get_db_connection
from helpers.prediction import run_prediction
from helpers.storage import (
    fetch_history,
    fetch_report,
    save_heatmap,
    save_report,
    save_uploaded_image,
    to_public_static_url,
)

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image as ReportLabImage
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    REPORTLAB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    REPORTLAB_AVAILABLE = False


web_bp = Blueprint("web", __name__)


def _require_login():
    if "user_id" not in session:
        return redirect(url_for("web.login"))
    return None


def _build_recommendations(prediction: str, risk_level: str) -> list[str]:
    base_recommendations = [
        "Review the scan with a qualified radiologist before making any decision.",
        "Use the result as a clinical support tool, not a standalone diagnosis.",
    ]

    if risk_level == "High":
        return [
            "Arrange urgent specialist review and correlate with clinical symptoms.",
            "Consider follow-up imaging and multidisciplinary assessment.",
            *base_recommendations,
        ]

    if risk_level == "Medium":
        return [
            "Schedule a timely review with neurology or oncology.",
            "Compare with prior studies if available.",
            *base_recommendations,
        ]

    return [
        "Continue routine clinical monitoring if symptoms are stable.",
        "Escalate to a specialist if symptoms change or worsen.",
        *base_recommendations,
    ]


def _dashboard_stats(user_id: int) -> dict:
    with get_db_connection() as connection:
        ensure_database_schema(connection)
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM reports WHERE user_id = %s", (user_id,))
            total_reports = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM reports WHERE user_id = %s AND risk_level = 'High'",
                (user_id,),
            )
            high_risk = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM reports WHERE user_id = %s AND risk_level = 'Medium'",
                (user_id,),
            )
            medium_risk = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM reports WHERE user_id = %s AND risk_level = 'Low'",
                (user_id,),
            )
            low_risk = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM reports
                WHERE user_id = %s
                AND created_at >= NOW() - INTERVAL '7 days'
                """,
                (user_id,),
            )
            recent_activity = cursor.fetchone()[0]

    return {
        "total_reports": total_reports,
        "high_risk": high_risk,
        "medium_risk": medium_risk,
        "low_risk": low_risk,
        "recent_activity": recent_activity,
    }


def _predict_context(user_id: int) -> dict:
    return {
        "stats": _dashboard_stats(user_id),
        "recent_reports": fetch_history(user_id, limit=6),
        "username": session.get("username"),
    }


def _render_predict_page(user_id: int, prediction_context: dict | None = None):
    context = _predict_context(user_id)
    if prediction_context:
        context.update(prediction_context)

    return render_template("predict.html", **context)


@web_bp.route("/")
def home():
    login_redirect = _require_login()
    if login_redirect:
        return login_redirect

    user_id = session["user_id"]
    return render_template(
        "index.html",
        stats=_dashboard_stats(user_id),
        recent_reports=fetch_history(user_id, limit=6),
        username=session.get("username"),
    )


@web_bp.route("/predict", methods=["GET", "POST"])
def predict():
    login_redirect = _require_login()
    if login_redirect:
        return login_redirect

    if request.method == "GET":
        return _render_predict_page(session["user_id"])

    uploaded_file = request.files.get("file")
    if not uploaded_file or not uploaded_file.filename:
        flash("Please select an MRI image.", "danger")
        return _render_predict_page(session["user_id"])

    try:
        uploaded_path, image_db_path = save_uploaded_image(uploaded_file)
        image = Image.open(uploaded_path).convert("RGB")
        prediction_result = run_prediction(image)
        heatmap_path, heatmap_db_path = save_heatmap(prediction_result["heatmap_base64"])
        report_id = save_report(
                    user_id=session["user_id"],
                    prediction=prediction_result["prediction"],
                    confidence=round(prediction_result["confidence"], 2),
                    risk_level=prediction_result["risk_level"],
                    image_path=image_db_path,
                    heatmap_path=heatmap_db_path,
                )

        return _render_predict_page(
            session["user_id"],
            {
                "prediction": prediction_result["prediction"],
                "confidence": round(prediction_result["confidence"], 2),
                "risk_level": prediction_result["risk_level"],
                "all_probabilities": prediction_result["all_probabilities"],
                "heatmap_image": prediction_result["heatmap_base64"],
                "uploaded_image_url": to_public_static_url(image_db_path),
                "report_id": report_id,
                "report_url": url_for("web.report", report_id=report_id),
                "pdf_url": url_for("web.report_pdf", report_id=report_id),
                "recommendations": _build_recommendations(
                    prediction_result["prediction"], prediction_result["risk_level"]
                ),
            },
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise


@web_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("web.register"))

        hashed_password = generate_password_hash(password)

        try:
            with get_db_connection() as connection:
                ensure_database_schema(connection)
                with connection.cursor() as cursor:
                    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
                    existing_user = cursor.fetchone()
                    if existing_user:
                        flash("Email already registered.", "danger")
                        return redirect(url_for("web.register"))

                    cursor.execute(
                        """
                        INSERT INTO users (name, email, password)
                        VALUES (%s, %s, %s)
                        """,
                        (name, email, hashed_password),
                    )
                connection.commit()

            flash("Registration successful. Please log in.", "success")
            return redirect(url_for("web.login"))
        except Exception as exc:  # pragma: no cover - database failures are environment specific
            print("Registration error:", exc)
            flash("Unable to register right now. Please try again.", "danger")
            return redirect(url_for("web.register"))

    return render_template("register.html", username=session.get("username"))


@web_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Email and password are required.", "danger")
            return redirect(url_for("web.login"))

        try:
            with get_db_connection() as connection:
                ensure_database_schema(connection)
                with connection.cursor() as cursor:
                    cursor.execute("SELECT id, name, password FROM users WHERE email = %s", (email,))
                    user = cursor.fetchone()

            if user and check_password_hash(user[2], password):
                session["user_id"] = user[0]
                session["username"] = user[1]
                flash("Login successful.", "success")
                return redirect(url_for("web.home"))

            flash("Invalid email or password.", "danger")
            return redirect(url_for("web.login"))
        except Exception as exc:  # pragma: no cover - database failures are environment specific
            print("Login error:", exc)
            flash("Unable to log in right now. Please try again.", "danger")
            return redirect(url_for("web.login"))

    return render_template("login.html", username=session.get("username"))


@web_bp.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("web.login"))


@web_bp.route("/predict_web", methods=["GET", "POST"])
def predict_web():
    login_redirect = _require_login()
    if login_redirect:
        return login_redirect

    if request.method == "GET":
        return redirect(url_for("web.predict"))

    return predict()


@web_bp.route("/history")
def history():
    login_redirect = _require_login()
    if login_redirect:
        return login_redirect

    return render_template(
        "history.html",
        reports=fetch_history(session["user_id"]),
        username=session.get("username"),
    )


@web_bp.route("/report/<int:report_id>")
def report(report_id: int):
    login_redirect = _require_login()
    if login_redirect:
        return login_redirect

    report_data = fetch_report(report_id, user_id=session["user_id"])
    if not report_data:
        abort(404)

    return render_template(
        "report.html",
        report=report_data,
        recommendations=_build_recommendations(report_data["prediction"], report_data["risk_level"]),
        username=session.get("username"),
    )


@web_bp.route("/report/<int:report_id>/pdf")
def report_pdf(report_id: int):
    login_redirect = _require_login()
    if login_redirect:
        return login_redirect

    report_data = fetch_report(report_id, user_id=session["user_id"])
    if not report_data:
        abort(404)

    if not REPORTLAB_AVAILABLE:
        flash(
            "PDF support is not installed on this environment. Use Print to save as PDF.",
            "warning",
        )
        return redirect(url_for("web.report", report_id=report_id))

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()
    # Absolute path to the backend/static folder
    static_dir = Path(current_app.root_path) / "static"

    title_style = ParagraphStyle(
    "TitleStyle",
    parent=styles["Title"],
    alignment=1,
    fontSize=22,
    textColor=colors.HexColor("#0F172A"),
    spaceAfter=8,
        )

    subtitle_style = ParagraphStyle(
            "SubtitleStyle",
            parent=styles["Heading2"],
            alignment=1,
            textColor=colors.HexColor("#2563EB"),
            spaceAfter=16,
        )

    story = []

    story.append(Paragraph("Brain Tumor Detection Report", title_style))
    story.append(Paragraph("AI-Based MRI Analysis", subtitle_style))

    summary = Table(
            [
                ["Report ID", str(report_id)],
                ["Generated On", str(report_data["created_at"])],
                ["Prediction", report_data["prediction"]],
                ["Confidence", f"{report_data['confidence']:.2f}%"],
                ["Risk Level", report_data["risk_level"]],
            ],
            colWidths=[5 * cm, 10 * cm],
        )

    summary.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

    story.append(summary)
    story.append(Spacer(1, 0.5 * cm))

        # Original MRI
    if report_data["image_path"]:
            image_path = static_dir / report_data["image_path"]

            if image_path.exists():
                story.append(Paragraph("Original MRI Scan", styles["Heading2"]))
                story.append(
                    ReportLabImage(
                        str(image_path),
                        width=14 * cm,
                        height=9 * cm,
                    )
                )
                story.append(Spacer(1, 0.5 * cm))

        # Heatmap
    if report_data["heatmap_path"]:
            heatmap_path = static_dir / report_data["heatmap_path"]

            if heatmap_path.exists():
                story.append(Paragraph("AI Generated Heatmap", styles["Heading2"]))
                story.append(
                    ReportLabImage(
                        str(heatmap_path),
                        width=14 * cm,
                        height=9 * cm,
                    )
                )
                story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Disclaimer", styles["Heading2"]))

    story.append(
            Paragraph(
                "This report was generated using an AI-assisted brain tumor detection model. "
                "The results are intended only as a clinical decision-support aid and should "
                "always be verified by a qualified medical professional.",
                styles["BodyText"],
            )
        )

    document.build(story)

    pdf = buffer.getvalue()
    buffer.close()

    response = Response(pdf, mimetype="application/pdf")
    response.headers[
            "Content-Disposition"
        ] = f'attachment; filename="brain_tumor_report_{report_id}.pdf"'

    return response