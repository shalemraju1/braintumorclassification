from __future__ import annotations

from flask import Flask
from flask_cors import CORS

from helpers.config import MAX_CONTENT_LENGTH, PORT, SECRET_KEY
from helpers.database import bootstrap_database
from routes.api import api_bp
from routes.web import web_bp


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(
        SECRET_KEY=SECRET_KEY,
        MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    CORS(app)
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp)

    try:
        with app.app_context():
            bootstrap_database()
    except Exception as exc:  # pragma: no cover - depends on deployment env
        app.logger.warning("Database bootstrap skipped: %s", exc)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
