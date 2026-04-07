from pathlib import Path
from flask import Flask, Response, jsonify, render_template, request

from ml import reinforcement_pipeline as training_service


PROJECT_ROOT = Path(__file__).resolve().parent
VIEWS_ROOT = PROJECT_ROOT / "presentation" / "views"
UI_ROOT = PROJECT_ROOT / "presentation" / "ui"

APP = Flask(__name__, template_folder=str(VIEWS_ROOT), static_folder=str(UI_ROOT), static_url_path="/ui")


@APP.get("/")
def index():
    return render_template(
        "index.html",
        app_name="AttSystem",
    )


@APP.get("/training")
def training():
    return render_template("training.html", **training_service.get_training_template_context("AttSystem"))


@APP.get("/api/health")
def health():
    return jsonify(training_service.get_health_payload())


@APP.get("/api/latest")
def api_latest():
    return jsonify(training_service.get_latest_payload())


@APP.get("/video_feed")
def video_feed():
    return Response(
        training_service.stream_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@APP.get("/training_video_feed")
def training_video_feed():
    return Response(
        training_service.stream_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@APP.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "Missing image payload."}), 400

    try:
        return jsonify(training_service.predict_from_payload(image_data))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@APP.post("/api/feedback")
def api_feedback():
    payload = request.get_json(silent=True) or {}
    action = str(payload.get("action", "confirm")).strip().lower()
    selected_label = str(payload.get("label", "")).strip()

    try:
        return jsonify(training_service.save_feedback(action, selected_label))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@APP.post("/api/retrain")
def api_retrain():
    try:
        return jsonify(training_service.retrain_with_feedback_async())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@APP.get("/api/retrain/status")
def api_retrain_status():
    try:
        return jsonify(training_service.get_retrain_status())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@APP.post("/api/camera/start")
def api_camera_start():
    try:
        return jsonify(training_service.start_camera())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@APP.post("/api/camera/stop")
def api_camera_stop():
    try:
        return jsonify(training_service.stop_camera())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
