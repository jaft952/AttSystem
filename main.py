import json
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, Response, jsonify, render_template, request, send_from_directory, stream_with_context

from service import recognition_service


PROJECT_ROOT = Path(__file__).resolve().parent
VIEWS_ROOT = PROJECT_ROOT / "presentation" / "views"
UI_ROOT = PROJECT_ROOT / "presentation" / "ui"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
ATTENDANCE_ROOT = PROJECT_ROOT / "data" / "attendance"
ATTENDANCE_LOCK = threading.Lock()

ATTENDANCE_ROOT.mkdir(parents=True, exist_ok=True)

APP = Flask(__name__, template_folder=str(VIEWS_ROOT), static_folder=str(UI_ROOT), static_url_path="/ui")


@APP.get("/scripts/<path:filename>")
def script_asset(filename: str):
    return send_from_directory(str(SCRIPTS_ROOT), filename)


@APP.get("/")
def index():
    return render_template(
        "index.html",
        app_name="AttSystem",
        today_date=datetime.now().strftime("%Y-%m-%d"),
    )


def _today_key() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _attendance_file_path(day_key: str) -> Path:
    return ATTENDANCE_ROOT / f"{day_key}.json"


def _load_attendance(day_key: str):
    file_path = _attendance_file_path(day_key)
    labels = sorted(recognition_service.get_label_names())
    payload = {"date": day_key, "records": {}}

    if file_path.exists():
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
                if isinstance(loaded, dict):
                    payload = loaded
        except Exception:
            payload = {"date": day_key, "records": {}}

    payload["date"] = day_key
    records = payload.get("records")
    if not isinstance(records, dict):
        records = {}
    payload["records"] = records

    for name in labels:
        row = records.get(name)
        if not isinstance(row, dict):
            row = {}
        row.setdefault("present", False)
        row.setdefault("first_seen_at", None)
        row.setdefault("last_seen_at", None)
        row.setdefault("best_confidence", None)
        records[name] = row

    return payload, file_path, labels


def _default_attendance_row() -> dict:
    return {
        "present": False,
        "first_seen_at": None,
        "last_seen_at": None,
        "best_confidence": None,
    }


def _save_attendance(file_path: Path, payload: dict):
    tmp_path = file_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(file_path)


def _build_attendance_summary(payload: dict, labels: list[str]) -> dict:
    records = payload.get("records", {})
    present_people = []
    absent_people = []

    for name in labels:
        row = records.get(name, {})
        if row.get("present"):
            present_people.append(
                {
                    "name": name,
                    "first_seen_at": row.get("first_seen_at"),
                    "last_seen_at": row.get("last_seen_at"),
                    "best_confidence": row.get("best_confidence"),
                }
            )
        else:
            absent_people.append(name)

    return {
        "date": payload.get("date"),
        "total": len(labels),
        "present_count": len(present_people),
        "absent_count": len(absent_people),
        "present": present_people,
        "absent": absent_people,
    }


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@APP.get("/api/attendance/today")
def api_attendance_today():
    day_key = _today_key()
    with ATTENDANCE_LOCK:
        payload, _, labels = _load_attendance(day_key)
        return jsonify({"status": "ok", "attendance": _build_attendance_summary(payload, labels)})


@APP.post("/api/attendance/mark")
def api_attendance_mark():
    body = request.get_json(silent=True) or {}
    label = str(body.get("label", "")).strip()
    confidence = body.get("confidence")
    now_iso = datetime.now().isoformat(timespec="seconds")

    if not label:
        return jsonify({"error": "Missing label."}), 400

    day_key = _today_key()
    with ATTENDANCE_LOCK:
        payload, file_path, labels = _load_attendance(day_key)
        if label not in labels:
            return jsonify({"error": "Unknown label."}), 400

        row = payload["records"].setdefault(label, _default_attendance_row())

        row["present"] = True
        if not row.get("first_seen_at"):
            row["first_seen_at"] = now_iso
        row["last_seen_at"] = now_iso

        confidence_value = _to_float(confidence)
        if confidence_value is not None:
            previous_best = row.get("best_confidence")
            previous_best_value = _to_float(previous_best)
            if previous_best_value is None or confidence_value < previous_best_value:
                row["best_confidence"] = confidence_value

        _save_attendance(file_path, payload)
        summary = _build_attendance_summary(payload, labels)

    return jsonify({"status": "ok", "marked": label, "attendance": summary})


@APP.post("/api/attendance/unmark")
def api_attendance_unmark():
    body = request.get_json(silent=True) or {}
    label = str(body.get("label", "")).strip()

    if not label:
        return jsonify({"error": "Missing label."}), 400

    day_key = _today_key()
    with ATTENDANCE_LOCK:
        payload, file_path, labels = _load_attendance(day_key)
        if label not in labels:
            return jsonify({"error": "Unknown label."}), 400

        row = payload["records"].setdefault(label, _default_attendance_row())
        row["present"] = False
        row["first_seen_at"] = None
        row["last_seen_at"] = None
        row["best_confidence"] = None

        _save_attendance(file_path, payload)
        summary = _build_attendance_summary(payload, labels)

    return jsonify({"status": "ok", "unmarked": label, "attendance": summary})


@APP.get("/dev")
def developer_tools():
    return render_template("dev.html", **recognition_service.get_developer_tools_template_context("AttSystem"))


@APP.get("/api/health")
def health():
    return jsonify(recognition_service.get_health_payload())


@APP.post("/api/model/switch")
def api_model_switch():
    body = request.get_json(silent=True) or {}
    model_type = str(body.get("model_type", "")).strip().lower()
    
    if not model_type:
        return jsonify({"error": "Missing model_type parameter."}), 400
    
    try:
        result = recognition_service.switch_model(model_type)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@APP.get("/api/prediction/stream")
def api_prediction_stream():
    """SSE endpoint: pushes a prediction event each time inference completes.

    Replaces the client-side 450 ms REST poll with a push-based channel so
    the browser receives results within milliseconds of inference finishing.
    """
    def generate():
        for prediction in recognition_service.stream_predictions():
            if prediction is None:
                yield ": keepalive\n\n"
            else:
                yield f"data: {json.dumps(prediction)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@APP.get("/api/latest")
def api_latest():
    return jsonify(recognition_service.get_latest_payload())


@APP.get("/video_feed")
def video_feed():
    return Response(
        recognition_service.stream_frames(),
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
        return jsonify(recognition_service.predict_from_payload(image_data))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@APP.post("/api/camera/start")
def api_camera_start():
    try:
        return jsonify(recognition_service.start_camera())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@APP.post("/api/camera/stop")
def api_camera_stop():
    try:
        return jsonify(recognition_service.stop_camera())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
