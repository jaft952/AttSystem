# 🎓 AttSystem — LBPH Face Attendance System

A Flask-based attendance prototype powered by an LBPH face recognizer, with live camera streaming, reinforcement feedback, and runtime retraining.

---

## ✨ Features

- 📷 **Live camera stream** via Flask MJPEG endpoint (OpenCV backend)
- 🧠 **LBPH face recognition** with augmentation and multi-stage preprocessing
- 🔁 **Reinforcement feedback loop** — confirm or correct predictions on the fly
- ⚡ **Runtime retraining** without restarting the server
- 🌐 **Web UI** for attendance and training management

---

## 📋 Prerequisites

| Requirement | Details                                              |
| ----------- | ---------------------------------------------------- |
| OS          | Windows (recommended)                                |
| Python      | 3.11+                                                |
| Webcam      | Working USB or built-in camera                       |
| IDE         | VS Code + Jupyter extension (for pipeline notebooks) |

---

## 🚀 Quick Start

### 1. Environment Setup

```powershell
# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirement.txt
```

### 2. Prepare Raw Dataset

Place your images into `data/1_raw/<person_name>/`:

```
data/
└── 1_raw/
    ├── benjamin/
    │   └── *.jpg
    └── chern_tak/
        └── *.jpg
```

> **Tip:** Use one folder per identity. Include varied lighting conditions and angles for best results.

### 3. Run the ML Pipeline (Notebooks 1 → 3)

Open and run these notebooks **in order**:

| Step | Notebook                           | Output                                                    |
| ---- | ---------------------------------- | --------------------------------------------------------- |
| 1    | `ml/1_roi_pipeline.ipynb`          | ROI + preprocessed images (`data/2_*` through `data/6_*`) |
| 2    | `ml/2_augmentation_pipeline.ipynb` | Augmented final set (`data/7_*`)                          |
| 3    | `ml/3_training_pipeline.ipynb`     | LBPH models + runtime config (`models/`)                  |

After pipeline 3, you'll have:

```
models/
├── lbph_model.yml
├── lbph_with_aug.yml
├── lbph.yml                    ← used at runtime
└── realtime_model_config.json
```

### 4. Launch the Web App

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

| Page            | URL                       |
| --------------- | ------------------------- |
| Attendance      | http://127.0.0.1:5000     |
| Developer Tools | http://127.0.0.1:5000/dev |

> Also accessible on your local network at `http://192.168.1.82:5000`.

---

## 🧪 Developer Tools

### Step-by-Step

1. Navigate to `/dev` and ensure the camera stream is live.
2. Switch runtime model between **LBPH** and **CBIR**.
3. Review live metrics such as accepted rate, known-face rate, no-face rate, and frame count.

Reinforcement and retraining flow has been removed from the runtime app.

---

## 🛠️ Troubleshooting

<details>
<summary><strong>App startup issues</strong></summary>

Run only one server instance at a time. Start with:

```powershell
python main.py
```

Verify the health endpoint in another terminal:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5000/api/health | Select-Object -ExpandProperty Content
```

</details>

<details>
<summary><strong>Camera not working</strong></summary>

Ensure the webcam is not being used by another application. Restart the app and reopen `/dev`.

</details>

---

## 📁 Key Paths

| Path                                | Description                   |
| ----------------------------------- | ----------------------------- |
| `main.py`                           | App entry point               |
| `service/recognition_service.py`    | Runtime recognition backend   |
| `service/camera_service.py`         | Camera stream service         |
| `presentation/views/dev.html`       | Developer tools page template |
| `scripts/dev.js`                    | Developer tools UI script     |
| `models/lbph.yml`                   | Runtime LBPH model            |
| `models/realtime_model_config.json` | Runtime model configuration   |
