# 🎓 AttSystem — Dual-CBIR Face Attendance System

A Flask-based attendance prototype powered by two CBIR pipelines (different preprocessing methods), with live camera streaming and runtime model switching.

---

## ✨ Features

- 📷 **Live camera stream** via Flask MJPEG endpoint (OpenCV backend)
- 🧠 **Dual CBIR face recognition** with two preprocessing methods
- 🔀 **Runtime switching** between CBIR Method 1 and CBIR Method 2
- ⚡ **Notebook-based retraining** to refresh each method's index artifacts
- 🌐 **Web UI** for attendance and developer monitoring

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

Place your images into `data/face/<person_name>/`:

```
data/
└── face/
    ├── benjamin/
    │   └── *.jpg
    └── chern_tak/
        └── *.jpg
```

> **Tip:** Use one folder per identity. Include varied lighting conditions and angles for best results.

### 3. Run the ML Pipelines (CBIR Method 1 + Method 2)

Open and run these notebooks:

| Step | Notebook                | Output                                                           |
| ---- | ----------------------- | ---------------------------------------------------------------- |
| 1    | `ml/cbir_method1.ipynb` | `models/cbir_method1_index.npz`, `models/cbir_method1_meta.json` |
| 2    | `ml/cbir_method2.ipynb` | `models/cbir_method2_index.npz`, `models/cbir_method2_meta.json` |

After both runs, `config/realtime_model_config.json` can switch between `cbir_method1` and `cbir_method2` at runtime.

#### Preprocessing Used By Each CBIR Method

Both CBIR notebooks start from a grayscale face image, detect the largest face ROI with a Haar cascade, add a small padding around the face, and resize the result to 128 x 128 before extracting the embedding.

| Method        | Preprocessing summary                                                                               |
| ------------- | --------------------------------------------------------------------------------------------------- |
| CBIR Method 1 | Uses CLAHE to boost local contrast, then applies a light Gaussian blur before resizing.             |
| CBIR Method 2 | Uses global histogram equalization, then applies stronger denoising and sharpening before resizing. |

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
2. Switch runtime model between **CBIR Method 1** and **CBIR Method 2**.
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

| Path                                | Description                     |
| ----------------------------------- | ------------------------------- |
| `main.py`                           | App entry point                 |
| `service/recognition_service.py`    | Runtime recognition backend     |
| `service/camera_service.py`         | Camera stream service           |
| `presentation/views/dev.html`       | Developer tools page template   |
| `scripts/dev.js`                    | Developer tools UI script       |
| `ml/cbir_method1.ipynb`             | CBIR method 1 training notebook |
| `ml/cbir_method2.ipynb`             | CBIR method 2 training notebook |
| `models/cbir_method1_index.npz`     | Runtime CBIR index (method 1)   |
| `models/cbir_method2_index.npz`     | Runtime CBIR index (method 2)   |
| `config/realtime_model_config.json` | Runtime model configuration     |
