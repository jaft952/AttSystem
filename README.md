# 🎓 AttSystem — Advanced CBIR Face Attendance System

A Flask-based attendance prototype powered by three Content-Based Image Retrieval (CBIR) pipelines. It features live camera streaming, runtime model switching, MediaPipe face detection, and Euclidean distance-based deep metric learning for highly accurate face recognition.

---

## ✨ Features

- 📷 **Live camera stream** via Flask MJPEG endpoint with multiprocessing backend.
- 🧠 **Triple CBIR face recognition** with dynamic pre-processing pipelines.
- 📐 **Euclidean Distance Metric** (`Sim = 1.0 - Euclidean`) optimized for dlib's ResNet-34 embeddings.
- 👤 **BlazeFace Detection** utilizing Google's MediaPipe for robust, lightning-fast face localization.
- 🔀 **Runtime switching** seamlessly between CBIR Method 1, 2, and 3.
-📊 **Rigorous Evaluation** with FAR/FRR threshold tuning, Equal Error Rate (EER) analysis, and Genuine vs. Impostor distributions.
- 🌐 **Web UI** for attendance and developer monitoring.

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

# Install dependencies (Includes OpenCV, MediaPipe, dlib, SciPy, etc.)
python -m pip install -r requirement.txt
```

### 2. Prepare Raw Dataset

Place your raw images into `data/face/<person_name>/`:

```
data/
└── face/
    ├── benjamin/
    │   └── *.jpg
    └── chern_tak/
        └── *.jpg
```

> **Tip:** Use one folder per identity. Include varied lighting conditions and angles for best results.

### 3. Standardize + Augment Dataset

Run `ml/augmentation.ipynb` first.

- It uses **Albumentations** to generate realistic environmental variations (blur, noise, lighting changes, perspective shifts).
- Generates up to 8 augmented variations per original image, all saved inside the same `data/face` subfolders.

### 4. Run the ML Pipelines (CBIR Methods)

Open and run these notebooks to build the reference galleries (indexes). Each uses different image enhancement strategies before passing the ROI to the ResNet-34 embedding model:

| Notebook                | Preprocessing Strategy                                                                                   | Output Artifacts                                               |
| ----------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `ml/cbir_method1.ipynb` | **Baseline:** Simple standard resizing to 128x128.                                                       | `index/cbir_method1_index.npz`, `_meta.json`                   |
| `ml/cbir_method2.ipynb` | **Color Enhanced:** HSV conversion + CLAHE on the V-channel for dynamic contrast.                        | `index/cbir_method2_index.npz`, `_meta.json`                   |
| `ml/cbir_method3.ipynb` | **Aggressive:** Grayscale + Histogram Equalization + Bilateral Filter + Unsharp Masking.                 | `index/cbir_method3_index.npz`, `_meta.json`                   |

*(All notebooks automatically use MediaPipe BlazeFace to tightly crop the face before applying these filters).*

### 5. Run Extreme Evaluation

Run `ml/evaluation.ipynb`.

- Cross-evaluates all 3 CBIR methods on a hold-out test set.
- Utilizes `scipy.spatial.distance.cdist(metric="euclidean")` to calculate true spatial separation instead of Cosine (which compresses unnormalized vectors).
- Generates detailed academic charts: **FAR vs FRR Threshold Tuning**, **Genuine vs. Impostor Distributions**, and **Confusion Matrices**.

### 6. Launch the Web App

Once the `.npz` indexes and `.json` metadatas are generated, start the backend server:

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

| Page            | URL                       |
| --------------- | ------------------------- |
| Attendance      | http://127.0.0.1:5000     |
| Developer Tools | http://127.0.0.1:5000/dev |

---

## 🧪 Developer Tools

### Step-by-Step

1. Navigate to `/dev` and ensure the camera stream is live.
2. Switch runtime model between **CBIR Method 1, 2, and 3** via the dropdown.
3. Review live metrics such as accepted similarity, identity, bounding box coordinates, and threshold demarcations.

---

## 🛠️ Troubleshooting

<details>
<summary><strong>App startup issues / Address already in use</strong></summary>

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
<summary><strong>Camera not working / Freezing</strong></summary>

Ensure the webcam is not being used by another application (like Zoom or Teams). The `camera_service.py` runs in a separate multiprocessing loop—if it crashes, check the terminal for `cv2.VideoCapture` errors.

</details>

---

## 📁 Key Paths

| Path                                | Description                     |
| ----------------------------------- | ------------------------------- |
| `main.py`                           | App entry point                 |
| `service/recognition_service.py`    | Runtime recognition backend     |
| `service/camera_service.py`         | Multiprocessing camera stream   |
| `ml/augmentation.ipynb`             | Albumentations pipeline         |
| `ml/cbir_method*.ipynb`             | Training / Indexing notebooks   |
| `ml/evaluation.ipynb`               | Academic Evaluation charts      |
| `index/`                            | Generated NPZ/JSON model data   |
