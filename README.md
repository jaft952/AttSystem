# AttSystem (LBPH Face Attendance)

This project is a Flask-based attendance prototype with an LBPH face recognizer.

Current architecture:

- Camera stream: backend (OpenCV) via Flask MJPEG endpoint
- Prediction + feedback + retraining: backend Python services
- UI: web frontend in presentation/views and presentation/ui
- Runtime model files: models/lbph_final.yml + models/realtime_model_config.json

## Project Objective

- Build a face recognition attendance flow using LBPH.
- Improve robustness under different lighting conditions.
- Support reinforcement feedback and runtime retraining.

## Prerequisites

- Windows (recommended for current camera capture path)
- Python 3.11+
- A working webcam
- VS Code with Jupyter extension (for pipeline notebooks)

## 1) Kick Start: Environment Setup

From project root:

```powershell
# create venv (if not created yet)
python -m venv .venv

# activate venv
.\.venv\Scripts\Activate.ps1

# install dependencies
python -m pip install -r requirement.txt
```

## 2) Prepare Raw Dataset (Baseline Input)

Put your raw images into:

- data/1_raw/<person_name>/\*.jpg|png

Example:

- data/1_raw/harry/\*.jpg
- data/1_raw/daniel/\*.jpg

Important:

- Keep one folder per identity.
- Use clear face images with varied lighting and angles.

## 3) Run ML Pipeline 1 -> 3 (Baseline + Final Runtime Model)

Open and run these notebooks in order:

1. ml/1_roi_pipeline.ipynb
2. ml/2_augmentation_pipeline.ipynb
3. ml/3_training_pipeline.ipynb

Note: the file name is currently spelled 3_traininig_pipeline.ipynb.

### Pipeline 1 output (ROI + preprocessing)

Expected folders produced/updated:

- data/2_standarized
- data/3_train
- data/3_test
- data/4_processed_train
- data/4_processed_test
- data/5_roi_train
- data/5_roi_test
- data/6_enhanced_roi_train
- data/6_enhanced_roi_test

### Pipeline 2 output (augmentation + final processed set)

Expected folders produced/updated:

- data/7_final_processed
- data/7_augmented

### Pipeline 3 output (LBPH models + runtime config)

Expected files produced/updated:

- models/lbph_model.yml
- models/lbph_with_aug.yml
- models/lbph_final.yml
- models/realtime_model_config.json

This is your initial baseline model path before reinforcement.

## 4) Run the Web App

From project root:

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

Open:

- http://127.0.0.1:5000
  or
- http://192.168.1.82:5000

Training/reinforcement page:

- http://127.0.0.1:5000/training
  or
- http://192.168.1.82:5000/training

## 5) Reinforcement Training (Feedback Loop)

### A) Start training UI

1. Go to /training page.
2. Ensure camera stream is running.
3. Watch live prediction status.

### B) Save feedback samples

Use either:

- Confirm prediction (when predicted identity is correct)
- Correct label (select the right label and save)

Feedback data is stored in:

- data/8_feedback/<label_name>/\*.jpg
- data/8_feedback/<label_name>/\*.json

### C) Retrain runtime model

1. Click Retrain model now.
2. Camera is paused automatically during retraining.
3. When retraining completes, camera resumes.
4. Updated model is written to models/lbph_final.yml and runtime config is refreshed.

## 6) Clean Up Feedback (Optional)

If you are done with reinforcement and do not need history, you can remove:

- data/8_feedback

Warning:

- Deleting it removes all reinforcement samples permanently.
- Future retraining will only use baseline datasets unless new feedback is collected.

## 7) Troubleshooting

### App exits after retrain

- Run only one server instance at a time.
- Start app with:

```powershell
python main.py
```

- Confirm health endpoint from another terminal:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5000/api/health | Select-Object -ExpandProperty Content
```

### No face samples saved for feedback

- Keep your face in frame before pressing confirm/correct.
- If no ROI is available, the app rejects feedback to avoid bad labels.

### Camera issues

- Ensure webcam is not locked by another app.
- Restart app and reopen /training.

## Key Paths

- App entry: main.py
- Training backend service: ml/reinforcement_pipeline.py
- Camera service: service/camera_service.py
- Training UI script: presentation/ui/training.js
- Training page: presentation/views/training.html
- Runtime model config: models/realtime_model_config.json
