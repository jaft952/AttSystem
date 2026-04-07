Objectives

- To develop a Python-based face recognition attendance prototype that detects a face from a webcam frame, preprocesses it into a standardized ROI, and identifies the most similar enrolled student using LBPH face recognition.
- To build and train an LBPHFaceRecognizer model using an augmented dataset, and evaluate its robustness under different real-world lighting conditions.
- To evaluate the proposed system using quantitative metrics (e.g., Top-1 accuracy, FAR/FRR) and present results clearly.

Training steps:

- Step 1: Data Collection & Augmentation
  Member 3 expands the raw dataset from 16-20 images to ~100 images per person using Albumentations, simulating different lighting conditions.
- Step 2: Preprocessing
  The augmented dataset goes through Member 1's preprocessing pipeline (Face Detection, Crop, Resize, Grayscale, Illumination Normalization).
- Step 3: Model Training
  Member 2 trains the LBPHFaceRecognizer model on the preprocessed dataset and tunes LBPH parameters (radius, neighbors, grid).
- Step 4: Evaluation
  Member 3 evaluates the model under different lighting conditions and reports accuracy, FAR/FRR, and confusion matrix.

Notes from lecturer:

- PCA removed — overengineered for this scenario.
- Logistic Regression threshold removed — not suitable for live/real-world conditions (e.g., dark room).
- Focus should be on demonstrating illumination robustness through image processing techniques.
- LBPHFaceRecognizer is approved. Model has built-in similarity matching via confidence score.
- Pre-trained weights are allowed as long as image processing techniques and algorithms are showcased.
- 100 images per person confirmed for training.

---

## Camera Setup

The live camera runs in the browser now. The backend handles prediction, feedback storage, and retraining only.

The app now starts over HTTP by default so the page loads normally on a hosted IP. Browser camera access still requires a secure origin, so use `https://localhost:5000` for local development or supply real TLS certs through `ATT_SSL_CERT` and `ATT_SSL_KEY` if you need HTTPS on a machine IP.

# System Architecture

┌─────────────────────────────────────────────────────────┐
│ Presentation Layer │
│ - Streamlit UI │
│ - User input handling │
│ - Result display │
└────────────────────┬────────────────────────────────────┘
│
┌────────────────────▼────────────────────────────────────┐
│ Application Layer │
│ - AttendanceService │
│ - RegistrationService │
│ - Business logic & workflow orchestration │
└────────────────────┬────────────────────────────────────┘
│
┌────────────────────▼────────────────────────────────────┐
│ Processing Layer │
│ - FaceDetector (Member 1) │
│ - LBPHModelTrainer (Member 2) │
│ - AugmentationAndEvaluator (Member 3) │
└────────────────────┬────────────────────────────────────┘
│
┌────────────────────▼────────────────────────────────────┐
│ Data Layer │
│ - DatabaseManager (Member 4) │
│ - File I/O operations │
│ - Data persistence (CSV, numpy files, .yml model) │
└─────────────────────────────────────────────────────────┘

---

## Member 1: Face Detection and Preprocessing

**Coding Task:** 1. Preprocessing (before detection)
_ Resize raw frame to a fixed input scale (e.g., width 640) to reduce computational load. 2. Face detection
_ Use Haar Cascade Classifier (cv2.CascadeClassifier) or face*recognition library.
* Detect face bounding box from a raw image/frame.
_ Run face detection every N frames (5-10 is best practice).
_ Handle multi-face cases (select largest face or process all faces based on system mode). 3. Preprocessing (after detection)
_ Crop face ROI with margin (padding) and boundary checking (prevent crashes).
_ Convert to grayscale and resize to fixed size (e.g., 128×128) for mathematical consistency.
_ Apply illumination normalization on face ROI: - Histogram Equalization - CLAHE (Contrast Limited Adaptive Histogram Equalization) - Gamma Correction (for low-light / dark room scenarios) 4. Input/Output interface (module contract)
_ Input: raw RGB/BGR image or camera frame.
\_ Output: standardized grayscale face image + detection metadata (bbox, confidence, status).

**Documentation:**

1. Section 2.2
   - Compare face detection methods (Haar Cascade vs face_recognition vs YOLO)
2. Section 3.3
   - Describe preprocessing pipeline
   - Explain illumination normalization techniques and why they are used
3. Section 4.1
   - Show detection success rate
4. Section 1.1

---

## Member 2: LBPH Model Training

**Coding Task:** 1. Model training
_ Train cv2.face.LBPHFaceRecognizer_create() on the preprocessed dataset (output from Member 1).
_ Tune LBPH parameters: radius, neighbors, grid*x, grid_y to find the best configuration.
* Save the trained model to a .yml file for inference. 2. Inference & matching
_ Load trained model and run predict() on a live face ROI.
_ Return predicted student ID and confidence score.
_ Apply confidence threshold for match / no-match decision (lower confidence = better match in LBPH). 3. Module contract
_ Input: standardized grayscale face ROI + student labels.
\_ Output: predicted student ID + confidence score + match status.

**Documentation:**

1. Section 2.1
   - Explain what are features in face recognition
2. Section 2.2
   - Compare LBPH with other feature extraction algorithms (HOG, Eigenfaces)
3. Section 3.3
   - Explain how LBPHFaceRecognizer works internally
   - Describe parameter tuning process
4. Section 1.2

---

## Member 3: Data Augmentation & Evaluation

**Coding Task:** 1. Data augmentation
_ Use Albumentations library to expand dataset from 16-20 images to ~100 images per person.
_ Apply augmentations that simulate real-world conditions: - Brightness & contrast adjustments (simulate different lighting) - Shadow overlays (simulate side lighting / backlight) - Horizontal flips, rotations - Gaussian blur & noise 2. Illumination robustness evaluation
_ Test system accuracy under different lighting conditions:
Normal light, Low light, Backlight, Side lighting, Overexposure.
_ Compare results with and without preprocessing (CLAHE, Histogram EQ, Gamma Correction).
_ Present findings as a comparison table and chart. 3. Evaluation & metrics
_ Set confidence threshold for match / no-match decision.
_ Generate confusion matrix.
_ Calculate FAR (False Acceptance Rate) and FRR (False Rejection Rate). \* Calculate Top-1 accuracy.

**Documentation:**

1. Section 2.2
   - Compare augmentation techniques
2. Section 3.2
   - Describe dataset structure and augmentation process
3. Section 4.1
   - Confusion matrix, accuracy, FAR/FRR results
4. Section 4.2
   - Discuss illumination robustness findings
   - Explain why certain lighting conditions are harder to handle
5. Section 1.3

---

## Member 4: Integration & System Management

**Coding Task:**

1. Database Management
   - Manage student records (IDs, names) and attendance logs (timestamp, predicted ID, confidence score, status).
2. Business Logic (Application Layer)
   - Integrate Member 1, 2, 3's modules.
   - Implement complete attendance workflow.
   - Exception handling.
3. User Interface
   - Build Streamlit UI for live attendance and registration.

**Documentation:**

1. Section 3.1
   - Draw complete system architecture
   - Draw workflow flowchart
2. Section 3.2
   - Describe test dataset and data structure
3. Section 4.1
   - UI screenshots showing successful attendance marking
4. Section 4.2
   - Overall system performance analysis
5. Section 5
   - All
