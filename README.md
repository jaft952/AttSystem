┌─────────────────────────────────────────────────────────┐
│              Presentation Layer                         │
│  - Streamlit UI                                         │
│  - User input handling                                  │
│  - Result display                                       │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│             Application Layer                           │
│  - AttendanceService                                    │
│  - RegistrationService                                  │
│  - Business logic & workflow orchestration              │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│            Processing Layer                             │
│  - FaceDetector (Member 1)                              │
│  - FeatureExtractor (Member 2)                          │
│  - CBIRMatchingEngine (Member 3)                        │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Data Layer                                 │
│  - DatabaseManager (Member 4)                           │
│  - File I/O operations                                  │
│  - Data persistence (CSV, numpy files)                  │
└─────────────────────────────────────────────────────────┘

Member 1 : Face Detection and Preprocessing
    Coding Task : 
    1. Face detection 
    2. image preprocessing 
    3. input/output interface 
        - input raw image output standardized grayscale face image
    Documentation:
    1. section 2.2 
        - compare face detection methods
    2. section 3.3
        - describe preprocessing pipeline
        - explain how ur approach works
    3. section 4.1 
        - show detection success rate 
    4. section 1.1
Member 2: Feature Extraction
    Coding Task: 
    1. Feature Extraction Algorithm 
    2. Feature Optimization 
        - feature normalization
    3. Feature Storage 
        - save features to .npy files
    Documentation:
    1. section 2.1 
        - Explain what are features in face recognition
    2. section 2.2
        - compare existing algorithm
    3. section 3.3
        - Explain how your chosen method extracts features
    4. section 1.2
Member 3: CBIR Matching Engine
    Coding Task: 
    1. Similarity Calculation
    2. Retrieval Algorithm
    3. Threshold Determination
        - Set reasonable similarity threshold
        - handle no match scenario
    Documentation:
    1. section 2.2 
        - compare existing algo
    2. section 3.3
        - Explain CBIR retrieval process
    3. section 4.1
        - confusion matrix
    4. section 4.2 
        - Discuss why certain students are harder to match
    5. section 1.3
Member 4: Integration & System Management
    Coding Task: 
    1. Database Management
    2. Business Logic(Application Layer)
        - integrate member 1,2,3 's modules
        - Implement complete attendance workflow
        - Exception handling
    3. user interface
    Documentation:
    1. section 3.1
        - Draw complete system architecture
        - Draw workflow flowchart
    2. section 3.2
        - Describe your test dataset and data structure
    3. section 4.1 
        - UI screenshots showing successful
    4. section 4.2 
        - Overall system performance analysis
    5. section 5
        - all

