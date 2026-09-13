# Real-Time Hand Gesture Recognition & Computer Vision-Based Game Controller

An end-to-end Computer Vision and Machine Learning system that captures 21 3D hand landmarks via webcam, normalizes spatial coordinates, extracts 90 engineered features, trains and benchmarks multiple ML classifiers (KNN, SVM, Random Forest, Multi-Layer Perceptron, XGBoost), and deploys real-time gesture-based game control with safety handling and gameplay telemetry.

---

## 📌 Problem Statement

Traditional keyboard-based gaming controls can be limiting or unintuitive for certain interactive arcade games such as *Hill Climb Racing*. While raw MediaPipe demos demonstrate landmark visualization, they lack robust machine learning pipelines, feature engineering, latency optimization, confidence safety thresholds, and gameplay telemetry analytics. This project builds an end-to-end ML pipeline that translates real-time hand gestures into responsive, safe, and logged vehicle controls.

---

## 🎯 System Architecture

```
Webcam Frame (OpenCV)
        │
        ▼
MediaPipe Hands Detection
(21 3D Hand Landmarks)
        │
        ▼
Spatial Preprocessing
(Wrist Translation & Palm Scale Normalization)
        │
        ▼
Feature Engineering
(90 Geometric Features)
        │
        ▼
ML Model Inference
(Best Model: MLP)
        │
        ▼
Gesture Prediction
(ACCELERATE / BRAKE / NEUTRAL)
        │
        ├──────────────────────────┐
        ▼                          ▼
Keyboard Game Control       2D Vehicle Simulator
        │                          │
        └────────────┬─────────────┘
                     ▼
          Telemetry Logging (CSV)
                     │
                     ▼
          Streamlit Analytics Dashboard
```

---

## 🚀 Key Features & Technologies

- **Languages & Frameworks:** Python 3.11+, OpenCV, MediaPipe, NumPy, Pandas, Scikit-learn, XGBoost, Streamlit, PyYAML, pynput, Pytest, Matplotlib, Seaborn.
- **3 Gesture Classes:** `ACCELERATE`, `BRAKE`, `NEUTRAL`.
- **Feature Engineering:** 90 engineered geometric features including normalized relative coordinates, inter-fingertip distances, fingertip-to-wrist distances, finger extension distances, joint angles, and palm orientation features.
- **Multi-Model Benchmark:** Automated training and evaluation across KNN, SVM, Random Forest, MLP, and XGBoost.
- **Real-Time Safety:** Responsive gesture-to-key control with confidence handling, safe key-release behavior, and automatic release on application shutdown or hand loss.
- **Integrated Fallback Simulator:** Built-in 2D Hill Climb vehicle physics engine rendered directly in OpenCV canvas.
- **Telemetry & Dashboard:** Real-time logging of prediction confidence, inference latency, reaction delay, FPS, and interactive Streamlit analytics.

---

## 📂 Project Structure

```
Gesture-Game-Controller/
│
├── app.py                      # Main real-time application entry point
├── collect_data.py             # Interactive webcam dataset collector CLI
├── train.py                    # Main ML model training & evaluation pipeline
├── dashboard.py                # Streamlit analytics dashboard
├── requirements.txt            # Project Python dependencies
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git exclusion rules
│
├── config/
│   └── config.yaml             # Central configuration (camera, thresholds, key mappings)
│
├── data/
│   ├── raw/
│       └── gestures.csv        # Collected / generated raw landmark dataset
│
├── models/
│   ├── best_model.pkl          # Trained best performing classifier
│   ├── preprocessor.pkl        # Scaler, label encoder, and feature names
│   └── model_metadata.json     # Model metadata and class parameters
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb # Jupyter notebook for EDA
│
├── reports/
│   ├── eda_summary.txt         # EDA summary metrics
│   ├── eda_class_distribution.png
│   ├── eda_correlation_heatmap.png
│   ├── eda_pca_separability.png
│   ├── model_comparison.csv    # Benchmark metrics comparison table
│   ├── model_comparison.png    # Model performance comparison chart
│   ├── confusion_matrix.png    # Best model confusion matrix
│   ├── classification_report.txt
│
├── scripts/
│   └── generate_sample_data.py # Synthetic landmark data generator for offline testing
│
├── src/
│   ├── __init__.py
│   ├── hand_tracker.py         # MediaPipe 21-landmark detector wrapper
│   ├── preprocessing.py        # Spatial translation & scale normalizer
│   ├── feature_engineering.py  # 90-feature engineering module
│   ├── exploratory_data_analysis.py # EDA script generator
│   ├── model_training.py       # Multi-model training
│   ├── model_evaluation.py     # Evaluation metrics & visualization exporter
│   ├── gesture_predictor.py    # Real-time gesture inference and confidence handling
│   ├── game_controller.py      # Safe pynput keyboard controller
│   ├── chrome_game_controller.py  # Browser game keyboard controller
│   ├── game_simulator.py       # 2D Hill Climb vehicle physics simulator
│   ├── telemetry.py            # Real-time telemetry logger
│   └── analytics.py            # Telemetry metrics analyzer
│
└── tests/
    ├── test_preprocessing.py   # Unit tests for landmark normalizer
    ├── test_features.py        # Unit tests for feature extraction
    ├── test_model.py           # Unit tests for training & inference
    ├── test_controller.py      # Unit tests for key mapping & safety
    └── test_telemetry.py       # Unit tests for telemetry logging
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure Python 3.11 is installed on Windows.

### 2. Create Virtual Environment & Install Dependencies
```powershell
# Create virtual environment
py -3.11 -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install requirements
py -3.11 -m pip install -r requirements.txt
```

---

## 📊 Workflow & Execution Steps

### Step 1: Collect Hand Landmark Dataset (Webcam)
```powershell
python collect_data.py --gesture ACCELERATE --samples 1000
python collect_data.py --gesture BRAKE --samples 1000
python collect_data.py --gesture NEUTRAL --samples 1000
```
*Note: Press `SPACE` to toggle recording on/off. Press `Q` or `ESC` to save and exit.*

#### Bootstrap / Offline Data Generation (Optional for CI/Testing)
If webcam data is not immediately collected, generate 500 synthetic landmark samples for offline testing:
```powershell
python scripts/generate_sample_data.py --samples 500
```

### Step 2: Exploratory Data Analysis (EDA)
```powershell
python src/exploratory_data_analysis.py
```
Outputs EDA charts (`eda_class_distribution.png`, `eda_correlation_heatmap.png`, `eda_pca_separability.png`) and summary report to `reports/`.

### Step 3: Train & Benchmark ML Models
```powershell
python train.py
```
This script trains KNN, SVM, Random Forest, Multi-Layer Perceptron, and XGBoost, evaluates their performance, saves the best model to `models/best_model.pkl`, and exports reports to `reports/`.

### Step 4: Run Real-Time Gesture Control Application
```powershell
python app.py
```
Captures hand gestures in real time, displays the webcam HUD with landmarks, predicted gesture, confidence score, and FPS, and sends the corresponding keyboard input to the game controller. The application also supports the integrated 2D vehicle simulator for local testing. Press `Q` or `ESC` to quit safely.

### Step 5: Launch Streamlit Performance Analytics Dashboard
```powershell
streamlit run dashboard.py
```
Opens interactive browser dashboard displaying total predictions, gesture usage, confidence distribution, inference latency timeline, and model comparison metrics.

### Step 6: Execute Unit Test Suite
```powershell
py -3.11 -m pytest -v tests/
```

---

## 📈 Model Comparison & Empirical Results

Final benchmark results on the cleaned three-class dataset containing 3,000 samples.

| Model | Accuracy | F1 Score |
| :--- | :---: | :---: |
| KNN | 99.83% | 0.9983 |
| SVM | 99.83% | 0.9983 |
| Random Forest | **100.00%** | **1.0000** |
| Multi-Layer Perceptron (Best) | **100.00%** | **1.0000** |
| XGBoost | 99.83% | 0.9983 |


## ⚠️ Limitations & Future Improvements

1. **Extreme Lighting & Occlusion:** MediaPipe tracking accuracy can degrade under extreme low-light conditions or severe self-occlusion. Future work could include integrating depth sensor data.
2. **Two-Handed Controls:** Current implementation tracks 1 primary hand. Extending to two-handed multi-gesture tracking will enable simultaneous steering and braking.
3. **Adaptive Thresholding:** Future versions could implement dynamic confidence thresholds based on runtime conditions and user behavior.
