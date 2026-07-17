# 🧠 Parkinson's Disease Analysis & Modeling
### AI4ALL Project — Group O1C

An interactive machine learning application built to analyze biomedical vocal measurements for both **Parkinson's Disease detection** (classification) and **remote symptom tracking** (regression). 

This project transitions from our initial Google Colab exploratory analysis to a live **Streamlit Web Dashboard**.

---

## 📂 Project Structure

```text
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Files excluded from Git tracking
└── data/                      # Bundled datasets
    ├── parkinsons.data        # Dataset for classification task
    └── parkinsons_updrs.data  # Dataset for telemonitoring regression task
```

## 🎯 Project Objectives
This project tackles Parkinson's Disease (PD) tracking from two distinct, machine learning-driven perspectives:
### Parkinson's Detection (Classification) 
* **Goal**: Discriminate healthy individuals from patients diagnosed with Parkinson's Disease using voice recordings.
* **Target**: status (0 for healthy, 1 for PD).
#### Models Evaluated: 
* Tuned Random Forest Classifier 
* Logistic Regression with SMOTE.
* add more here...

**Key Metrics:** Balanced Accuracy, Recall (Sensitivity), and F1-Score.
### Telemonitoring Severity Tracking (Regression) 
* **Goal**: Predict the progression and severity of early-stage Parkinson's symptoms remotely.
* **Target**: total_UPDRS (Unified Parkinson's Disease Rating Scale, score range 0-199).
#### Model Evaluated: 
* LightGBM Regressor.
* add more here...

**Key Metrics**: Mean Absolute Error (MAE) and R-squared ($R^2$) score.