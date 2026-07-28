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
* **Goal**: Predict the progression and severity of early-stage Parkinson's symptoms remotely. Using a proper patient-level split so no patient's recordings leaked between train and test.
* **Target**: total_UPDRS (Unified Parkinson's Disease Rating Scale, score range 0-199).
#### Model Evaluated: 
* LightGBM Regressor
* XGBoost Regressor

**Key Metrics**: Root Mean Square Error (RMSE) and R-squared ($R^2$) score.
* LightGBM Regressor: RMSE 13.597489, R² -1.126417	
* XGBoost Regressor: RMSE 13.136339, R² -1.066467 <br>

This means both models did worse than simply guessing the average severity for everyone. Tuning made this worse since tuning can't manufacture a signal that isn't there. This failed because Parkinson's severity depends mostly on who the patient is not on subtle acoustic wobbles a model can pick up from a single recording of a stranger. Two patients can sound similarly "off" in their voice while having very different actual severity, simply because one started worse than the other. Voice alone carries almost no information about a stranger's absolute severity level.

#### First fix:
The first tempting fix was to instead predict a patient's next recording from their current one. That scored 99%+ accuracy, but it was an illusion. This dataset's severity scores aren't measured fresh at each recording. They are linearly interpolated between infrequent real doctor visits. Consecutive recordings are often only hours apart, so the "next" score is almost guaranteed to be nearly identical to the current one by construction. This came to light because a dumb "persistence" baseline (guess zero change) matched or beat XGBoost/LightGBM outright. The models weren't learning a voice→severity relationship, they were exploiting the fact that adjacent interpolated points sit close together.

#### Second fix:
The second, more rigorous fix was to force a real time gap of days to weeks between recordings, and predict the change in severity rather than the raw next number. This removes the interpolation shortcut entirely: "assume no change" is now a real, beatable baseline rather than a guaranteed win. Even so, the "assume no change" baseline still beat XGBoost and LightGBM. RMSE dropped to 2.25, but R² was still negative, meaning predicting no change outperformed the regression models even under the fairest test we could construct.

#### Conclusion:
Based on all these attempts we concluded that the voice features in this dataset simply don't carry the signal needed to predict how severity moves. That is why we pivoted away from this dataset.

### Daphnet Freezing of Gait


