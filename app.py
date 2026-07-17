import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    mean_absolute_error, mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMRegressor

# Page configuration
st.set_page_config(
    page_title="Parkinson's Disease Analysis & Modeling",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths to your bundled data
DATA_DIR = "data"
CLASSIFICATION_DATA_PATH = os.path.join(DATA_DIR, "parkinsons.data")
REGRESSION_DATA_PATH = os.path.join(DATA_DIR, "parkinsons_updrs.data")

# Cache data loading to ensure high performance
@st.cache_data
def load_classification_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_regression_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

st.title("🧠 Parkinson's Disease Machine Learning Dashboard")
st.markdown("This application analyzes vocal measurement datasets for clinical diagnosis and remote tracking.")

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose the dataset/task:", [
    "About the App",
    "Parkinson's Detection (Classification)",
    "Telemonitoring UPDRS (Regression)"
])

# --- PAGE 1: ABOUT ---
if app_mode == "About the App":
    st.header("Overview")
    st.markdown("""
    This interactive dashboard analyzes vocal measurement datasets to:
    1. **Detect Parkinson's Disease (PD)** using voice recording classifications (Random Forest & Logistic Regression).
    2. **Track Symptom Severity** by predicting Unified Parkinson's Disease Rating Scale (UPDRS) scores using remote telemonitoring metrics (LightGBM).
    
    ### How to Use:
    - Select a task from the sidebar. 
    - The data is already loaded and ready to analyze!
    """)

# --- PAGE 2: CLASSIFICATION ---
elif app_mode == "Parkinson's Detection (Classification)":
    st.header("🎙️ Parkinson's Detection from Voice Measurements")
    
    df = load_classification_data(CLASSIFICATION_DATA_PATH)
    
    if df is not None:
        st.subheader("📊 Dataset Exploration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Recordings", df.shape[0])
        with col2:
            st.metric("Total Features", df.shape[1] - 2) # excluding name and status
        with col3:
            healthy_count = (df['status'] == 0).sum()
            pd_count = (df['status'] == 1).sum()
            st.metric("Healthy / PD Cases", f"{healthy_count} / {pd_count}")

        if st.checkbox("Show raw data sample"):
            st.dataframe(df.head(10))

        # Class distribution plot
        st.write("### Target Distribution")
        fig, ax = plt.subplots(figsize=(6, 3))
        df['status'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'], edgecolor='black', ax=ax)
        ax.set_xticklabels(["Healthy (0)", "Parkinson's (1)"], rotation=0)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Feature Correlation
        st.write("### Feature Correlations with Diagnosis")
        features = df.drop(['status', 'name'], axis=1)
        target = df['status']
        corr = features.corrwith(target).abs().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        corr.head(15).plot(kind='barh', color='teal', ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel("Absolute Correlation Strength")
        st.pyplot(fig)

        # Modeling Section
        st.markdown("---")
        st.subheader("🤖 Model Training & Evaluation")
        
        # Prepare Data (Patient-wise split)
        unique_patients = df['name'].apply(lambda x: '_'.join(x.split('_')[:3])).unique()
        # Convert to a standard NumPy array to make it compatible with train_test_split
        patient_train, patient_test = train_test_split(np.array(unique_patients), test_size=0.3, random_state=24)

        train_df = df[df['name'].apply(lambda x: '_'.join(x.split('_')[:3])).isin(patient_train)]
        test_df = df[df['name'].apply(lambda x: '_'.join(x.split('_')[:3])).isin(patient_test)]

        X_train = train_df.drop(['name', 'status'], axis=1)
        y_train = train_df['status']
        X_test = test_df.drop(['name', 'status'], axis=1)
        y_test = test_df['status']

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        model_choice = st.selectbox("Select Model to Train", ["Tuned Random Forest", "Logistic Regression (SMOTE)"])

        if model_choice == "Tuned Random Forest":
            clf = RandomForestClassifier(n_estimators=100, max_features='sqrt', class_weight='balanced', random_state=42)
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        else: 
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
            
            clf = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
            clf.fit(X_train_res, y_train_res)
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]

        # Display Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        col2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
        col3.metric("Recall (Sensitivity)", f"{recall_score(y_test, y_pred):.4f}")
        col4.metric("F1-Score", f"{f1_score(y_test, y_pred):.4f}")
        col5.metric("ROC AUC", f"{roc_auc_score(y_test, y_prob):.4f}")

        # Plots
        st.write("### Evaluation Visualizations")
        plot_col1, plot_col2 = st.columns(2)
        
        with plot_col1:
            fig, ax = plt.subplots(figsize=(5, 5))
            ConfusionMatrixDisplay.from_estimator(clf, X_test_scaled, y_test, cmap=plt.cm.Blues, ax=ax)
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
            
        with plot_col2:
            fig, ax = plt.subplots(figsize=(5, 5))
            RocCurveDisplay.from_estimator(clf, X_test_scaled, y_test, ax=ax)
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_title("ROC Curve")
            st.pyplot(fig)
    else:
        st.error(f"Data file not found at `{CLASSIFICATION_DATA_PATH}`. Please ensure the 'data' directory and file exist.")

# --- PAGE 3: REGRESSION ---
elif app_mode == "Telemonitoring UPDRS (Regression)":
    st.header("📈 Parkinson's Telemonitoring UPDRS Prediction")
    
    df = load_regression_data(REGRESSION_DATA_PATH)

    if df is not None:
        st.subheader("📊 Telemonitoring Dataset Exploration")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Audio Samples", df.shape[0])
        with col2:
            st.metric("Unique Subjects Tracked", df['subject#'].nunique())
        with col3:
            st.metric("Average Total UPDRS", f"{df['total_UPDRS'].mean():.2f}")

        if st.checkbox("Show raw data sample"):
            st.dataframe(df.head(10))

        # Visualizing UPDRS Target Distribution
        st.write("### Distribution of Total UPDRS Scores")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.hist(df['total_UPDRS'], bins=30, color='skyblue', edgecolor='black')
        ax.set_xlabel("Total UPDRS Score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # ML Pipeline: Subject-wise Split & LightGBM Regressor
        st.markdown("---")
        st.subheader("🤖 LightGBM Regression Engine")

        unique_subjects = df['subject#'].unique()
        # Convert to a standard NumPy array to avoid PyArrow index issues
        subject_train, subject_test = train_test_split(np.array(unique_subjects), test_size=0.2, random_state=42)

        train_df = df[df['subject#'].isin(subject_train)]
        test_df = df[df['subject#'].isin(subject_test)]

        features_to_drop = ['subject#', 'age', 'sex', 'motor_UPDRS', 'total_UPDRS']
        X_train = train_df.drop(features_to_drop, axis=1)
        y_train = train_df['total_UPDRS']
        X_test = test_df.drop(features_to_drop, axis=1)
        y_test = test_df['total_UPDRS']

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        # Clean Column names for LightGBM
        X_train_scaled.columns = [''.join(c if c.isalnum() else '_' for c in str(x)) for x in X_train_scaled.columns]
        X_test_scaled.columns = [''.join(c if c.isalnum() else '_' for c in str(x)) for x in X_test_scaled.columns]

        # Model Execution
        with st.spinner("Training LightGBM Regressor..."):
            lgbm = LGBMRegressor(random_state=42)
            lgbm.fit(X_train_scaled, y_train)
            y_pred = lgbm.predict(X_test_scaled)

        # Show Performance Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean Absolute Error (MAE)", f"{mean_absolute_error(y_test, y_pred):.4f}")
        m2.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test, y_pred):.4f}")
        m3.metric("Root MSE (RMSE)", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        m4.metric("R-squared (R²)", f"{r2_score(y_test, y_pred):.4f}")

        # Scatter Plot
        st.write("### Actual vs. Predicted UPDRS Scores")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(y_test, y_pred, alpha=0.3, color='purple')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Total UPDRS Scores')
        ax.set_ylabel('Predicted Total UPDRS Scores')
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.error(f"Data file not found at `{REGRESSION_DATA_PATH}`. Please ensure the 'data' directory and file exist.")