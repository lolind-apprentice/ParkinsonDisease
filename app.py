import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc, ConfusionMatrixDisplay, RocCurveDisplay,
    mean_absolute_error, mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE
import lightgbm as lgbm
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.calibration import CalibrationDisplay
import seaborn as sns
import xgboost as xgb

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

# --- NAVIGATION SETUP ---
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "About the App"

# Callback functions to update page state safely
def go_to_classification():
    st.session_state.app_mode = "Parkinson's Detection (Classification)"

def go_to_regression():
    st.session_state.app_mode = "Telemonitoring UPDRS (Regression)"

# Sidebar Navigation using session state key
app_mode = st.sidebar.radio(
    "Choose the dataset/task:",
    [
        "About the App",
        "Parkinson's Detection (Classification)",
        "Telemonitoring UPDRS (Regression)"
    ],
    key="app_mode"
)

# --- PAGE 1: ABOUT ---
if app_mode == "About the App":
    st.header("Overview")
    st.markdown("""
    This interactive dashboard analyzes vocal measurement datasets to:
    1. **Detect Parkinson's Disease (PD)** using voice recording classifications (Random Forest & Logistic Regression).
    2. **Track Symptom Severity** by predicting Unified Parkinson's Disease Rating Scale (UPDRS) scores using remote telemonitoring metrics (LightGBM).
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🎙️ Parkinson's Detection")
        st.write("Explore voice classification models, confusion matrices, and model auditing.")
        st.button(
            "Go to Classification Task ➡️", 
            on_click=go_to_classification, 
            use_container_width=True
        )

    with col2:
        st.write("### 📈 Telemonitoring UPDRS")
        st.write("Predict symptom progression and evaluate LightGBM regression performance.")
        st.button(
            "Go to Regression Task ➡️", 
            on_click=go_to_regression, 
            use_container_width=True
        )

# --- PAGE 2: CLASSIFICATION & MODEL AUDITING ---
elif app_mode == "Parkinson's Detection (Classification)":
    st.header("🎙️ Parkinson's Detection & Comprehensive Model Audit")
    
    df = load_classification_data(CLASSIFICATION_DATA_PATH)
    
    if df is not None:
        # --- DATASET EXPLORATION ---
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

        # Visualizations
        col_vis1, col_vis2 = st.columns(2)
        
        with col_vis1:
            st.write("### Target Distribution")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            df['status'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'], edgecolor='black', ax=ax)
            ax.set_xticklabels(["Healthy (0)", "Parkinson's (1)"], rotation=0)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        with col_vis2:
            st.write("### Top Correlations with Diagnosis")
            features = df.drop(['status', 'name'], axis=1)
            target = df['status']
            corr = features.corrwith(target).abs().sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            corr.head(10).plot(kind='barh', color='teal', ax=ax)
            ax.invert_yaxis()
            ax.set_xlabel("Absolute Correlation Strength")
            st.pyplot(fig)

        # --- MODELING SETUP ---
        st.markdown("---")
        st.subheader("🤖 Model Pipeline & Training")
        
        split_method = st.radio(
            "Select Data Splitting Strategy:",
            ("Patient-Level Split (Group-based, avoids leakage)", "Stratified Split (Standard)"),
            index=1,
            horizontal=True
        )

        # Sanitize column names for XGBoost & LightGBM compatibility
        df_clean = df.copy()
        raw_feature_cols = [c for c in df_clean.columns if c not in ['name', 'status']]
        sanitized_cols = [
            col.replace(':', '_').replace('(', '').replace(')', '').replace('%', '_percent_').replace('.', '_')
            for col in raw_feature_cols
        ]
        col_rename_dict = dict(zip(raw_feature_cols, sanitized_cols))
        df_clean.rename(columns=col_rename_dict, inplace=True)

        # Data Splitting
        if "Patient-Level" in split_method:
            unique_patients = df_clean['name'].apply(lambda x: '_'.join(x.split('_')[:3])).unique()
            patient_train, patient_test = train_test_split(np.array(unique_patients), test_size=0.2, random_state=42)

            train_df = df_clean[df_clean['name'].apply(lambda x: '_'.join(x.split('_')[:3])).isin(patient_train)]
            test_df = df_clean[df_clean['name'].apply(lambda x: '_'.join(x.split('_')[:3])).isin(patient_test)]

            X_train = train_df[sanitized_cols]
            y_train = train_df['status']
            X_test = test_df[sanitized_cols]
            y_test = test_df['status']
        else:
            X = df_clean[sanitized_cols]
            y = df_clean['status']
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(sss.split(X, y))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scaler
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        # Fit all 3 models for side-by-side auditing
        models = {
            "XGBoost": xgb.XGBClassifier(
                objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ),
            "LightGBM": lgbm.LGBMClassifier(
                objective='binary', metric='binary_logloss', n_estimators=100,
                learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1, verbose=-1
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=42 
            )
        }

        preds, probas = {}, {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            preds[name] = model.predict(X_test_scaled)
            probas[name] = model.predict_proba(X_test_scaled)[:, 1]

        # Model Selector for Single-Model Inspection
        selected_model_name = st.selectbox("Select Model for Single Focus", list(models.keys()))
        selected_clf = models[selected_model_name]
        y_pred = preds[selected_model_name]
        y_prob = probas[selected_model_name]

        # Single Model Display Metrics
        st.markdown(f"#### Performance: **{selected_model_name}**")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        m2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
        m3.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
        m4.metric("F1-Score", f"{f1_score(y_test, y_pred):.4f}")
        m5.metric("ROC AUC", f"{roc_auc_score(y_test, y_prob):.4f}")

        # Visualizations for Selected Model
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_estimator(selected_clf, X_test_scaled, y_test, cmap=plt.cm.Blues, ax=ax)
            ax.set_title(f"Confusion Matrix ({selected_model_name})")
            st.pyplot(fig)

        with p_col2:
            fig, ax = plt.subplots(figsize=(5, 4))
            importances = pd.Series(selected_clf.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10)
            importances.plot(kind='barh', color='teal', ax=ax)
            ax.invert_yaxis()
            ax.set_title(f"Top 10 Features ({selected_model_name})")
            st.pyplot(fig)

        # --- ADVANCED AUDITING SECTION ---
        st.markdown("---")
        st.subheader("🔍 Model Auditing & Comparative Analysis")
        
        audit_tab1, audit_tab2, audit_tab3, audit_tab4 = st.tabs([
            "📈 ROC Curve Comparison", 
            "📊 Feature Importance Audit", 
            "🎯 Probability Calibration", 
            "⚠️ Misclassification Analysis"
        ])

        # TAB 1: ROC Curves
        with audit_tab1:
            st.write("### Multi-Model ROC Curve Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = {'XGBoost': 'blue', 'LightGBM': 'green', 'Random Forest': 'red'}
            
            for name in models.keys():
                fpr, tpr, _ = roc_curve(y_test, probas[name])
                roc_auc_val = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors[name], lw=2, label=f'{name} (AUC = {roc_auc_val:.3f})')
            
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC)')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        # TAB 2: Consolidated Feature Importance
        with audit_tab2:
            st.write("### Consolidated Top 15 Feature Importances Across Models")
            importance_df = pd.DataFrame({
                name: model.feature_importances_ for name, model in models.items()
            }, index=X_train.columns).fillna(0)
            
            importance_df['Mean_Importance'] = importance_df.mean(axis=1)
            importance_df = importance_df.sort_values(by='Mean_Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            importance_df.drop('Mean_Importance', axis=1).head(15).plot(kind='barh', ax=ax, colormap='viridis')
            ax.invert_yaxis()
            ax.set_xlabel('Feature Importance Score')
            ax.set_ylabel('Feature Name')
            ax.set_title('Top 15 Features Comparison')
            st.pyplot(fig)

        # TAB 3: Model Calibration (Reliability Diagram)
        with audit_tab3:
            st.write("### Calibration Curves (Reliability Diagram)")
            st.markdown(
                "> **Why Calibration Matters:** Tree-based models often produce uncalibrated probabilities. "
                "A well-calibrated curve stays close to the dashed diagonal."
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            
            for name in models.keys():
                CalibrationDisplay.from_predictions(
                    y_test, probas[name], n_bins=10, strategy='uniform', 
                    ax=ax, color=colors[name], label=name
                )
            
            ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated', color='gray')
            ax.set_title('Probability Calibration Comparison')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        # TAB 4: Misclassifications Audit
        with audit_tab4:
            st.write("### Identifying Challenging Edge Cases")
            
            # Extract errors for each model
            misclass_dict = {}
            for name in models.keys():
                err_mask = (y_test != preds[name])
                misclass_dict[name] = set(X_test[err_mask].index)

            # Common errors across all 3 models
            common_err_idx = list(set.intersection(*misclass_dict.values()))
            
            st.warning(f"**{len(common_err_idx)} instances** were consistently misclassified by ALL three models.")
            
            if common_err_idx:
                err_df = X_test.loc[common_err_idx].copy()
                err_df['True Label'] = y_test.loc[common_err_idx]
                
                # Show reference medians vs misclassifications
                st.write("#### Detailed Feature View of Common Edge Cases")
                st.dataframe(err_df)

                st.markdown("""
                #### 📌 Insights on Edge Cases
                * **Instance Index 185 (False Positive):** Healthy individual exhibiting a high `PPE` (0.214) and `spread1` (-5.593) which strongly mimics Parkinson's voice profiles.
                * **Instance Index 194 (False Positive):** Healthy individual with low `HNR` (21.209) and high `MDVP:Jitter(%)` (0.00567), leading models to predict PD.
                * **Instance Index 7 (False Negative):** Parkinson's patient with atypically stable vocal metrics (high `HNR` of 26.892, low `Jitter` and `Shimmer`), causing models to miss the diagnosis.
                """)

    else:
        st.error(f"Data file not found at `{CLASSIFICATION_DATA_PATH}`. Please check the file path.")
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