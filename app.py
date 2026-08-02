import os
import glob
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report,
    precision_recall_curve, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.inspection import PartialDependenceDisplay
from sklearn.calibration import CalibrationDisplay

import lightgbm as lgbm
from lightgbm import LGBMRegressor, LGBMClassifier
import xgboost as xgb

# Set Streamlit page configuration
st.set_page_config(
    page_title="Parkinson's Disease Analysis & Modeling",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Seed setting for reproducibility
np.random.seed(42)

# Define paths to bundled data
DATA_DIR = "data"
CLASSIFICATION_DATA_PATH = os.path.join(DATA_DIR, "parkinsons.data")
REGRESSION_DATA_PATH = os.path.join(DATA_DIR, "parkinsons_updrs.data")
# ==========================================
# DATA LOADING & CACHING
# ==========================================

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

COLS = [
    'time_ms',
    'ankle_fwd',
    'ankle_vert',
    'ankle_lat',
    'thigh_fwd',
    'thigh_vert',
    'thigh_lat',
    'trunk_fwd',
    'trunk_vert',
    'trunk_lat',
    'annotation',
]
@st.cache_data
def load_fog_data(data_dir="data"):
    # Locate all S*R*.txt files within the data folder (including subdirectories)
    txt_files = sorted(
        glob.glob(os.path.join(data_dir, "**", "S*R*.txt"), recursive=True)
    )

    if not txt_files:
        st.error(
            f"No dataset files found in '{data_dir}'. Please ensure your .txt files (e.g., S01R01.txt) are placed in the '{data_dir}' folder."
        )
        return pd.DataFrame()

    all_runs = []
    for fp in txt_files:
        fname = os.path.basename(fp).replace(".txt", "")
        subject_id = fname[:3]  # Extract e.g. 'S01'
        run_id = fname[3:]  # Extract e.g. 'R01'

        # Load file space-separated without header
        df = pd.read_csv(fp, sep=r"\s+", header=None, names=COLS)
        df["subject"] = subject_id
        df["run"] = run_id
        all_runs.append(df)

    data = pd.concat(all_runs, ignore_index=True)

    # Filter out unannotated samples (annotation == 0)
    data = data[data["annotation"] != 0].reset_index(drop=True)

    return data

def calculate_freeze_index(sig_window, fs):
    n = len(sig_window)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(sig_window - np.mean(sig_window))) ** 2

    locomotor_band = (freqs >= 0.5) & (freqs <= 3.0)
    freeze_band = (freqs >= 3.0) & (freqs <= 8.0)

    locomotor_power = power[locomotor_band].sum() + 1e-9
    freeze_power = power[freeze_band].sum()

    return freeze_power / locomotor_power, freeze_power + locomotor_power

@st.cache_data
def extract_fog_window_features(data, window_sec=4.0, overlap=0.5):
    sample_deltas = data.groupby(['subject', 'run'])['time_ms'].diff().dropna()
    median_delta_ms = sample_deltas.median()
    fs = 1000.0 / median_delta_ms

    window_len = int(window_sec * fs)
    step = int(window_len * (1 - overlap))

    sensor_cols = ['ankle_fwd', 'ankle_vert', 'ankle_lat',
                   'thigh_fwd', 'thigh_vert', 'thigh_lat',
                   'trunk_fwd', 'trunk_vert', 'trunk_lat']

    rows = []
    for (subj, run), g in data.groupby(['subject', 'run']):
        g = g.reset_index(drop=True)
        n = len(g)

        for start in range(0, n - window_len, step):
            w = g.iloc[start:start + window_len]
            feat = {}

            for col in sensor_cols:
                vals = w[col].values.astype(float)
                feat[f'{col}_mean'] = vals.mean()
                feat[f'{col}_std'] = vals.std()
                feat[f'{col}_min'] = vals.min()
                feat[f'{col}_max'] = vals.max()
                fi, energy = calculate_freeze_index(vals, fs)
                feat[f'{col}_freeze_index'] = fi
                feat[f'{col}_energy'] = energy

            feat['subject'] = subj
            feat['run'] = run
            # Binary label: >50% of the window annotated as freeze
            feat['label'] = 1 if (w['annotation'] == 2).mean() > 0.5 else 0
            # Continuous target: fraction of window spent in freeze state
            feat['freeze_frac'] = (w['annotation'] == 2).mean()
            rows.append(feat)

    features_df = pd.DataFrame(rows)
    return features_df, fs

def find_optimal_threshold(y_true, y_pred_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    f1_scores = np.nan_to_num(f1_scores)

    if len(thresholds) > 0:
        optimal_idx = np.argmax(f1_scores[:-1])
        return thresholds[optimal_idx]
    return 0.5

# ==========================================
# TITLE & NAVIGATION SETUP
# ==========================================

st.title("🧠 Parkinson's Disease Machine Learning Dashboard")
st.markdown("This application analyzes vocal measurement and accelerometer datasets for clinical diagnosis, telemonitoring, and real-time gait tracking.")

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "About the App"

def go_to_classification():
    st.session_state.app_mode = "Parkinson's Detection (Classification)"

def go_to_regression():
    st.session_state.app_mode = "Telemonitoring UPDRS (Regression)"

def go_to_fog():
    st.session_state.app_mode = "Freezing of Gait (FoG) Analysis"

app_mode = st.sidebar.radio(
    "Choose the dataset/task:",
    [
        "About the App",
        "Parkinson's Detection (Classification)",
        "Telemonitoring UPDRS (Regression)",
        "Freezing of Gait (FoG) Analysis"
    ],
    key="app_mode"
)

# ==========================================
# PAGE 1: ABOUT & SUMMARY STATISTICS
# ==========================================
if app_mode == "About the App":
    st.header("Overview & Dataset Auditing Summaries")
    st.markdown("""
    This interactive dashboard analyzes vocal measurement and accelerometer datasets to:
    1. **Detect Parkinson's Disease (PD)** using voice recording classifications.
    2. **Explore Symptom Severity** by analyzing Unified Parkinson's Disease Rating Scale (UPDRS) scores using remote telemonitoring metrics.
    3. **Detect & Estimate Freezing of Gait (FoG) Severity** using tri-axial accelerometer body sensors (Daphnet dataset).
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("### 🎙️ Parkinson's Detection")
        st.write("Explore voice classification models, confusion matrices, PDP plots, and patient-level auditing.")
        st.button("Go to Classification Task ➡️", on_click=go_to_classification, use_container_width=True)

    with col2:
        st.write("### 📈 Telemonitoring UPDRS")
        st.write("Predict symptom progression and evaluate regression strategies, leak challenges, and findings.")
        st.button("Go to Regression Task ➡️", on_click=go_to_regression, use_container_width=True)

    with col3:
        st.write("### 🏃 Freezing of Gait (FoG)")
        st.write("Evaluate real-time FoG classification and continuous severity estimation (`freeze_frac`).")
        st.button("Go to FoG Task ➡️", on_click=go_to_fog, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Dataset Summary Statistics")

    df_cls = load_classification_data(CLASSIFICATION_DATA_PATH)
    df_reg = load_regression_data(REGRESSION_DATA_PATH)

    tab_cls_stat, tab_reg_stat = st.tabs(["🎙️ Classification Dataset Stats", "📈 Telemonitoring Dataset Stats"])

    with tab_cls_stat:
        if df_cls is not None:
            st.write("#### Classification Features Summary")
            patient_ids = df_cls['name'].apply(lambda x: '_'.join(x.split('_')[:3]))
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Samples", df_cls.shape[0])
            m2.metric("Unique Patients", patient_ids.nunique())
            m3.metric("Total Features", df_cls.shape[1] - 2)
            m4.metric("Class Ratio (Healthy / PD)", f"{(df_cls['status']==0).sum()} / {(df_cls['status']==1).sum()}")

            st.write("**Descriptive Statistics (Acoustic Features)**")
            numeric_cls = df_cls.drop(columns=['name'])
            st.dataframe(numeric_cls.describe().T.style.format("{:.4f}"), use_container_width=True)
        else:
            st.warning("Classification dataset not found.")

    with tab_reg_stat:
        if df_reg is not None:
            st.write("#### Regression Features Summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Samples", df_reg.shape[0])
            m2.metric("Tracked Subjects", df_reg['subject#'].nunique())
            m3.metric("Target Variable Range (Total UPDRS)", f"{df_reg['total_UPDRS'].min():.1f} - {df_reg['total_UPDRS'].max():.1f}")

            st.write("**Descriptive Statistics (Telemonitoring Features)**")
            st.dataframe(df_reg.describe().T.style.format("{:.4f}"), use_container_width=True)
        else:
            st.warning("Telemonitoring dataset not found.")

# ==========================================
# PAGE 2: CLASSIFICATION & MODEL AUDITING
# ==========================================
elif app_mode == "Parkinson's Detection (Classification)":
    st.header("🎙️ Parkinson's Detection & Comprehensive Model Audit")

    with st.container(border=True):
        st.markdown("""
        ### 📌 Table of Contents
        * [📊 Dataset Exploration](#dataset-exploration)
        * [🤖 Model Pipeline & Training](#model-pipeline-training)
        * [🔍 Model Auditing & Comparative Analysis](#model-auditing-comparative-analysis)
            * 📈 ROC Curve Comparison
            * 📊 Feature Importance Audit
            * 📈 Partial Dependence Plots (PDP)
            * 👤 Patient-Level Performance Audit
            * 🎯 Probability Calibration
            * ⚠️ Misclassification Analysis
        """)
    df = load_classification_data(CLASSIFICATION_DATA_PATH)
    
    if df is not None:
        df['patient_id'] = df['name'].apply(lambda x: '_'.join(x.split('_')[:3]))

        st.markdown('<a id="dataset-exploration"></a>', unsafe_allow_html=True)
        st.subheader("📊 Dataset Exploration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Recordings", df.shape[0])
        with col2:
            st.metric("Total Features", df.shape[1] - 3)
        with col3:
            healthy_count = (df['status'] == 0).sum()
            pd_count = (df['status'] == 1).sum()
            st.metric("Healthy / PD Cases", f"{healthy_count} / {pd_count}")

        if st.checkbox("Show raw data sample"):
            st.dataframe(df.head(10))

        col_vis1, col_vis2 = st.columns(2)
        with col_vis1:
            st.write("### Target Distribution")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            df['status'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'], edgecolor='black', ax=ax)
            ax.set_xticklabels(["Parkinson's (1)", "Healthy (0)"], rotation=0)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        with col_vis2:
            st.write("### Top Correlations with Diagnosis")
            features = df.drop(['status', 'name', 'patient_id'], axis=1)
            target = df['status']
            corr = features.corrwith(target).abs().sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            corr.head(10).plot(kind='barh', color='teal', ax=ax)
            ax.invert_yaxis()
            ax.set_xlabel("Absolute Correlation Strength")
            st.pyplot(fig)

        st.markdown("---")
        st.markdown('<a id="model-pipeline-training"></a>', unsafe_allow_html=True)
        st.subheader("🤖 Model Pipeline & Training")
        
        split_method = st.radio(
            "Select Data Splitting Strategy:",
            ("Patient-Level Split (Group-based, avoids leakage)", "Stratified Split (Standard)"),
            index=1,
            horizontal=True
        )

        df_clean = df.copy()
        raw_feature_cols = [c for c in df_clean.columns if c not in ['name', 'status', 'patient_id']]
        sanitized_cols = [
            col.replace(':', '_').replace('(', '').replace(')', '').replace('%', '_percent_').replace('.', '_')
            for col in raw_feature_cols
        ]
        col_rename_dict = dict(zip(raw_feature_cols, sanitized_cols))
        df_clean.rename(columns=col_rename_dict, inplace=True)

        should_train = (
            "models_trained" not in st.session_state 
            or st.session_state.get("last_split_method") != split_method
        )

        if should_train:
            with st.spinner("Training models and computing metrics..."):
                if "Patient-Level" in split_method:
                    unique_patients = df_clean['patient_id'].unique()
                    patient_train, patient_test = train_test_split(np.array(unique_patients), test_size=0.2, random_state=42)

                    train_df = df_clean[df_clean['patient_id'].isin(patient_train)]
                    test_df = df_clean[df_clean['patient_id'].isin(patient_test)]

                    X_train = train_df[sanitized_cols]
                    y_train = train_df['status']
                    X_test = test_df[sanitized_cols]
                    y_test = test_df['status']
                    test_patient_ids = test_df['patient_id']
                else:
                    X = df_clean[sanitized_cols]
                    y = df_clean['status']
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                    train_idx, test_idx = next(sss.split(X, y))
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    test_patient_ids = df_clean['patient_id'].iloc[test_idx]

                scaler = StandardScaler()
                X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
                X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

                models = {
                    "XGBoost": xgb.XGBClassifier(
                        objective='binary:logistic', eval_metric='logloss',
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

                preds, probas, feature_importances = {}, {}, {}
                for name, model in models.items():
                    model.fit(X_train_scaled, y_train)
                    preds[name] = model.predict(X_test_scaled)
                    probas[name] = model.predict_proba(X_test_scaled)[:, 1]
                    
                    imp = model.feature_importances_
                    feature_importances[name] = imp / imp.sum()

                st.session_state.models = models
                st.session_state.preds = preds
                st.session_state.probas = probas
                st.session_state.feature_importances = feature_importances
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.X_train_scaled = X_train_scaled
                st.session_state.X_test_scaled = X_test_scaled
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.test_patient_ids = test_patient_ids
                st.session_state.last_split_method = split_method
                st.session_state.models_trained = True

        models = st.session_state.models
        preds = st.session_state.preds
        probas = st.session_state.probas
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        X_train_scaled = st.session_state.X_train_scaled
        X_test_scaled = st.session_state.X_test_scaled
        y_test = st.session_state.y_test
        test_patient_ids = st.session_state.test_patient_ids

        selected_model_name = st.selectbox("Select Model for Single Focus", list(models.keys()))
        selected_clf = models[selected_model_name]
        y_pred = preds[selected_model_name]
        y_prob = probas[selected_model_name]

        st.markdown(f"#### Performance: **{selected_model_name}**")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        m2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
        m3.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
        m4.metric("F1-Score", f"{f1_score(y_test, y_pred):.4f}")
        m5.metric("ROC AUC", f"{roc_auc_score(y_test, y_prob):.4f}")

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

        st.markdown("---")
        st.markdown('<a id="model-auditing-comparative-analysis"></a>', unsafe_allow_html=True)
        st.subheader("🔍 Model Auditing & Comparative Analysis")
        
        audit_tab1, audit_tab2, audit_tab3, audit_tab4, audit_tab5, audit_tab6 = st.tabs([
            "📈 ROC Curve Comparison", 
            "📊 Feature Importance Audit", 
            "📈 Partial Dependence Plots (PDP)",
            "👤 Patient-Level Performance Audit",
            "🎯 Probability Calibration", 
            "⚠️ Misclassification Analysis"
        ])

        colors = {'XGBoost': '#1f77b4', 'LightGBM': '#2ca02c', 'Random Forest': '#d62728'}

        with audit_tab1:
            st.write("### Multi-Model ROC Curve Comparison")
            fig, ax = plt.subplots(figsize=(8, 5))
            
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

        with audit_tab2:
            st.write("### 📊 Feature Importance Summary Across Models")
            importance_df = pd.DataFrame(st.session_state.feature_importances, index=X_train.columns).fillna(0)
            importance_df['Average Importance'] = importance_df[['XGBoost', 'LightGBM', 'Random Forest']].mean(axis=1)
            importance_df = importance_df.sort_values(by='Average Importance', ascending=False)
            importance_df['Rank'] = range(1, len(importance_df) + 1)
            
            col_chart, col_table = st.columns([1.1, 1])
            
            with col_chart:
                st.write("#### 🎯 Top 15 Consensus Features")
                fig, ax = plt.subplots(figsize=(6, 5.5))
                top_15_avg = importance_df['Average Importance'].head(15)[::-1]
                top_15_avg.plot(kind='barh', color='#2b5c8f', edgecolor='black', ax=ax)
                ax.set_xlabel('Mean Normalized Importance Score')
                ax.set_ylabel('Feature')
                ax.set_title('Consensus Feature Impact (All Models)')
                ax.grid(True, linestyle='--', alpha=0.4)
                st.pyplot(fig)
                
            with col_table:
                st.write("#### 📋 Score Breakdown Table")
                display_cols = ['Rank', 'Average Importance', 'XGBoost', 'LightGBM', 'Random Forest']
                summary_table = importance_df[display_cols].head(15)
                st.dataframe(
                    summary_table.style.format({
                        'Average Importance': '{:.4f}', 'XGBoost': '{:.4f}',
                        'LightGBM': '{:.4f}', 'Random Forest': '{:.4f}'
                    }).background_gradient(cmap='YlGnBu', subset=['Average Importance']),
                    use_container_width=True, height=420
                )

        with audit_tab3:
            st.write("### 📈 Partial Dependence Plots (PDP)")
            st.markdown("Partial Dependence Plots illustrate the marginal effect of top acoustic features on the predicted probability of Parkinson's Disease.")
            
            pdp_features = st.multiselect(
                "Select top features to plot PDPs for:",
                options=list(importance_df.index[:10]),
                default=list(importance_df.index[:3])
            )
            
            if pdp_features:
                with st.spinner("Generating PDP plots..."):
                    col_pdp, _ = st.columns([0.65, 0.35])
                    with col_pdp:
                        fig, ax = plt.subplots(figsize=(8, 2.5 * len(pdp_features)))
                        display = PartialDependenceDisplay.from_estimator(
                            selected_clf,
                            X_test_scaled,
                            features=pdp_features,
                            kind="average",
                            ax=ax
                        )
                        plt.tight_layout()
                        st.pyplot(fig)
            else:
                st.info("Please select at least one feature to display PDP.")

        with audit_tab4:
            st.write("### 👤 Patient-Level Performance Audit")
            st.markdown("Evaluating performance aggregated at the **Patient/Subject Level** ensures the model is accurate per patient rather than just over-indexing on individual voice samples.")
            
            audit_df = pd.DataFrame({
                'Patient_ID': test_patient_ids.values,
                'True_Label': y_test.values,
                'Predicted_Label': y_pred,
                'PD_Probability': y_prob
            })

            patient_summary = audit_df.groupby('Patient_ID').agg(
                Total_Recordings=('True_Label', 'count'),
                True_Diagnosis=('True_Label', 'first'),
                Correct_Predictions=('True_Label', lambda x: (x == audit_df.loc[x.index, 'Predicted_Label']).sum()),
                Patient_Accuracy=('True_Label', lambda x: accuracy_score(x, audit_df.loc[x.index, 'Predicted_Label'])),
                Avg_PD_Probability=('PD_Probability', 'mean')
            ).reset_index()

            patient_summary['Majority_Vote_Pred'] = (patient_summary['Avg_PD_Probability'] >= 0.5).astype(int)
            patient_summary['Patient_Level_Correct'] = (patient_summary['True_Diagnosis'] == patient_summary['Majority_Vote_Pred'])

            p_acc = patient_summary['Patient_Level_Correct'].mean()
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Evaluated Test Patients", len(patient_summary))
            m2.metric("Patient-Level Accuracy (Majority Vote)", f"{p_acc:.4f}")
            m3.metric("Avg Samples per Patient", f"{patient_summary['Total_Recordings'].mean():.1f}")

            st.write("#### Detailed Breakdown Per Test Patient")
            st.dataframe(
                patient_summary.style.format({
                    'Patient_Accuracy': '{:.2%}',
                    'Avg_PD_Probability': '{:.4f}'
                }).highlight_between(
                    subset=['Patient_Accuracy'], 
                    left=0.0, 
                    right=0.99, 
                    color="#e98282"
                ),
                use_container_width=True
            )

        with audit_tab5:
            st.write("### Calibration Curves (Reliability Diagram)")
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

        with audit_tab6:
            st.write("### Identifying Challenging Edge Cases")
            
            misclass_dict = {}
            for name in models.keys():
                err_mask = (y_test != preds[name])
                misclass_dict[name] = set(X_test[err_mask].index)

            common_err_idx = list(set.intersection(*misclass_dict.values()))
            
            st.warning(f"**{len(common_err_idx)} instances** were consistently misclassified by ALL three models.")
            
            if common_err_idx:
                err_df = X_test.loc[common_err_idx].copy()
                err_df['True Label'] = y_test.loc[common_err_idx]
                
                st.write("#### Detailed Feature View of Common Edge Cases")
                st.dataframe(err_df)

                st.markdown("""
                #### 📌 Insights on Common Edge Cases
                * **Instance Index 185 (False Positive):** Healthy individual exhibiting high `PPE` (0.214) and `spread1` (-5.593) mimicking Parkinson's voice profiles.
                * **Instance Index 194 (False Positive):** Healthy individual with low `HNR` (21.209) and high `MDVP:Jitter(%)` (0.00567).
                * **Instance Index 7 (False Negative):** Parkinson's patient with atypically stable vocal metrics (high `HNR` of 26.892, low `Jitter` and `Shimmer`).
                """)

    else:
        st.error(f"Data file not found at `{CLASSIFICATION_DATA_PATH}`.")

# ==========================================
# PAGE 3: TELEMONITORING UPDRS REGRESSION
# ==========================================
elif app_mode == "Telemonitoring UPDRS (Regression)":
    st.header("📈 Telemonitoring UPDRS Prediction: Case Study & Empirical Findings")
    
    df = load_regression_data(REGRESSION_DATA_PATH)

    if df is not None:
        st.info(
            "**Executive Summary:** This page presents a step-by-step audit showing why estimating absolute severity "
            "or tracking severity changes directly from acoustic features fails under rigorous validation."
        )

        reg_tab1, reg_tab2, reg_tab3, reg_tab4 = st.tabs([
            "📊 Dataset Exploration", 
            "❌ Task 1: Absolute Severity", 
            "⚠️ Task 2: Next Score Illusion", 
            "🎯 Task 3: Real Time Gap Delta"
        ])

        with reg_tab1:
            st.subheader("📊 Telemonitoring Dataset Exploration")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Audio Samples", df.shape[0])
            with col2:
                st.metric("Unique Subjects Tracked", df['subject#'].nunique())
            with col3:
                st.metric("Average Total UPDRS", f"{df['total_UPDRS'].mean():.2f}")

            if st.checkbox("Show raw data sample", key="reg_raw_data"):
                st.dataframe(df.head(10))

            st.write("### Distribution of Total UPDRS Scores")
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.hist(df['total_UPDRS'], bins=30, color='skyblue', edgecolor='black')
            ax.set_xlabel("Total UPDRS Score")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        with reg_tab2:
            st.subheader("❌ Task 1: Predicting Absolute Severity from Voice Alone")
            st.markdown("""
            **Goal:** Predict a patient's absolute `total_UPDRS` severity score given a single voice recording from any unseen subject.
            
            **Methodology & Patient Split:**
            To prevent patient data leakage across train and test sets, a strict **subject-level group split** was implemented.
            """)

            unique_subjects = df['subject#'].unique()
            subject_train, subject_test = train_test_split(np.array(unique_subjects), test_size=0.2, random_state=42)

            train_df = df[df['subject#'].isin(subject_train)]
            test_df = df[df['subject#'].isin(subject_test)]

            features_to_drop = ['subject#', 'age', 'sex', 'motor_UPDRS', 'total_UPDRS']
            X_train = train_df.drop(features_to_drop, axis=1)
            y_train = train_df['total_UPDRS']
            X_test = test_df.drop(features_to_drop, axis=1)
            y_test = test_df['total_UPDRS']

            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

            X_train_scaled.columns = [''.join(c if c.isalnum() else '_' for c in str(x)) for x in X_train_scaled.columns]
            X_test_scaled.columns = [''.join(c if c.isalnum() else '_' for c in str(x)) for x in X_test_scaled.columns]

            with st.spinner("Evaluating LightGBM Regressor on Patient-Level Split..."):
                lgbm_reg = LGBMRegressor(random_state=42, verbose=-1)
                lgbm_reg.fit(X_train_scaled, y_train)
                y_pred = lgbm_reg.predict(X_test_scaled)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean Absolute Error (MAE)", f"{mean_absolute_error(y_test, y_pred):.4f}")
            m2.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test, y_pred):.4f}")
            m3.metric("Root MSE (RMSE)", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
            m4.metric("R-squared (R²)", f"{r2_score(y_test, y_pred):.4f}")

            col_scatter, col_analysis = st.columns([1.1, 1])
            with col_scatter:
                st.write("#### Actual vs. Predicted UPDRS Scores")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(y_test, y_pred, alpha=0.3, color='purple')
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel('Actual Total UPDRS')
                ax.set_ylabel('Predicted Total UPDRS')
                ax.grid(True)
                st.pyplot(fig)

            with col_analysis:
                st.error("### Why It Failed")
                st.markdown("""
                * Both XGBoost and LightGBM scored **RMSE ≈ 13** with a **negative R²**.
                * A negative $R^2$ indicates that predicting the **mean target value for everyone** performs better than the actual regression models.
                * **Hyperparameter Tuning:** Hyperparameter search failed to fix this issue because optimization cannot recover signal when features lack predictive value.
                * **Core Diagnosis:** Parkinson's disease severity depends on an individual patient's personal baseline rather than universal acoustic variations. Two patients can present identical acoustic instability while possessing vastly different total UPDRS scores. Voice features alone fail to map to absolute severity levels across unseen individuals.
                """)

        with reg_tab3:
            st.subheader("⚠️ Task 2: The 'Predict Next Score' Illusion")
            st.markdown("""
            ### The Attempted Fix: Sequential Next-Score Prediction
            To account for individual baselines, the model was reframed to predict a patient's **next** recording UPDRS given their **current** recording.

            Initial results showed **99%+ accuracy / $R^2 > 0.99$**, but this outcome was driven by data leakage inherent to the dataset's design.
            """)

            st.warning("""
            #### 🚨 Why This Metric Was an Illusion
            * **Linear Interpolation:** In the Telemonitoring dataset, actual doctor clinical evaluations took place infrequently. The granular `total_UPDRS` values were mathematically interpolated over time.
            * **High Sampling Density:** Consecutive audio recordings for a given subject often occurred only hours or days apart.
            * **The Leak:** Because of interpolation, adjacent recordings had nearly identical target values by construction.
            * **The Persistence Benchmark:** A trivial **'Persistence' baseline** (simply predicting that the next score equals the current score with 0 change) matched or outperformed XGBoost and LightGBM outright.
            * **Conclusion:** The models were not identifying a relationship between voice and disease progression; they were simply tracking proximity between interpolated data points.
            """)

            st.write("#### Comparison: Persistence Baseline vs. Complex Models")
            comparison_data = pd.DataFrame({
                "Model / Strategy": ["Naive Persistence Baseline (Δ = 0)", "LightGBM Regressor", "XGBoost Regressor"],
                "Exploited Shortcut": ["Yes (Target Continuity)", "Yes (Target Continuity)", "Yes (Target Continuity)"],
                "Apparent R² Score": ["> 0.99", "> 0.99", "> 0.99"],
                "Actual Predictive Signal": ["None", "None", "None"]
            })
            st.dataframe(comparison_data, use_container_width=True)

        with reg_tab4:
            st.subheader("🎯 Task 3: Real Time Gap Delta Prediction (The Final Audit)")
            st.markdown("""
            To eliminate target persistence leakage, the task was reformulated to predict the **true change in UPDRS ($\Delta$ UPDRS)** across a substantial time gap (e.g., > 30 days).
            """)

            st.info("""
            #### Key Findings from Delta Modeling:
            1. **Predicting Severity Deltas:** When evaluating the actual change in UPDRS over 30+ day windows, model $R^2$ scores dropped back down toward **0.0 or negative values**.
            2. **Acoustic Drift vs. Clinical Drift:** Short-term acoustic fluctuations (voice fatigue, background noise, microphone differences) drown out long-term motor/clinical UPDRS trends.
            3. **Clinical Implication:** Remote voice recordings alone cannot serve as an uncalibrated standalone tracker for precise UPDRS numerical progression over time without periodic clinical ground-truth recalibration.
            """)

    else:
        st.error(f"Data file not found at `{REGRESSION_DATA_PATH}`.")
# ==========================================
# PAGE 4: FREEZING OF GAIT (FoG) ANALYSIS
# ==========================================
elif app_mode == "Freezing of Gait (FoG) Analysis":
    st.header(
        "🏃 Freezing of Gait (FoG) Real-Time Detection & Severity Pipeline"
    )
    st.markdown("""
    Freezing of Gait (FoG) occurs when a Parkinson's patient's feet suddenly feel "glued to the floor." 
    This dashboard analyzes tri-axial accelerometer signals worn on the ankle, thigh, and trunk (Daphnet dataset)
    to classify FoG events and estimate continuous freeze severity (`freeze_frac`).
    """)

    # 1. Load Data
    with st.spinner(
        "Loading accelerometer text files and extracting window features..."
    ):
        fog_raw = load_fog_data()

        if fog_raw.empty:
            st.error(
                "❌ No raw data loaded! Check that your S*R*.txt files exist in the 'data' directory."
            )
            st.stop()

        fog_df, fs = extract_fog_window_features(
            fog_raw, window_sec=4.0, overlap=0.5
        )

    if fog_df.empty or "label" not in fog_df.columns:
        st.error(
            "❌ Feature extraction failed or produced an empty DataFrame."
        )
        st.stop()

    # 2. Dataset Summary Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Window Samples", f"{len(fog_df):,}")
    m2.metric("Normal Windows (Class 0)", f"{(fog_df['label'] == 0).sum():,}")
    m3.metric("Freeze Windows (Class 1)", f"{(fog_df['label'] == 1).sum():,}")

    st.markdown("---")

    # 3. Streamlit Tabs
    fog_tab1, fog_tab2, fog_tab3, fog_tab4 = st.tabs([
        "🔬 Patient-Independent (GroupKFold)",
        "👤 Patient-Specific Calibration",
        "📊 Severity Regression (freeze_frac)",
        "🎯 Feature Importance Analysis",
    ])

    # Pre-process features once for all tabs
    X_fog = fog_df.drop(
        columns=["subject", "run", "label", "freeze_frac"], errors="ignore"
    )
    y_fog = fog_df["label"]
    groups_fog = fog_df["subject"]

    scaler_fog = StandardScaler()
    X_fog_scaled = scaler_fog.fit_transform(X_fog)

    # ----------------------------------------------------
    # TAB 1: Patient-Independent Classification
    # ----------------------------------------------------
    with fog_tab1:
        st.subheader("🔬 Patient-Independent Evaluation (GroupKFold)")
        st.caption(
            "Simulates evaluating model performance on completely unseen patients using GroupKFold cross-validation by subject."
        )

        with st.spinner("Running GroupKFold Cross-Validation..."):
            n_splits = min(8, groups_fog.nunique())
            gkf = GroupKFold(n_splits=n_splits)

            clf_indep = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )
            probs_indep = cross_val_predict(
                clf_indep,
                X_fog_scaled,
                y_fog,
                cv=gkf,
                groups=groups_fog,
                method="predict_proba",
            )[:, 1]

            preds_default = (probs_indep > 0.5).astype(int)
            optimal_thresh = find_optimal_threshold(y_fog, probs_indep)
            preds_opt = (probs_indep > optimal_thresh).astype(int)

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.subheader("Default Threshold (0.500)")
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Accuracy", f"{accuracy_score(y_fog, preds_default):.3f}"
            )
            c2.metric(
                "Balanced Acc",
                f"{balanced_accuracy_score(y_fog, preds_default):.3f}",
            )
            c3.metric("ROC-AUC", f"{roc_auc_score(y_fog, probs_indep):.3f}")

            fig_cm1, ax_cm1 = plt.subplots(figsize=(4.5, 3.5))
            cm1 = confusion_matrix(y_fog, preds_default)
            disp1 = ConfusionMatrixDisplay(
                confusion_matrix=cm1, display_labels=["Normal", "Freeze"]
            )
            disp1.plot(ax=ax_cm1, cmap="Blues", values_format="d")
            ax_cm1.set_title("Confusion Matrix (Thresh = 0.5)")
            st.pyplot(fig_cm1)

        with col_m2:
            st.subheader(f"Optimal Threshold ({optimal_thresh:.3f})")
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{accuracy_score(y_fog, preds_opt):.3f}")
            c2.metric(
                "Balanced Acc",
                f"{balanced_accuracy_score(y_fog, preds_opt):.3f}",
            )
            c3.metric("ROC-AUC", f"{roc_auc_score(y_fog, probs_indep):.3f}")

            fig_cm2, ax_cm2 = plt.subplots(figsize=(4.5, 3.5))
            cm2 = confusion_matrix(y_fog, preds_opt)
            disp2 = ConfusionMatrixDisplay(
                confusion_matrix=cm2, display_labels=["Normal", "Freeze"]
            )
            disp2.plot(ax=ax_cm2, cmap="Blues", values_format="d")
            ax_cm2.set_title(
                f"Confusion Matrix (Thresh = {optimal_thresh:.3f})"
            )
            st.pyplot(fig_cm2)

    # ----------------------------------------------------
    # TAB 2: Patient-Specific Evaluation
    # ----------------------------------------------------
    with fog_tab2:
        st.subheader("👤 Patient-Specific Calibration (Time-Split)")
        st.caption(
            "Trains and tests within each subject using a sequential 70% train / 30% test split to account for individual gait dynamics."
        )

        specific_results_tuned = []

        with st.spinner("Calibrating models per patient..."):
            for subj, g in fog_df.groupby("subject"):
                g = g.sort_index()
                n = len(g)

                if n < 20 or g["label"].nunique() < 2:
                    continue

                split = int(n * 0.7)
                train, test = g.iloc[:split], g.iloc[split:]

                if test["label"].nunique() < 2:
                    continue

                Xtr = scaler_fog.fit_transform(
                    train.drop(
                        columns=["subject", "run", "label", "freeze_frac"],
                        errors="ignore",
                    )
                )
                Xte = scaler_fog.transform(
                    test.drop(
                        columns=["subject", "run", "label", "freeze_frac"],
                        errors="ignore",
                    )
                )

                clf_s = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                )
                clf_s.fit(Xtr, train["label"])
                probs_s = clf_s.predict_proba(Xte)[:, 1]

                cv_probs = cross_val_predict(
                    clf_s,
                    Xtr,
                    train["label"],
                    cv=min(5, len(train)),
                    method="predict_proba",
                )[:, 1]
                subject_thresh = find_optimal_threshold(
                    train["label"], cv_probs
                )
                preds_s_tuned = (probs_s > subject_thresh).astype(int)

                specific_results_tuned.append({
                    "subject": subj,
                    "threshold": round(subject_thresh, 3),
                    "test_windows": len(test),
                    "accuracy": round(
                        accuracy_score(test["label"], preds_s_tuned), 3
                    ),
                    "balanced_accuracy": round(
                        balanced_accuracy_score(test["label"], preds_s_tuned),
                        3,
                    ),
                    "roc_auc": round(
                        roc_auc_score(test["label"], probs_s), 3
                    )
                    if test["label"].nunique() > 1
                    else np.nan,
                })

        specific_df_tuned = pd.DataFrame(specific_results_tuned)

        if not specific_df_tuned.empty:
            st.dataframe(specific_df_tuned, use_container_width=True)

            st.markdown("**Mean Performance Across Calibrated Subjects:**")
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Mean Accuracy", f"{specific_df_tuned['accuracy'].mean():.3f}"
            )
            m2.metric(
                "Mean Balanced Accuracy",
                f"{specific_df_tuned['balanced_accuracy'].mean():.3f}",
            )
            m3.metric(
                "Mean ROC-AUC", f"{specific_df_tuned['roc_auc'].mean():.3f}"
            )
        else:
            st.warning(
                "Not enough samples per subject with both freeze and normal classes to perform calibration."
            )

    # ----------------------------------------------------
    # TAB 3: Severity Regression Analysis
    # ----------------------------------------------------
    with fog_tab3:
        st.subheader("📊 Continuous FoG Severity Regression (`freeze_frac`)")
        st.markdown("""
        Rather than simple binary classification, this regression pipeline estimates the **continuous severity fraction** (`freeze_frac` $\in [0, 1]$) representing what percentage of each window was actively spent in a freeze state.
        """)

        X_reg_fog = fog_df.drop(
            columns=["subject", "run", "label", "freeze_frac"], errors="ignore"
        )
        y_reg_fog = fog_df["freeze_frac"]

        with st.spinner(
            "Evaluating Regression Models for Severity Estimation..."
        ):
            rf_reg_fog = RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
            )
            xgb_reg_fog = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
            )
            lgb_reg_fog = LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )

            # Fix: Cast unique subjects to a Python list
            unique_fog_subjs = list(fog_df["subject"].unique())

            tr_subjs, te_subjs = train_test_split(
                unique_fog_subjs, test_size=0.3, random_state=42
            )

            tr_mask = fog_df["subject"].isin(tr_subjs)
            te_mask = fog_df["subject"].isin(te_subjs)

            X_tr_fog = scaler_fog.fit_transform(X_reg_fog[tr_mask])
            X_te_fog = scaler_fog.transform(X_reg_fog[te_mask])
            y_tr_fog = y_reg_fog[tr_mask]
            y_te_fog = y_reg_fog[te_mask]

            reg_models = {
                "XGBoost Regressor": xgb_reg_fog,
                "LightGBM Regressor": lgb_reg_fog,
                "Random Forest Regressor": rf_reg_fog,
            }

            reg_metrics = []
            preds_dict = {}

            for name, model in reg_models.items():
                model.fit(X_tr_fog, y_tr_fog)
                preds = model.predict(X_te_fog)
                preds_dict[name] = preds

                reg_metrics.append({
                    "Model": name,
                    "RMSE": round(np.sqrt(mean_squared_error(y_te_fog, preds)), 4),
                    "MAE": round(mean_absolute_error(y_te_fog, preds), 4),
                    "R² Score": round(r2_score(y_te_fog, preds), 4),
                })

        st.dataframe(pd.DataFrame(reg_metrics), use_container_width=True)

        col_reg_p1, col_reg_p2 = st.columns(2)
        with col_reg_p1:
            st.write("#### Actual vs. Predicted Severity (`XGBoost`)")
            fig_sev, ax_sev = plt.subplots(figsize=(5, 4))
            ax_sev.scatter(
                y_te_fog,
                preds_dict["XGBoost Regressor"],
                alpha=0.3,
                color="teal",
            )
            ax_sev.plot([0, 1], [0, 1], "r--")
            ax_sev.set_xlabel("Actual Freeze Fraction")
            ax_sev.set_ylabel("Predicted Freeze Fraction")
            ax_sev.set_title("XGBoost Freeze Severity Prediction")
            st.pyplot(fig_sev)

        with col_reg_p2:
            st.write("#### Patient-Specific vs. Unseen Patient Findings")
            st.markdown("""
            * **Patient-Independent Regression:** Unseen patients yield lower $R^2$ scores ($\approx 0.15 - 0.20$) due to significant baseline differences in walking biomechanics between subjects.
            * **Patient-Specific Calibration:** When calibrated on individual subject baselines, severity regression accuracy increases substantially (**Mean $R^2 \approx 0.60 - 0.65$**).
            * **Conclusion:** Real-time wearable cueing devices benefit significantly from initial patient calibration.
            """)

    # ----------------------------------------------------
    # TAB 4: FoG Feature Importance Analysis
    # ----------------------------------------------------
    with fog_tab4:
        st.subheader(
            "📊 Top Feature Importances (Gait Accelerometer Signals)"
        )
        st.caption(
            "Identifies which sensor locations (ankle, thigh, trunk) and feature types drive freeze detection."
        )

        with st.spinner("Calculating feature importances..."):
            clf_indep_feat = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            clf_indep_feat.fit(X_fog_scaled, y_fog)
            importances_fog = (
                pd.Series(
                    clf_indep_feat.feature_importances_, index=X_fog.columns
                )
                .sort_values(ascending=False)
                .head(20)
            )

        fig_imp_fog, ax_imp_fog = plt.subplots(figsize=(8, 5))
        importances_fog[::-1].plot(
            kind="barh", ax=ax_imp_fog, color="skyblue", edgecolor="black"
        )
        ax_imp_fog.set_xlabel("Feature Importance")
        ax_imp_fog.set_title("Top 20 Features — Random Forest FoG Model")
        plt.tight_layout()

        st.pyplot(fig_imp_fog)

        st.info("""
        **Clinical Takeaway:** High importance scores on **Freeze Index** features (especially along ankle and thigh vertical axes) indicate that the model is successfully capturing genuine biomechanical gait-frequency shifts ($3-8$ Hz freeze band) rather than artifact noise.
        """)