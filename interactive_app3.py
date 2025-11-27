import numpy as np
import pandas as pd
import streamlit as st
import joblib
import tensorflow as tf

from sklearn.metrics import roc_auc_score, brier_score_loss

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="CVD Risk – Predictor v3 (Stacking GenAI)",
    layout="wide"
)

DATA_PATH = "framingham_minimal.csv"

RISK_HORIZON_YEARS = 10
RRR_STATIN = 0.25   # 25% relative risk reduction
RRR_BP = 0.20       # 20% relative risk reduction

# These are the 11 inputs you defined
UI_FEATURES = [
    "gender",
    "age",
    "total_cholesterol",
    "hdl_cholesterol",
    "sbp",
    "diabetes",
    "smoker",
    "egfr",
    "anti_htn_med",
    "statin_use",
    "bmi",
]

TARGET_COL = "cvd_event"


# -------------------------------------------------------------------
# UTILS
# -------------------------------------------------------------------
def format_percent(p: float) -> str:
    return f"{p * 100:.1f}%"

def risk_category_label(p: float):
    if p < 0.05:
        return "Low risk (<5%) – lifestyle focus", "green"
    elif p < 0.075:
        return "Borderline (5–7.4%) – consider therapy", "orange"
    elif p < 0.20:
        return "Intermediate (7.5–19.9%) – likely benefit from therapy", "orange"
    else:
        return "High (≥20%) – strong indication for therapy", "red"

def treatment_impact(baseline_risk: float, rrr: float):
    # baseline_risk and rrr are decimals (0–1)
    treated_risk = baseline_risk * (1.0 - rrr)
    arr = baseline_risk - treated_risk
    nnt = None if arr <= 0 else 1.0 / arr
    return treated_risk, arr, nnt

def get_percentile(prob: float, dist: np.ndarray) -> float:
    """Return percentile of prob within distribution dist."""
    return float((dist <= prob).mean())

def jitter_samples(row: pd.DataFrame, n: int = 200):
    """Generate jittered versions of a single-row DataFrame."""
    rng = np.random.default_rng(42)
    rows = pd.concat([row] * n, ignore_index=True)

    # Numeric jitter with safe clinical ranges
    rows["age"] = np.clip(rows["age"] + rng.normal(0, 1.5, n), 30, 80).round()
    rows["total_cholesterol"] = np.clip(
        rows["total_cholesterol"] + rng.normal(0, 10, n), 130, 320
    ).round()
    rows["hdl_cholesterol"] = np.clip(
        rows["hdl_cholesterol"] + rng.normal(0, 3, n), 20, 90
    ).round()
    rows["sbp"] = np.clip(rows["sbp"] + rng.normal(0, 5, n), 90, 220).round()
    rows["bmi"] = np.clip(rows["bmi"] + rng.normal(0, 1.0, n), 16, 45)
    rows["egfr"] = np.clip(rows["egfr"] + rng.normal(0, 5, n), 20, 120)

    # Binary fields (diabetes, smoker, anti_htn_med, statin_use) kept fixed
    return rows


# -------------------------------------------------------------------
# LOAD DATA & MODELS
# -------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_minimal_data(path: str):
    df = pd.read_csv(path)
    # Make sure expected columns are there
    df = df[UI_FEATURES + [TARGET_COL]].dropna()
    df["cvd_event"] = df[TARGET_COL].astype(int)
    return df

@st.cache_resource(show_spinner=True)
def load_models_and_metadata():
    # Load models
    scaler = joblib.load("stack_scaler.pkl")
    rf_model = joblib.load("stack_rf.pkl")
    xgb_model = joblib.load("stack_xgb.pkl")
    meta_model = joblib.load("stack_meta.pkl")
    cnn_model = tf.keras.models.load_model("stack_cnn.h5")

    # Load minimal dataset for distribution & feature layout
    df = load_minimal_data(DATA_PATH)
    y = df[TARGET_COL].values

    # One-hot encode gender for training layout
    X_raw = df[UI_FEATURES].copy()
    X_encoded = pd.get_dummies(X_raw, columns=["gender"], drop_first=True)
    feature_cols = X_encoded.columns.tolist()

    # Align & scale
    X_scaled = scaler.transform(X_encoded[feature_cols])
    X_cnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    # Base model predictions
    rf_probs = rf_model.predict_proba(X_scaled)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_scaled)[:, 1]
    cnn_probs = cnn_model.predict(X_cnn).ravel()

    # Stacking meta-learner
    meta_X = np.column_stack([rf_probs, xgb_probs, cnn_probs])
    stack_probs = meta_model.predict_proba(meta_X)[:, 1]

    # Metrics
    metrics = {}
    for label, probs in [
        ("Random Forest", rf_probs),
        ("XGBoost", xgb_probs),
        ("CNN", cnn_probs),
        ("Stacking GenAI", stack_probs),
    ]:
        auc = roc_auc_score(y, probs)
        brier = brier_score_loss(y, probs)
        metrics[label] = {"AUC": auc, "Brier": brier}

    prevalence = y.mean()

    return {
        "scaler": scaler,
        "rf": rf_model,
        "xgb": xgb_model,
        "cnn": cnn_model,
        "meta": meta_model,
        "feature_cols": feature_cols,
        "df": df,
        "y": y,
        "rf_probs": rf_probs,
        "xgb_probs": xgb_probs,
        "cnn_probs": cnn_probs,
        "stack_probs": stack_probs,
        "metrics": metrics,
        "prevalence": prevalence,
    }

def encode_for_model(df_small: pd.DataFrame, feature_cols):
    """
    Encode a DataFrame with columns == UI_FEATURES
    into the same numeric layout used in training.
    """
    temp = df_small.copy()
    temp = pd.get_dummies(temp, columns=["gender"], drop_first=True)
    # Align to training feature_cols, fill missing with 0
    temp = temp.reindex(columns=feature_cols, fill_value=0)
    return temp


def predict_stack_single(models_meta, row: pd.DataFrame) -> float:
    """Predict CVD risk for a single-row UI-feature DataFrame using stacking model."""
    scaler = models_meta["scaler"]
    rf_model = models_meta["rf"]
    xgb_model = models_meta["xgb"]
    cnn_model = models_meta["cnn"]
    meta_model = models_meta["meta"]
    feature_cols = models_meta["feature_cols"]

    X_enc = encode_for_model(row, feature_cols)
    X_scaled = scaler.transform(X_enc)
    X_cnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    rf_p = rf_model.predict_proba(X_scaled)[:, 1]
    xgb_p = xgb_model.predict_proba(X_scaled)[:, 1]
    cnn_p = cnn_model.predict(X_cnn).ravel()

    meta_X = np.column_stack([rf_p, xgb_p, cnn_p])
    stack_p = meta_model.predict_proba(meta_X)[:, 1][0]
    return float(stack_p)


def predict_stack_bulk(models_meta, df_rows: pd.DataFrame) -> np.ndarray:
    """Predict for many rows at once (used for distribution & jitter)."""
    scaler = models_meta["scaler"]
    rf_model = models_meta["rf"]
    xgb_model = models_meta["xgb"]
    cnn_model = models_meta["cnn"]
    meta_model = models_meta["meta"]
    feature_cols = models_meta["feature_cols"]

    X_enc = encode_for_model(df_rows, feature_cols)
    X_scaled = scaler.transform(X_enc)
    X_cnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    rf_p = rf_model.predict_proba(X_scaled)[:, 1]
    xgb_p = xgb_model.predict_proba(X_scaled)[:, 1]
    cnn_p = cnn_model.predict(X_cnn).ravel()
    meta_X = np.column_stack([rf_p, xgb_p, cnn_p])
    stack_p = meta_model.predict_proba(meta_X)[:, 1]
    return stack_p


# -------------------------------------------------------------------
# LOAD EVERYTHING
# -------------------------------------------------------------------
MODELS_META = load_models_and_metadata()
STACK_DIST = MODELS_META["stack_probs"]
METRICS = MODELS_META["metrics"]
PREVALENCE = MODELS_META["prevalence"]


# -------------------------------------------------------------------
# SIDEBAR – PATIENT INPUTS
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Patient Inputs")

    gender = st.radio("Gender", ["Male", "Female"], index=0, horizontal=True)
    age = st.slider("Age (years)", 30, 80, 55)

    total_cholesterol = st.slider("Total cholesterol (mg/dL)", 130, 320, 200)
    hdl_cholesterol = st.slider("HDL cholesterol (mg/dL)", 20, 90, 45)

    sbp = st.slider("Systolic BP (mmHg)", 90, 220, 130)

    diabetes = st.radio("Diabetes", ["No", "Yes"], index=0, horizontal=True)
    smoker = st.radio("Current smoker", ["No", "Yes"], index=0, horizontal=True)

    egfr = st.slider("eGFR (mL/min/1.73 m²)", 20, 120, 90)

    anti_htn_med = st.radio("Using anti-hypertensive medication", ["No", "Yes"], index=0, horizontal=True)
    statin_use = st.radio("Using statin therapy", ["No", "Yes"], index=0, horizontal=True)

    bmi = st.slider("BMI (kg/m²)", 16.0, 45.0, 27.0, step=0.1)

    st.markdown("---")
    show_advanced = st.checkbox("Show advanced metrics", value=True)


# Build a single-row DataFrame with UI features
input_row = pd.DataFrame([{
    "gender": gender,
    "age": float(age),
    "total_cholesterol": float(total_cholesterol),
    "hdl_cholesterol": float(hdl_cholesterol),
    "sbp": float(sbp),
    "diabetes": 1 if diabetes == "Yes" else 0,
    "smoker": 1 if smoker == "Yes" else 0,
    "egfr": float(egfr),
    "anti_htn_med": 1 if anti_htn_med == "Yes" else 0,
    "statin_use": 1 if statin_use == "Yes" else 0,
    "bmi": float(bmi),
}])

# -------------------------------------------------------------------
# MAIN – PREDICTION
# -------------------------------------------------------------------
st.title("CVD 10-Year Risk – Predictor v3 (Minimal Features + Stacking GenAI)")
st.caption(
    "This tool estimates 10-year cardiovascular risk using a Stacking Generative AI model "
    "trained on a minimal Framingham-based dataset (age, lipids, BP, diabetes, smoking, BMI, "
    "eGFR, medications). Educational use only – not a medical device."
)

left, right = st.columns([1.2, 1.8])

with st.spinner("Computing risk, uncertainty, and NNT..."):
    # Single-patient prediction
    risk = predict_stack_single(MODELS_META, input_row)

    # Local uncertainty with jitter
    jitter_df = jitter_samples(input_row, n=200)
    jitter_probs = predict_stack_bulk(MODELS_META, jitter_df)
    low_u, med_u, high_u = np.percentile(jitter_probs, [5, 50, 95])

    # Percentile vs training cohort
    percentile = get_percentile(risk, STACK_DIST)

risk_label, risk_color = risk_category_label(risk)

# -------------------------------------------------------------------
# LEFT COLUMN – INPUT SUMMARY & MODEL INFO
# -------------------------------------------------------------------
with left:
    st.subheader("Input Summary")

    st.markdown(
        f"- **Gender:** {gender}  \n"
        f"- **Age:** {age} years  \n"
        f"- **Total cholesterol:** {total_cholesterol} mg/dL  \n"
        f"- **HDL cholesterol:** {hdl_cholesterol} mg/dL  \n"
        f"- **Systolic BP:** {sbp} mmHg  \n"
        f"- **Diabetes:** {diabetes}  \n"
        f"- **Smoker:** {smoker}  \n"
        f"- **eGFR:** {egfr} mL/min/1.73 m²  \n"
        f"- **Anti-hypertensive meds:** {anti_htn_med}  \n"
        f"- **Statin use:** {statin_use}  \n"
        f"- **BMI:** {bmi:.1f} kg/m²"
    )

    st.markdown("---")
    st.markdown("### Model Performance (on training cohort)")

    m = METRICS["Stacking GenAI"]
    st.markdown(
        f"- **Model:** Stacking GenAI (RF + XGB + CNN + meta LR)  \n"
        f"- **ROC AUC:** **{m['AUC']:.3f}**  \n"
        f"- **Brier score:** **{m['Brier']:.3f}**  \n"
        f"- **Outcome prevalence:** {format_percent(PREVALENCE)}"
    )

    if show_advanced:
        st.markdown("---")
        st.markdown("### Base Model Metrics")
        st.markdown(
            f"- **Random Forest AUC:** {METRICS['Random Forest']['AUC']:.3f}  \n"
            f"- **XGBoost AUC:** {METRICS['XGBoost']['AUC']:.3f}  \n"
            f"- **CNN AUC:** {METRICS['CNN']['AUC']:.3f}"
        )

    st.markdown("---")
    st.markdown("### Risk Context in Cohort")
    st.markdown(
        f"- **Predicted {RISK_HORIZON_YEARS}-year risk:** {format_percent(risk)}  \n"
        f"- **Percentile vs training cohort:** ~**{percentile * 100:.0f}th** percentile"
    )

# -------------------------------------------------------------------
# RIGHT COLUMN – RISK, UNCERTAINTY, NNT
# -------------------------------------------------------------------
with right:
    st.subheader("10-Year CVD Risk")

    st.markdown(
        f"<h2 style='color:{risk_color}; margin-bottom:0;'>{format_percent(risk)}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**{risk_label}**")

    st.caption(
        "This risk is estimated using a Stacking GenAI model trained on a Framingham-based "
        "dataset with GAN-balanced outcomes and 11 key risk factors. It is intended only for "
        "education and research discussion."
    )

    if show_advanced:
        st.markdown("#### Local Uncertainty (Jittered Inputs)")
        st.markdown(
            f"- **5–95% band:** {format_percent(low_u)} – {format_percent(high_u)}  \n"
            f"- **Median under jitter:** {format_percent(med_u)}"
        )
        st.caption(
            "We create 200 slight variations of age, lipids, BP, BMI, and eGFR around the "
            "current values and recompute risk. This shows how sensitive the prediction is "
            "to small measurement changes."
        )

    st.markdown("### Treatment Impact & NNT (Illustrative)")

    st.markdown(
        f"Assuming a **{RISK_HORIZON_YEARS}-year** time horizon and typical relative risk "
        "reductions from large trials. Values are approximate and for educational use only."
    )

    rows = []
    for name, rrr, desc in [
        ("High-intensity statin", RRR_STATIN,
         "Lowers LDL and reduces major CVD events in many trials."),
        ("Optimized BP therapy", RRR_BP,
         "Achieving better BP control with guideline-directed therapy."),
    ]:
        treated_risk, arr, nnt = treatment_impact(risk, rrr)
        if nnt is None:
            nnt_str = "Not meaningful"
        elif nnt > 2000:
            nnt_str = "> 2000"
        else:
            nnt_str = f"{nnt:.0f}"

        rows.append({
            "Therapy": name,
            "Baseline risk": format_percent(risk),
            "Risk with therapy (approx.)": format_percent(treated_risk),
            "Absolute risk reduction\n(percentage points)": f"{arr * 100:.1f}",
            f"NNT over {RISK_HORIZON_YEARS} yrs\n(patients)": nnt_str,
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if show_advanced:
        st.caption(
            "NNT (Number Needed to Treat) is 1 / ARR, where ARR is the absolute difference "
            "between baseline and treated risk. For example, NNT=25 means that treating "
            "25 patients with a similar profile for the specified duration prevents about "
            "one additional CVD event on average."
        )

    # ===========================
    # Explanation: Why results differ from MDCalc
    # ===========================
    with st.expander("Why results differ from MDCalc?"):
        st.markdown("""
    ### Why Your Results May Differ from MDCalc

    MDCalc uses the SCORE2 equation, a guideline-calibrated model based on large European cohorts.  
    Your Stacking GenAI model is trained on a GAN-balanced Framingham dataset and behaves differently for several reasons:

    **1. Population Basis**
    - **MDCalc (SCORE2):** higher European baseline risk  
    - **Predictor v3:** U.S. Framingham + GAN-balanced  

    **2. Medication Interpretation**
    - MDCalc reduces predicted risk if medications (statins or BP meds) are used  
    - Predictor v3 learned real-world patterns: medication use often signals long-standing disease or more severe risk profiles  

    **3. Nonlinear Modeling**
    - MDCalc uses fixed linear/logarithmic equations  
    - Predictor v3 captures nonlinear interactions (e.g., diabetes + smoking + high SBP), producing sharper increases in risk  

    **4. Sensitivity**
    - MDCalc provides conservative, smoothed estimates  
    - Predictor v3 is more sensitive and individualized based on combined risk factor patterns  

    **Conclusion:**  
    MDCalc is best for guideline-aligned population estimates.  
    Predictor v3 is best for individualized, data-driven predictions with high accuracy (AUC=0.93).
    """)

# -------------------------------------------------------------------
# FOOTER / ABOUT
# -------------------------------------------------------------------
with st.expander("About this model and comparison to MDCalc-style tools"):
    st.markdown(
        """
        ### Inputs and design

        This app uses a **minimal feature set** inspired by clinical risk calculators:

        - Gender (Male/Female)  
        - Age  
        - Total cholesterol and HDL  
        - Systolic blood pressure (SBP)  
        - Diabetes (yes/no)  
        - Current smoker (yes/no)  
        - eGFR (synthetic but clinically plausible kidney function marker)  
        - Use of anti-hypertensive medications (yes/no)  
        - Statin use (yes/no)  
        - BMI  

        The outcome is a binary **10-year CVD event** (cvd_event).

        ### Model architecture

        - Data derived and simplified from the **Framingham Heart Study**.  
        - Outcomes are GAN-balanced to improve representation of CVD events.  
        - Base models: **Random Forest, XGBoost, and a 1D CNN** on scaled features.  
        - A **meta-learner (Logistic Regression)** combines base model predictions into a
          **Stacking GenAI ensemble**.

        This Stacking GenAI model achieved an AUC of about **0.93** on held-out data in your
        training run.

        ### Disclaimer

        - This app is for **education, prototyping, and research discussion only**.  
        - It is **not** a regulated medical device and must not be used for individual
          patient care or treatment decisions.
        """
    )

st.markdown(
    "<p style='text-align:center; color:gray; margin-top:2rem;'>"
    "© 2025. Howard Nguyen, PhD – Stacking Generative AI CVD Risk Prototype (Demonstration Use Only)"
    "</p>",
    unsafe_allow_html=True,
)
