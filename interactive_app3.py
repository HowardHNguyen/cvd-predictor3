import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="CVD 10-Year Risk – Predictor v3 (Synthetic 10K + NNT)",
    layout="wide"
)

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
DATA_PATH = "synthetic_cvd_10000.csv"  # <--- your new dataset

RISK_HORIZON_YEARS = 10

# Relative risk reductions (RRR) for treatment impact / NNT
RRR_STATIN = 0.25          # ~25% relative reduction
RRR_BP = 0.20              # ~20% relative reduction (optimized BP meds)

# ============================================================
# Helper functions
# ============================================================
def format_percent(p: float) -> str:
    return f"{p * 100:.1f}%"


def get_risk_category(p: float):
    """
    Basic 10-year ASCVD-style categories.
    """
    if p < 0.05:
        return "Low risk (<5%) – lifestyle focus", "green"
    elif p < 0.075:
        return "Borderline (5–7.4%) – consider therapy", "orange"
    elif p < 0.20:
        return "Intermediate (7.5–19.9%) – likely benefit from therapy", "orange"
    else:
        return "High (≥20%) – strong indication for therapy", "red"


def treatment_impact(baseline_risk: float, rrr: float, horizon_years: int = 10):
    """
    Simple treatment impact model:
    - baseline_risk: model-predicted probability (0–1)
    - rrr: relative risk reduction (0–1)
    Returns treated risk, ARR, NNT.
    """
    treated_risk = baseline_risk * (1.0 - rrr)
    arr = baseline_risk - treated_risk

    if arr <= 0:
        nnt = None
    else:
        nnt = 1.0 / arr

    return treated_risk, arr, nnt


def local_jitter_uncertainty(model, base_row: pd.DataFrame, n_samples: int = 200):
    """
    Approximate local uncertainty by jittering numeric predictors slightly
    and recomputing risk N times. Returns (low, median, high) 5–50–95%.
    """
    rows = pd.concat([base_row] * n_samples, ignore_index=True)

    rng = np.random.default_rng(42)

    # Jitter numeric fields
    rows["age"] = np.clip(
        rows["age"] + rng.normal(0, 1, n_samples), 30, 80
    ).round().astype(int)

    rows["total_cholesterol"] = np.clip(
        rows["total_cholesterol"] + rng.normal(0, 10, n_samples), 120, 320
    ).round().astype(int)

    rows["hdl_cholesterol"] = np.clip(
        rows["hdl_cholesterol"] + rng.normal(0, 4, n_samples), 25, 90
    ).round().astype(int)

    rows["sbp"] = np.clip(
        rows["sbp"] + rng.normal(0, 5, n_samples), 90, 200
    ).round().astype(int)

    rows["egfr"] = np.clip(
        rows["egfr"] + rng.normal(0, 5, n_samples), 30, 120
    ).round().astype(int)

    rows["bmi"] = np.clip(
        rows["bmi"] + rng.normal(0, 0.7, n_samples), 18.0, 45.0
    )

    probs = model.predict_proba(rows)[:, 1]
    low, med, high = np.percentile(probs, [5, 50, 95])
    return float(low), float(med), float(high)


def get_percentile(prob: float, dist: np.ndarray) -> float:
    """Return percentile (0–1) of prob within distribution dist."""
    return float((dist <= prob).mean())


# ============================================================
# Data loading + synthetic label generation
# ============================================================
@st.cache_data(show_spinner=False)
def load_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset {path} not found. Please place synthetic_cvd_10000.csv "
            f"in the same folder as this app."
        )
    df = pd.read_csv(path)

    # Expect columns:
    # gender, age, total_cholesterol, hdl_cholesterol, sbp,
    # diabetes, smoker, egfr, anti_hypertensive_med, statin_use, bmi

    # If no outcome column, generate a synthetic CVD event label
    if "cvd_event" not in df.columns:
        rng = np.random.default_rng(2025)

        age = df["age"].to_numpy()
        chol = df["total_cholesterol"].to_numpy()
        hdl = df["hdl_cholesterol"].to_numpy()
        sbp = df["sbp"].to_numpy()
        bmi = df["bmi"].to_numpy()
        egfr = df["egfr"].to_numpy()

        diabetes = (df["diabetes"].str.upper() == "YES").astype(int).to_numpy()
        smoker = (df["smoker"].str.upper() == "YES").astype(int).to_numpy()
        anti_htn = (df["anti_hypertensive_med"].str.upper() == "YES").astype(int).to_numpy()
        statin = (df["statin_use"].str.upper() == "YES").astype(int).to_numpy()

        # Simple, hand-crafted risk equation (logistic model)
        logit = (
            -7.5
            + 0.04 * (age - 55)
            + 0.018 * (sbp - 130)
            + 0.010 * (chol - 190)
            - 0.025 * (hdl - 45)
            + 0.30 * diabetes
            + 0.35 * smoker
            + 0.015 * (bmi - 27)
            - 0.015 * (egfr - 90)
            + 0.20 * (1 - anti_htn)   # higher risk if NOT on BP meds
            + 0.20 * (1 - statin)     # higher risk if NOT on statin
        )

        p = 1.0 / (1.0 + np.exp(-logit))
        p = np.clip(p, 0.01, 0.80)
        df["cvd_event"] = rng.binomial(1, p)

    return df


# ============================================================
# Model training (Logistic Regression with preprocessing)
# ============================================================
@st.cache_resource(show_spinner=True)
def train_model(df: pd.DataFrame):
    target = "cvd_event"

    feature_cols = [
        "gender",
        "age",
        "total_cholesterol",
        "hdl_cholesterol",
        "sbp",
        "diabetes",
        "smoker",
        "egfr",
        "anti_hypertensive_med",
        "statin_use",
        "bmi",
    ]

    X = df[feature_cols].copy()
    y = df[target].astype(int).to_numpy()

    cat_cols = [
        "gender",
        "diabetes",
        "smoker",
        "anti_hypertensive_med",
        "statin_use",
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    lr = LogisticRegression(max_iter=1000)
    pipe = Pipeline([("pre", pre), ("lr", lr)])
    pipe.fit(X, y)

    # In-sample performance (synthetic; just for display)
    preds = pipe.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, preds)
    brier = brier_score_loss(y, preds)

    metrics = {"AUC": auc, "Brier": brier}
    return pipe, metrics, preds


# ============================================================
# Load data + train model
# ============================================================
df = load_dataset(DATA_PATH)
MODEL, METRICS, RISK_DIST = train_model(df)
PREVALENCE = df["cvd_event"].mean()

# ============================================================
# Sidebar: inputs
# ============================================================
with st.sidebar:
    st.markdown("### Dataset status")
    st.success(
        f"Rows: **{len(df):,}**  \n"
        f"Prevalence (CVD event): **{PREVALENCE*100:.1f}%**"
    )

    st.markdown("---")
    st.markdown("### Patient Profile")

    gender = st.radio("Sex", ["Male", "Female"], index=0, horizontal=True)

    age = st.slider("Age", 30, 80, 55, step=1)

    total_chol = st.slider("Total cholesterol (mg/dL)", 120, 320, 190, step=1)
    hdl = st.slider("HDL cholesterol (mg/dL)", 25, 90, 45, step=1)

    sbp = st.slider("Systolic BP (mmHg)", 90, 200, 130, step=1)

    diabetes = st.radio("Diabetes", ["No", "Yes"], index=0, horizontal=True)
    smoker = st.radio("Current smoker", ["No", "Yes"], index=0, horizontal=True)

    egfr = st.slider("eGFR (mL/min/1.73 m²)", 30, 120, 90, step=1)

    anti_htn = st.radio(
        "Using anti-hypertensive medication", ["No", "Yes"], index=0, horizontal=True
    )
    statin_use = st.radio(
        "Using statins", ["No", "Yes"], index=0, horizontal=True
    )

    bmi = st.slider("BMI (kg/m²)", 18.0, 45.0, 27.0, step=0.1)

    st.markdown("---")
    show_advanced = st.checkbox(
        "Show advanced metrics (uncertainty, NNT details)",
        value=True,
    )

# Build single-row DataFrame for prediction
input_row = pd.DataFrame(
    [{
        "gender": gender,
        "age": age,
        "total_cholesterol": total_chol,
        "hdl_cholesterol": hdl,
        "sbp": sbp,
        "diabetes": diabetes,
        "smoker": smoker,
        "egfr": egfr,
        "anti_hypertensive_med": anti_htn,
        "statin_use": statin_use,
        "bmi": bmi,
    }]
)

# ============================================================
# Main layout
# ============================================================
st.title("CVD 10-Year Risk – Predictor v3 (Synthetic 10K + NNT)")
st.caption(
    "Educational demo using a 10,000-record synthetic CVD risk-factor dataset. "
    "This is **not** a medical device and must not be used for real patient care."
)

left, right = st.columns([1.1, 1.9])

with st.spinner("Computing risk and uncertainty..."):
    prob = float(MODEL.predict_proba(input_row)[:, 1])
    low_u, med_u, high_u = local_jitter_uncertainty(MODEL, input_row, n_samples=200)
    percentile = get_percentile(prob, RISK_DIST)

risk_label, risk_color = get_risk_category(prob)

# ------------------------------------------------------------
# LEFT: summary & model metrics
# ------------------------------------------------------------
with left:
    st.subheader("Input Summary")

    st.markdown(
        f"- **Sex:** {gender}  \n"
        f"- **Age:** {age} years  \n"
        f"- **Total cholesterol:** {total_chol} mg/dL  \n"
        f"- **HDL cholesterol:** {hdl} mg/dL  \n"
        f"- **Systolic BP:** {sbp} mmHg  \n"
        f"- **Diabetes:** {diabetes}  \n"
        f"- **Current smoker:** {smoker}  \n"
        f"- **eGFR:** {egfr} mL/min/1.73 m²  \n"
        f"- **Anti-hypertensive medication:** {anti_htn}  \n"
        f"- **Statin use:** {statin_use}  \n"
        f"- **BMI:** {bmi:.1f} kg/m²"
    )

    st.markdown("---")
    st.markdown("### Model Performance (synthetic)")

    st.markdown(
        f"- **Logistic Regression (internal):**  \n"
        f"  - ROC AUC: **{METRICS['AUC']:.3f}**  \n"
        f"  - Brier score: **{METRICS['Brier']:.3f}**  \n"
        f"- **Training prevalence:** {format_percent(PREVALENCE)}"
    )

    st.markdown("---")
    st.markdown("### Risk Context in Cohort")
    st.markdown(
        f"- **Predicted 10-year risk:** {format_percent(prob)}  \n"
        f"- **Percentile in synthetic cohort:** ~**{percentile*100:.0f}th** percentile."
    )

# ------------------------------------------------------------
# RIGHT: risk, uncertainty, treatment impact & NNT
# ------------------------------------------------------------
with right:
    st.subheader("10-Year CVD Risk")

    st.markdown(
        f"<h2 style='color:{risk_color}; margin-bottom:0;'>{format_percent(prob)}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**{risk_label}**")

    st.caption(
        "Risk is estimated from a logistic regression model trained on a 10,000-record "
        "synthetic dataset with age, blood pressure, cholesterol, kidney function, diabetes, "
        "smoking, BMI, and medication use."
    )

    if show_advanced:
        st.markdown("#### Local Uncertainty (Jittered Inputs)")
        st.markdown(
            f"- **5–95% band:** {format_percent(low_u)} – {format_percent(high_u)}  \n"
            f"- **Median under jitter:** {format_percent(med_u)}"
        )
        st.caption(
            "We slightly perturb numeric values (age, BP, lipids, BMI, eGFR) 200 times and "
            "recompute risk. The band shows how sensitive the prediction is to small changes "
            "around the current profile."
        )

    # --------------------------------------------------------
    # Treatment impact & NNT
    # --------------------------------------------------------
    st.markdown("### Estimated Treatment Impact & NNT")

    st.markdown(
        f"These estimates assume a **{RISK_HORIZON_YEARS}-year CVD risk horizon** and "
        "average relative risk reductions from large clinical trials. "
        "They are for **illustration and shared decision-making training only**."
    )

    treatments = []

    # Statin therapy (assumes initiation/intensification of high-intensity statin)
    treatments.append(
        {
            "name": "High-intensity statin",
            "rrr": RRR_STATIN,
            "description": "Assumes starting or intensifying statin therapy for lipid control.",
        }
    )

    # Blood pressure therapy
    treatments.append(
        {
            "name": "Optimized blood pressure therapy",
            "rrr": RRR_BP,
            "description": "Assumes achieving better BP control with guideline-directed therapy.",
        }
    )

    rows_nnt = []
    for t in treatments:
        treated_risk, arr, nnt = treatment_impact(prob, t["rrr"], RISK_HORIZON_YEARS)

        if nnt is None:
            nnt_str = "Not meaningful at this risk level"
        elif nnt > 2000:
            nnt_str = "> 2000"
        else:
            nnt_str = f"{nnt:.0f}"

        rows_nnt.append(
            {
                "Therapy": t["name"],
                "Baseline risk": format_percent(prob),
                f"Risk with therapy\n(approx.)": format_percent(treated_risk),
                "Absolute risk reduction\n(percentage points)": f"{(arr*100):.1f}",
                f"NNT over {RISK_HORIZON_YEARS} yrs\n(patients)": nnt_str,
            }
        )

    st.dataframe(
        pd.DataFrame(rows_nnt),
        use_container_width=True,
        hide_index=True,
    )

    if show_advanced:
        st.caption(
            "NNT (Number Needed to Treat) is 1 / ARR, where ARR is the absolute risk reduction "
            "between baseline and treated risk. For example, NNT=20 means that if 20 patients "
            "with a profile like this are treated for the specified duration, we expect to "
            "prevent about 1 additional CVD event on average."
        )

# ============================================================
# About / methodology
# ============================================================
with st.expander("About this app & methodology"):
    st.markdown(
        f"""
        ### Overview

        - This is **CVD Predictor v3**, built on a **10,000-patient synthetic dataset** that follows
          clinically plausible distributions for age, blood pressure, cholesterol, kidney function,
          diabetes, smoking, BMI, and medication use.
        - A **logistic regression model** is trained inside the app (cached) to estimate
          **{RISK_HORIZON_YEARS}-year CVD event risk**.
        - The dataset includes an internally generated **`cvd_event`** label, constructed via a
          hand-crafted risk equation using these risk factors.

        ### How the synthetic outcome is generated

        We compute a log-odds score using:
        - Age, SBP, total cholesterol, HDL, BMI, eGFR  
        - Diabetes, current smoking, and whether anti-hypertensive or statin therapy is used  

        The score is converted to a probability via a logistic function and used to draw
        a Bernoulli outcome. This yields a prevalence around {format_percent(PREVALENCE)}.

        ### Treatment impact & NNT

        - Baseline model risk is adjusted using literature-inspired **relative risk reductions (RRR)**:
          - High-intensity statin: ~{int(RRR_STATIN*100)}% RRR  
          - Optimized BP therapy: ~{int(RRR_BP*100)}% RRR  
        - Treated risk = baseline risk × (1 − RRR).  
        - **Absolute risk reduction (ARR)** = baseline − treated risk.  
        - **Number Needed to Treat (NNT)** = 1 / ARR.

        These are **simplified educational estimates** and not patient-specific treatment effects.

        ### Important disclaimer

        - This tool is built entirely on **synthetic data**.  
        - It is intended for **education, modelling experiments, and stakeholder demos only**.  
        - It must **not** be used for real clinical decision-making or patient management.
        """
    )

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "© 2025 Howard Nguyen, PhD – Synthetic CVD Risk & NNT Demo. "
    "Research & Demo use only."
    "</p>",
    unsafe_allow_html=True,
)
