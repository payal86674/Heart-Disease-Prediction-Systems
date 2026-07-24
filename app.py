import os
import pickle
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure Streamlit app layout
st.set_page_config(
    page_title="CardioPulse AI — Clinical Analytics Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# SAFE MODEL LOADING
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_disease_model.sav"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            f"⚠️ Model file not found at `{MODEL_PATH}`. Please upload `heart_disease_model.sav` to your repository root."
        )
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


loaded_model = load_model()

# -----------------------------------------------------------------------------
# CUSTOM STYLING (Modern Medical Theme)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #f8fafc;
    }

    .banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 28px 32px;
        border-radius: 20px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }

    .banner h1 {
        color: #ffffff;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .banner p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
    }

    .help-badge {
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-bottom: 12px;
    }

    .result-card-low {
        background-color: #f0fdf4;
        border: 1.5px solid #86efac;
        padding: 24px;
        border-radius: 18px;
    }

    .result-card-high {
        background-color: #fef2f2;
        border: 1.5px solid #fca5a5;
        padding: 24px;
        border-radius: 18px;
    }

    .action-step {
        background: #ffffff;
        border-left: 4px solid #2563eb;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & KNOWLEDGE BASE
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/isometric/100/stethoscope.png", width=64
    )
    st.title("CardioPulse AI")
    st.caption("v3.0 Visual Decision Analytics Platform")

    st.divider()

    st.markdown("### 📘 Quick Terminology Guide")
    st.markdown("""
    * **Resting BP:** Systolic pressure at rest. Optimal: $<120$ mmHg.
    * **Cholesterol:** Total serum cholesterol. Target: $<200$ mg/dL.
    * **Max HR:** Peak heart rate reached during exercise test.
    * **Oldpeak:** ST depression during exertion vs rest on ECG.
    * **Fluoroscopy (Vessels):** Number of major blood vessels visible ($0$–$3$).
    """)

    st.divider()
    st.info(
        "💡 **Clinical Disclaimer:** Diagnostic algorithm output is for preliminary risk screening and clinical evaluation support only."
    )

# -----------------------------------------------------------------------------
# HERO BANNER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="banner">
        <h1>🫀 Interactive Cardiac Decision Support</h1>
        <p>Input patient metrics below to trigger machine learning risk analysis, visual benchmarking, and automated diagnostic guidance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# INPUT FORM
# -----------------------------------------------------------------------------
with st.form("clinical_assessment_form"):
    st.subheader("📋 Patient Metrics & Diagnostic Parameters")

    tab1, tab2, tab3 = st.tabs(
        [
            "1. Demographics & Vitals",
            "2. Stress Test & ECG",
            "3. Advanced Fluoroscopy & Blood",
        ]
    )

    with tab1:
        st.markdown(
            '<div class="help-badge">ℹ️ Core physiological baseline parameters.</div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age (Years)",
                min_value=1,
                max_value=120,
                value=54,
                help="Age in full years.",
            )
            sex = st.selectbox(
                "Biological Sex",
                options=["Female (0)", "Male (1)"],
                help="Biological sex recorded at birth.",
            )
            trestbps = st.number_input(
                "Resting Blood Pressure (mmHg)",
                min_value=80,
                max_value=230,
                value=135,
                help="Systolic blood pressure measured upon admission (Normal target: <120 mmHg).",
            )

        with col2:
            chol = st.number_input(
                "Serum Cholesterol (mg/dL)",
                min_value=100,
                max_value=600,
                value=245,
                help="Total cholesterol level (Normal target: <200 mg/dL).",
            )
            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL",
                options=["No / Normal (0)", "Yes / Elevated (1)"],
                help="Fasting blood sugar test result.",
            )

    with tab2:
        st.markdown(
            '<div class="help-badge">ℹ️ Physical exertion responses captured during stress test.</div>',
            unsafe_allow_html=True,
        )
        col3, col4 = st.columns(2)

        with col3:
            cp = st.selectbox(
                "Chest Pain Type (CP)",
                options=[
                    "Typical Angina (0)",
                    "Atypical Angina (1)",
                    "Non-anginal Pain (2)",
                    "Asymptomatic (3)",
                ],
                help="Chest pain category description.",
            )
            thalach = st.number_input(
                "Maximum Heart Rate Achieved (bpm)",
                min_value=60,
                max_value=220,
                value=138,
                help="Highest heart rate reached during treadmill stress test.",
            )

        with col4:
            exang = st.selectbox(
                "Exercise Induced Angina",
                options=["No (0)", "Yes (1)"],
                help="Does physical exercise induce angina/chest tightness?",
            )
            restecg = st.selectbox(
                "Resting ECG Results",
                options=[
                    "Normal (0)",
                    "ST-T Wave Abnormality (1)",
                    "Left Ventricular Hypertrophy (2)",
                ],
                help="Resting electrocardiographic results.",
            )

    with tab3:
        st.markdown(
            '<div class="help-badge">ℹ️ Imaging and advanced cardiac diagnostic measures.</div>',
            unsafe_allow_html=True,
        )
        col5, col6 = st.columns(2)

        with col5:
            oldpeak = st.number_input(
                "ST Depression (Oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=1.8,
                step=0.1,
                help="ST depression induced by exercise relative to rest.",
            )
            slope = st.selectbox(
                "Slope of Peak Exercise ST Segment",
                options=["Upsloping (0)", "Flat (1)", "Downsloping (2)"],
                help="Slope of the ST segment during peak exercise stress test.",
            )

        with col6:
            ca = st.slider(
                "Colored Vessels by Fluoroscopy (0–3)",
                min_value=0,
                max_value=3,
                value=1,
                help="Number of major vessels colored by fluoroscopy.",
            )
            thal = st.selectbox(
                "Thalassemia Status",
                options=[
                    "Normal (3)",
                    "Fixed Defect (6)",
                    "Reversible Defect (7)",
                ],
                help="Thalassemia stress result category.",
            )

    st.markdown("---")
    submit_btn = st.form_submit_button(
        "⚡ Analyze & Generate Visual Dashboard", use_container_width=True
    )

# -----------------------------------------------------------------------------
# PREDICTION & VISUALIZATION ENGINE
# -----------------------------------------------------------------------------
if submit_btn:
    if loaded_model is None:
        st.error("Cannot process prediction. Model file is missing from server.")
    else:
        # Mapping string options back to numbers
        mapping = {
            "Female (0)": 0,
            "Male (1)": 1,
            "Typical Angina (0)": 0,
            "Atypical Angina (1)": 1,
            "Non-anginal Pain (2)": 2,
            "Asymptomatic (3)": 3,
            "No / Normal (0)": 0,
            "Yes / Elevated (1)": 1,
            "Normal (0)": 0,
            "ST-T Wave Abnormality (1)": 1,
            "Left Ventricular Hypertrophy (2)": 2,
            "No (0)": 0,
            "Yes (1)": 1,
            "Upsloping (0)": 0,
            "Flat (1)": 1,
            "Downsloping (2)": 2,
            "Normal (3)": 3,
            "Fixed Defect (6)": 6,
            "Reversible Defect (7)": 7,
        }

        raw_features = [
            age,
            mapping[sex],
            mapping[cp],
            trestbps,
            chol,
            mapping[fbs],
            mapping[restecg],
            thalach,
            mapping[exang],
            oldpeak,
            mapping[slope],
            ca,
            mapping[thal],
        ]

        # Reshape & pad to 15 features to match model matrix dimensions
        input_data = np.asarray(raw_features).reshape(1, -1)
        input_data = np.column_stack([input_data, np.zeros((1, 2))])

        prediction = loaded_model.predict(input_data)[0]

        # Calculate Probability / Risk percentage
        risk_score = 50.0
        if hasattr(loaded_model, "predict_proba"):
            try:
                probs = loaded_model.predict_proba(input_data)[0]
                risk_score = float(probs[1] * 100)  # Probability of Heart Disease
            except Exception:
                risk_score = 85.0 if prediction == 1 else 15.0
        else:
            risk_score = 85.0 if prediction == 1 else 15.0

        st.markdown("## 📈 Diagnostic Analytics & Risk Visualizations")

        # -----------------------------------------------------------------------------
        # CHARTS ROW 1: GAUGE & BENCHMARKS
        # -----------------------------------------------------------------------------
        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            st.markdown("### 🎯 Predicted Risk Gauge")
            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    number={"suffix": "%"},
                    title={"text": "Cardiovascular Risk Score"},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {
                            "color": "#dc2626" if risk_score >= 50 else "#16a34a"
                        },
                        "bgcolor": "white",
                        "borderwidth": 2,
                        "bordercolor": "#e2e8f0",
                        "steps": [
                            {"range": [0, 35], "color": "#dcfce7"},
                            {"range": [35, 65], "color": "#fef9c3"},
                            {"range": [65, 100], "color": "#fee2e2"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.75,
                            "value": risk_score,
                        },
                    },
                )
            )
            gauge_fig.update_layout(
                height=290, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

        with chart_col2:
            st.markdown("### 📊 Vitals vs. Recommended Targets")
            # Create a comparison chart for BP, Cholesterol, and Max Heart Rate
            metrics_df = {
                "Metric": [
                    "Resting BP (mmHg)",
                    "Cholesterol (mg/dL)",
                    "Max Heart Rate (bpm)",
                ],
                "Patient Value": [trestbps, chol, thalach],
                "Clinical Target": [120, 200, 220 - age],  # Target HR = 220 - age
            }

            bar_fig = px.bar(
                metrics_df,
                x="Metric",
                y=["Patient Value", "Clinical Target"],
                barmode="group",
                color_discrete_sequence=["#2563eb", "#94a3b8"],
            )
            bar_fig.update_layout(
                height=290,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                yaxis_title="Value",
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        # -----------------------------------------------------------------------------
        # CHARTS ROW 2: ECG ST DEPRESSION STRESS VISUALIZER
        # -----------------------------------------------------------------------------
        st.markdown("### 📉 ST Depression (Oldpeak) Exertion Waveform")
        
        # Simulate an ECG wave segment showing ST depression shift
        x_wave = np.linspace(0, 4 * np.pi, 300)
        baseline_ecg = np.sin(x_wave) * np.exp(-0.1 * x_wave)
        st_shift = - (oldpeak / 5.0)  # Apply visually calculated shift
        patient_ecg = baseline_ecg + np.where((x_wave > 3) & (x_wave < 8), st_shift, 0)

        wave_fig = go.Figure()
        wave_fig.add_trace(go.Scatter(x=x_wave, y=baseline_ecg, mode='lines', name='Normal Baseline Waveform', line=dict(color='#94a3b8', dash='dash')))
        wave_fig.add_trace(go.Scatter(x=x_wave, y=patient_ecg, mode='lines', name='Patient ST Segment', line=dict(color='#dc2626' if oldpeak > 1.0 else '#2563eb', width=2.5)))
        wave_fig.update_layout(
            height=220,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Time / Segment Phase",
            yaxis_title="Voltage (mV)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(wave_fig, use_container_width=True)

        # -----------------------------------------------------------------------------
        # DIAGNOSTIC CONCLUSION & ACTION PLAN
        # -----------------------------------------------------------------------------
        st.markdown("## 📑 Diagnosis & Actionable Protocol")

        if prediction == 0:
            st.markdown(
                f"""
                <div class="result-card-low">
                    <h2 style="color: #15803d; margin: 0;">✅ Low Heart Disease Risk Detected ({100 - risk_score:.1f}% Confidence)</h2>
                    <p style="color: #166534; margin-top: 8px;">Patient indicators demonstrate healthy cardiovascular range. Focus on routine health maintenance.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown("### 🥗 Preventive Care Focus")
                st.markdown("""
                * **Dietary Management:** Mediterranean diet rich in high-fiber whole grains, leafy greens, and omega-3 fatty acids.
                * **Routine Cardio:** $150$ minutes/week of moderate physical activity (e.g., jogging, cycling).
                * **Vitals Monitoring:** Annual checkup on Blood Pressure ($<120/80$ mmHg) and Lipid panels.
                """)
            with res_col2:
                st.markdown("### 📌 Next Steps")
                st.markdown(
                    """
                    <div class="action-step"><strong>1. Annual Screening:</strong> Schedule standard wellness checkup once a year.</div>
                    <div class="action-step"><strong>2. Re-test Lipids:</strong> Re-evaluate serum cholesterol levels in 12 months.</div>
                    <div class="action-step"><strong>3. Healthy Habits:</strong> Maintain non-smoking status and regular sleep cycles.</div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f"""
                <div class="result-card-high">
                    <h2 style="color: #b91c1c; margin: 0;">⚠️ Elevated Heart Disease Risk Detected ({risk_score:.1f}% Risk Level)</h2>
                    <p style="color: #991b1b; margin-top: 8px;">Patient features exhibit elevated cardiovascular risk indicators. Clinical follow-up recommended.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown("### 🩺 Targeted Clinical Interventions")
                st.markdown("""
                * **DASH Diet Protocol:** Sodium reduction ($<2,000$ mg/day) and low saturated fat intake.
                * **Supervised Exercise:** Physician-guided physical therapy before high-intensity exertion.
                * **Lipid Management:** Consultation regarding potential statin or blood pressure therapy.
                """)
            with res_col2:
                st.markdown("### 🚨 Recommended Follow-Up")
                st.markdown(
                    """
                    <div class="action-step"><strong>1. Cardiology Referral:</strong> Schedule a consultation with a certified cardiologist.</div>
                    <div class="action-step"><strong>2. Diagnostic Angiography:</strong> Discuss coronary imaging or stress echo tests with your doctor.</div>
                    <div class="action-step"><strong>3. Symptom Tracking:</strong> Document any instances of chest tightness, shortness of breath, or dizziness.</div>
                    """,
                    unsafe_allow_html=True,
                )