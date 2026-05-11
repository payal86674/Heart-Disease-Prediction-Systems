import streamlit as st
import pickle
import numpy as np

# Configure the Streamlit app
st.set_page_config(
    page_title='AI Heart Disease Dashboard',
    page_icon='❤️‍🩹',
    layout='wide'
)

# Load the trained model
loaded_model = pickle.load(open('heart_disease_model.sav', 'rb'))

# Custom CSS styling for a polished healthcare dashboard
st.markdown(
    """
    <style>
    :root {
        --primary: #0f4c81;
        --accent: #46a5ff;
        --muted: #5f6f86;
        --background: #eef6ff;
        --card: rgba(255,255,255,0.95);
        --shadow: 0 28px 80px rgba(15, 76, 129, 0.12);
    }

    body {
        background: linear-gradient(180deg, #e7f2ff 0%, #ffffff 100%);
    }

    .main > div.block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    .dashboard-card,
    .stSidebar .css-1lcbmhc {
        background: var(--card) !important;
        border-radius: 24px;
        box-shadow: var(--shadow) !important;
        border: 1px solid rgba(15, 76, 129, 0.08) !important;
    }

    .metric-card {
        background: linear-gradient(135deg, #125a9d 0%, #46a5ff 100%);
        color: white;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 20px 40px rgba(15, 76, 129, 0.18);
        min-height: 135px;
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
    }

    .metric-card::before {
        content: '';
        position: absolute;
        width: 140px;
        height: 140px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.16);
        right: -40px;
        top: -40px;
    }

    .metric-card h3 { 
        margin-bottom: 10px;
        font-size: 1.5rem;
        letter-spacing: -0.03em;
    }

    .metric-card p {
        margin: 0;
        color: rgba(255, 255, 255, 0.88);
        line-height: 1.7;
    }

    .section-header {
        display: flex;
        justify-content: space-between;
        gap: 18px;
        flex-wrap: wrap;
    }

    .title-block h1 {
        margin: 0;
        font-size: clamp(2.4rem, 3vw, 3.6rem);
        color: #0f2646;
        line-height: 1.02;
    }

    .title-block p {
        margin: 14px 0 0;
        color: var(--muted);
        font-size: 1rem;
        max-width: 720px;
    }

    .section-meta {
        min-width: 260px;
    }

    .stButton button {
        border-radius: 16px;
        padding: 14px 24px;
        background: linear-gradient(135deg, #0f4c81, #46a5ff);
        color: white;
        font-weight: 700;
        border: none;
        box-shadow: 0 16px 32px rgba(15, 76, 129, 0.18);
    }

    .stButton button:hover {
        transform: translateY(-1px);
    }

    .result-box {
        border-radius: 24px;
        padding: 26px;
        border: 1px solid rgba(15, 76, 129, 0.12);
        background: rgba(255, 255, 255, 0.98);
        box-shadow: var(--shadow);
        margin-top: 24px;
    }

    .result-success {
        background: linear-gradient(135deg, #e4f7eb 0%, #f5fbf6 100%);
        border-color: #8cd4a4;
    }

    .result-error {
        background: linear-gradient(135deg, #fce7e7 0%, #fff5f5 100%);
        border-color: #f2a9a9;
    }

    .footer {
        text-align: center;
        color: var(--muted);
        padding: 24px 0 12px;
        font-size: 0.95rem;
    }

    .footer a {
        color: #0f4c81;
        text-decoration: none;
        font-weight: 600;
    }

    @media (max-width: 900px) {
        .section-header {
            flex-direction: column;
        }
    }

    @media (max-width: 680px) {
        .main > div.block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar navigation and info
with st.sidebar:
    st.markdown('<div class="dashboard-card" style="padding: 24px">', unsafe_allow_html=True)
    st.markdown('## 🏥 Heart Disease AI Dashboard')
    st.markdown(
        'A modern medical dashboard for heart disease risk prediction with streamlined clinical inputs and clear output guidance.'
    )
    st.write('---')
    st.markdown('### 📌 Project Overview')
    st.markdown(
        '- Retains current prediction functionality\n'
        '- Uses `heart_disease_model.sav`\n'
        '- Responsive desktop/mobile layout\n'
        '- Friendly healthcare design'
    )
    st.write('---')
    st.markdown('### 🧾 Inputs Included')
    st.markdown(
        '- Age\n'
        '- Sex\n'
        '- Chest Pain Type\n'
        '- Resting Blood Pressure\n'
        '- Cholesterol\n'
        '- Fasting Blood Sugar\n'
        '- Rest ECG\n'
        '- Max Heart Rate\n'
        '- Exercise Angina\n'
        '- Oldpeak\n'
        '- Slope\n'
        '- Vessels\n'
        '- Thal'
    )
    st.write('---')
    st.markdown('### 👨‍💻 Developer')
    st.markdown('Built with Streamlit for a modern AI healthcare product experience.')
    st.markdown('</div>', unsafe_allow_html=True)

# Header and hero section
st.markdown('<div class="dashboard-card section-header" style="padding: 28px 32px">', unsafe_allow_html=True)
st.markdown(
    '<div class="title-block"><h1>Heart Disease Risk Prediction</h1>'
    '<p>Experience a professional medical dashboard presentation powered by your machine learning model. Enter patient values, analyze risk, and review confidence results.</p></div>'
    '<div class="section-meta"><div class="metric-card"><h3>Clinical AI</h3><p>Reliable, clean, and intuitive risk assessment for heart health screening.</p></div></div>'
    ,
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# Top metric cards
metric_cols = st.columns(4, gap='large')
metrics = [
    ('⚡ Fast AI', 'Quick predictions with a lightweight pipeline.'),
    ('🧠 Smart Model', 'Probabilistic output when available.'),
    ('📱 Responsive', 'Designed for desktop and mobile users.'),
    ('🩺 Healthcare', 'Blue-white medical theme with strong hierarchy.'),
]
for col, (title, description) in zip(metric_cols, metrics):
    col.markdown('<div class="metric-card">', unsafe_allow_html=True)
    col.markdown(f'<h3>{title}</h3><p>{description}</p>', unsafe_allow_html=True)
    col.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="dashboard-card" style="padding: 28px 24px 24px">', unsafe_allow_html=True)

# Input form section
with st.form(key='heart_disease_form'):
    st.markdown('### 🩺 Patient Health Inputs')
    col1, col2, col3 = st.columns(3, gap='large')

    with col1:
        age = st.number_input(
            '🧬 Age',
            min_value=1,
            max_value=120,
            value=45,
            help='Patient age in years.',
        )
        sex = st.selectbox(
            '👥 Sex',
            options=['Female (0)', 'Male (1)'],
            help='0 = Female, 1 = Male',
        )
        cp = st.selectbox(
            '💓 Chest Pain Type',
            options=[
                'Typical angina (0)',
                'Atypical angina (1)',
                'Non-anginal pain (2)',
                'Asymptomatic (3)',
            ],
            help='Choose the chest pain category that best matches symptoms.',
        )
        trestbps = st.number_input(
            '🩺 Resting Blood Pressure',
            min_value=80,
            max_value=220,
            value=120,
            help='Resting blood pressure in mmHg.',
        )
        chol = st.number_input(
            '🧪 Cholesterol',
            min_value=100,
            max_value=600,
            value=200,
            help='Serum cholesterol in mg/dL.',
        )

    with col2:
        fbs = st.selectbox(
            '🩸 Fasting Blood Sugar',
            options=['Normal (0)', 'High (1)'],
            help='0 = ≤ 120 mg/dL, 1 = > 120 mg/dL.',
        )
        restecg = st.selectbox(
            '📈 Rest ECG',
            options=[
                'Normal (0)',
                'ST-T wave abnormality (1)',
                'Left ventricular hypertrophy (2)',
            ],
            help='Electrocardiogram result category.',
        )
        thalach = st.number_input(
            '❤️ Maximum Heart Rate',
            min_value=60,
            max_value=250,
            value=150,
            help='Maximum heart rate achieved during exercise.',
        )
        exang = st.selectbox(
            '🏃 Exercise Induced Angina',
            options=['No (0)', 'Yes (1)'],
            help='0 = No, 1 = Yes',
        )
        oldpeak = st.number_input(
            '📉 Oldpeak',
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help='ST depression induced by exercise relative to rest.',
        )

    with col3:
        slope = st.selectbox(
            '📊 Slope',
            options=['Upsloping (0)', 'Flat (1)', 'Downsloping (2)'],
            help='Slope of the peak exercise ST segment.',
        )
        ca = st.slider(
            '🩻 Number of Major Vessels',
            min_value=0,
            max_value=3,
            value=0,
            help='Number of vessels colored by fluoroscopy (0-3).',
        )
        thal = st.selectbox(
            '🧬 Thalassemia Type',
            options=['Normal (3)', 'Fixed defect (6)', 'Reversible defect (7)'],
            help='Thalassemia result category for model input.',
        )
        st.markdown(
            '<div style="color: var(--muted); margin-top: 10px; font-size: 0.95rem;">Use medically realistic values for the best possible prediction quality. Categories are mapped to your model labels automatically.</div>',
            unsafe_allow_html=True,
        )

    submitted = st.form_submit_button('Run Prediction')

    if submitted:
        mapped_values = {
            'Female (0)': 0,
            'Male (1)': 1,
            'Typical angina (0)': 0,
            'Atypical angina (1)': 1,
            'Non-anginal pain (2)': 2,
            'Asymptomatic (3)': 3,
            'Normal (0)': 0,
            'High (1)': 1,
            'ST-T wave abnormality (1)': 1,
            'Left ventricular hypertrophy (2)': 2,
            'No (0)': 0,
            'Yes (1)': 1,
            'Upsloping (0)': 0,
            'Flat (1)': 1,
            'Downsloping (2)': 2,
            'Normal (3)': 3,
            'Fixed defect (6)': 6,
            'Reversible defect (7)': 7,
        }

        input_data = np.asarray([
            age,
            mapped_values[sex],
            mapped_values[cp],
            trestbps,
            chol,
            mapped_values[fbs],
            mapped_values[restecg],
            thalach,
            mapped_values[exang],
            oldpeak,
            mapped_values[slope],
            ca,
            mapped_values[thal],
        ]).reshape(1, -1)

        # Pad with 2 zero features to match the model's expected 15 features
        input_data = np.column_stack([input_data, np.zeros((1, 2))])

        with st.spinner('Analyzing data and evaluating risk...'):
            prediction = loaded_model.predict(input_data)
            confidence = 'Confidence unavailable for this model.'
            if hasattr(loaded_model, 'predict_proba'):
                try:
                    probabilities = loaded_model.predict_proba(input_data)
                    confidence_value = float(np.max(probabilities) * 100)
                    confidence = f'{confidence_value:.1f}% confidence'
                except Exception:
                    confidence = 'Confidence unavailable for this model.'

        result_class = 'result-success' if prediction[0] == 0 else 'result-error'
        result_title = 'No Heart Disease Detected' if prediction[0] == 0 else 'Elevated Heart Disease Risk'
        result_message = (
            'The model suggests this profile is currently low risk for heart disease. Continue regular care and healthy habits.'
            if prediction[0] == 0
            else 'The model suggests a higher risk profile. Please share this with a physician for follow-up evaluation.'
        )

        st.markdown(f'<div class="result-box {result_class}">', unsafe_allow_html=True)
        st.markdown(f'<h2>{result_title}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: var(--muted); margin-bottom: 18px;">{result_message}</p>', unsafe_allow_html=True)
        st.markdown(
            '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;">'
            f'<div style="background: rgba(255,255,255,0.95); border-radius: 18px; padding: 18px;">'
            f'<strong>Prediction</strong><p style="margin:10px 0 0; font-size:1.3rem;">{int(prediction[0])}</p></div>'
            f'<div style="background: rgba(255,255,255,0.95); border-radius: 18px; padding: 18px;">'
            f'<strong>Confidence</strong><p style="margin:10px 0 0; font-size:1.3rem;">{confidence}</p></div>'
            f'<div style="background: rgba(255,255,255,0.95); border-radius: 18px; padding: 18px;">'
            f'<strong>Model</strong><p style="margin:10px 0 0; font-size:1.3rem;">Pickle ML</p></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer section
st.markdown(
    '<div class="footer">Built as a professional AI healthcare dashboard. Developed with Streamlit and your existing heart disease model. <a href="https://streamlit.io" target="_blank">Learn more</a></div>',
    unsafe_allow_html=True,
)
