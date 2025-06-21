import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi halaman
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# Header
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 2.5em;
            color: #1f6feb;
            margin-bottom: 0;
        }
        .sub-title {
            text-align: center;
            font-size: 1.2em;
            color: #666;
            margin-top: 0;
            margin-bottom: 2em;
        }
        .metric-container {
            display: flex;
            justify-content: space-around;
            margin-top: 2em;
        }
        .influencer {
            font-size: 1.1em;
            margin: 0.5em 0;
        }
    </style>
    <h1 class='main-title'>Prediksi Performa Akademik Mahasiswa</h1>
    <h4 class='sub-title'>Berdasarkan Gaya Hidup dan Dukungan Sosial Menggunakan Random Forest</h4>
""", unsafe_allow_html=True)

st.header("🎓 Input Your Data")

# Input Slider
with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        previous_gpa = st.slider("Previous GPA (0-4)", 0.0, 4.0, 3.6, 0.1)
        attendance = st.slider("Attendance Percentage (%)", 0, 100, 95, 1)
        study_hours = st.slider("Study Hours Per Day", 0, 12, 4, 1)

    with col2:
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5, 1)
        screen_time = st.slider("Screen Time (hours/day)", 0, 24, 12, 1)
        time_management = st.slider("Time Management Score (1-10)", 1, 10, 6, 1)

    submit = st.form_submit_button("🔍 Predict Performance")
    reset = st.form_submit_button("🔄 Reset")

if submit:
    try:
        # Load model
        model = joblib.load("random_forest_model.pkl")

        # Buat DataFrame dari input
        input_data = pd.DataFrame([[
            previous_gpa,
            attendance,
            study_hours,
            stress_level,
            screen_time,
            time_management
        ]], columns=[
            'previous_gpa',
            'attendance_percentage',
            'study_hours_per_day',
            'stress_level',
            'screen_time',
            'time_management_score'
        ])

        # Prediksi
        prediction = model.predict(input_data)[0]
        performance_level = "Good" if prediction >= 75 else "Average" if prediction >= 60 else "Poor"
        improvement = round((100 - prediction))

        # Output Hasil
        st.markdown("## 📊 Prediction Results")
        st.progress(min(prediction / 100, 1.0))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Exam Score", f"{prediction:.1f}")
            st.success(f"Performance Level: {performance_level}")
        with col2:
            st.metric("Improvement Potential", f"+{improvement:.0f}%")
            st.info("Prediction based on lifestyle factors")

        st.markdown("## 🔍 Key Influencers")
        st.markdown("""
            <ul style="list-style-type: none; padding-left: 1em;">
                <li class='influencer'>📘 <b>Previous GPA:</b> High</li>
                <li class='influencer'>⏱️ <b>Study Hours:</b> High</li>
                <li class='influencer'>📋 <b>Time Management:</b> Medium</li>
            </ul>
        """, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("Model file 'random_forest_model.pkl' tidak ditemukan. Silakan latih dan simpan model terlebih dahulu.")

elif reset:
    st.rerun()
