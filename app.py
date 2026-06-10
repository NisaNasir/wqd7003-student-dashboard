import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Student Academic Risk Dashboard", layout="wide", page_icon="🎓")
st.title("🎓 Predictive Modeling of Students Academic Performance")
st.markdown("### Universiti Malaya | WQD7003 Data Analytics Deployment (Group 10)")
st.markdown("---")

# 2. Load Cached Machine Learning Assets
@st.cache_resource
def load_production_assets():
    model = joblib.load('final_linear_model.joblib')
    scaler = joblib.load('final_scaler.joblib')
    return model, scaler

try:
    live_lr, live_scaler = load_production_assets()
    st.sidebar.success("✅ Analytics Engine Connected")
except Exception as e:
    st.sidebar.error(f"Critical Error: Asset loading failed: {e}")
    st.stop()

# 3. Sidebar Panel Profile
st.sidebar.markdown("## Deployment Profile")
st.sidebar.info("This dashboard deploys our final selected Linear Regression (Baseline) baseline model to forecast student performance tracks.")
st.sidebar.markdown("**Model Performance Metrics:**")
st.sidebar.markdown("- **R² Variance Explained:** 76.92%")
st.sidebar.markdown("- **Mean Absolute Error (MAE):** 0.4620")
st.sidebar.markdown("- **Root Mean Squared Error (RMSE):** 1.8061")

# 4. Interface Tabs 
tab1, tab2 = st.tabs(["Strategic System Insights", "Individual Student Risk Predictor"])

with tab1:
    st.subheader("System Executive Insights Summary")
    st.markdown("### Key Operational Findings")
    st.markdown(
        """
        Our underlying statistical framework automatically reveals that final student performance is heavily driven by 
        **behavioral metrics and active engagement levels** rather than demographic background or fixed socioeconomic characteristics:
        1. **Attendance:** Identified as the absolute primary driver within the underlying dataset. Consistent exposure to the classroom curriculum remains vital.
        2. **Hours Studied:** Represents the secondary behavioral lever, proving independent home revision strongly updates score profiles.
        3. **Previous Scores:** Highlights historical academic continuity.
        
        *Operational Strategy:* By selecting Linear Regression (Baseline), educational institutions gain complete parameter transparency—allowing advisors to mathematically pinpoint exactly how shifting an attendance track will impact final performance grades.
        """
    )

with tab2:
    st.subheader("Interactive Student Metric Simulator")
    st.write("Modify student behavioral attributes below to simulate an academic profile and calculate estimated exam score values in real time.")
    
    with st.form("student_simulator_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🏃‍♂️ Student Behavioral Indicators")
            attendance = st.slider("Class Attendance Rate (%)", min_value=0, max_value=100, value=85, step=1)
            hours_studied = st.slider("Weekly Independent Study Hours", min_value=0, max_value=40, value=15, step=1)
            previous_scores = st.slider("Previous Academic Exam Score", min_value=0, max_value=100, value=70, step=1)
            motivation = st.select_slider("Student Motivation Level", options=["Low", "Medium", "High"], value="Medium")
        with col2:
            st.markdown("##### Institutional Support & Background")
            tutoring = st.slider("Weekly Tutoring Sessions Attended", min_value=0, max_value=10, value=2, step=1)
            resources = st.select_slider("Access to Learning Resources", options=["Low", "Medium", "High"], value="Medium")
            parental_involvement = st.select_slider("Parental Engagement Intensity", options=["Low", "Medium", "High"], value="Medium")
            teacher_quality = st.select_slider("Assessed Teacher Quality", options=["Low", "Medium", "High"], value="Medium")
            
        submit_btn = st.form_submit_button("Run Predictive Inference Engine")
        
        if submit_btn:
            motivation_map = {"Low": 1, "Medium": 2, "High": 3}
            resource_map = {"Low": 1, "Medium": 2, "High": 3}
            parental_map = {"Low": 1, "Medium": 2, "High": 3}
            teacher_map = {"Low": 1, "Medium": 2, "High": 3}
            
            try:
                # DYNAMIC MATRIX FIX: Build a full 19-column array initialized with zeros 
                # matching the shape expected by your fitted StandardScaler
                expected_features = 19
                full_feature_row = np.zeros(expected_features)
                
                # Fill the first 8 active numerical slots with your slider parameters
                full_feature_row[0] = hours_studied
                full_feature_row[1] = attendance
                full_feature_row[2] = parental_map[parental_involvement]
                full_feature_row[3] = resource_map[resources]
                full_feature_row[4] = tutoring
                full_feature_row[5] = previous_scores
                full_feature_row[6] = motivation_map[motivation]
                full_feature_row[7] = teacher_map[teacher_quality]
                
                # Convert to 2D row array shape required by scikit-learn
                feature_array = np.array([full_feature_row])
                
                # 1. Restandardize feature inputs safely using the full 19-column array shape
                scaled_features = live_scaler.transform(feature_array)
                
                # 2. Extract linear predictive point inference
                predicted_score = lr_model.predict(scaled_features)[0]
                predicted_score = max(0.0, min(100.0, float(predicted_score)))
                
                st.markdown("---")
                st.markdown(f"### 🎯 Predicted Final Exam Score: **{predicted_score:.2f} / 100**")
                
                if predicted_score < 50.0:
                    st.error("🚨 **High Academic Risk:** Simulated metrics fall below passing baselines. Early intervention counseling highly recommended.")
                elif predicted_score < 75.0:
                    st.warning("⚠️ **Moderate Performance Baseline:** Pass thresholds satisfied. Monitor closely to prevent potential grade drift.")
                else:
                    st.success("✨ **High Achievement Track:** Predicted metrics reflect strong subject mastery. Maintain current engagement levels!")
            except Exception as e:
                st.error(f"Execution Error: Mapping features failed. Details: {e}")
