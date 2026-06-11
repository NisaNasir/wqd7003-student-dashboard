import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import plotly.express as px

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
st.sidebar.markdown("## 📊 Deployment Profile")
st.sidebar.info("This dashboard deploys our final selected Linear Regression baseline model calibrated to the official Universiti Malaya grading structure.")
st.sidebar.markdown("**Model Performance Metrics:**")
st.sidebar.markdown("- **R² Variance Explained:** 76.92%")
st.sidebar.markdown("- **Mean Absolute Error (MAE):** 0.4620")
st.sidebar.markdown("- **Root Mean Squared Error (RMSE):** 1.8061")

# 4. Interface Tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Strategic System Insights", 
    "🔮 Individual Student Risk Predictor", 
    "📦 Semester Batch Processing & Visual Analytics"
])

# ==========================================
# TAB 1: STRATEGIC INSIGHTS
# ==========================================
with tab1:
    st.subheader("📋 System Executive Insights Summary")
    st.markdown("### 🔑 Key Operational Findings")
    st.markdown(
        """
        Our underlying statistical framework automatically reveals that final student performance is heavily driven by 
        **behavioral metrics and active engagement levels** rather than demographic background or fixed socioeconomic characteristics:
        1. **Attendance:** Identified as the absolute primary driver within the underlying dataset. Consistent exposure to the classroom curriculum remains vital.
        2. **Hours Studied:** Represents the secondary behavioral lever, proving independent home revision strongly updates score profiles.
        3. **Previous Scores:** Highlights historical academic continuity.
        
        *Operational Strategy:* By selecting Linear Regression, educational institutions gain complete parameter transparency—allowing advisors to mathematically pinpoint exactly how shifting an attendance track will impact final performance grades.
        """
    )

# ==========================================
# TAB 2: INDIVIDUAL SIMULATOR
# ==========================================
with tab2:
    st.subheader("🕹️ Interactive Student Metric Simulator")
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
            st.markdown("##### 🏫 Institutional Support & Background")
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
                expected_features = 19
                full_feature_row = np.zeros(expected_features)
                
                full_feature_row[0] = hours_studied
                full_feature_row[1] = attendance
                full_feature_row[2] = parental_map[parental_involvement]
                full_feature_row[3] = resource_map[resources]
                full_feature_row[4] = tutoring
                full_feature_row[5] = previous_scores
                full_feature_row[6] = motivation_map[motivation]
                full_feature_row[7] = teacher_map[teacher_quality]
                
                feature_array = np.array([full_feature_row])
                scaled_features = live_scaler.transform(feature_array)
                predicted_score = live_lr.predict(scaled_features)[0]
                predicted_score = max(0.0, min(100.0, float(predicted_score)))
                
                st.markdown("---")
                st.markdown(f"### 🎯 Predicted Final Exam Score: **{predicted_score:.2f} / 100**")
                
                if predicted_score < 50.0:
                    st.error("🚨 **High Academic Risk (UM Fail Track):** Predicted metrics fall into the C- to F grade band. Immediate structural intervention highly recommended.")
                elif predicted_score < 70.0:
                    st.warning("⚠️ **Moderate Performance Baseline (UM Pass/Good Track):** Predicted metrics fall into the C to B grade band. Progress stable, monitor closely to prevent down-drifts.")
                else:
                    st.success("✨ **Low Risk / High Achievement Track (UM Distinction Track):** Predicted metrics reflect strong performance trending in the B+ to A- tier. Maintain current metrics!")
            except Exception as e:
                st.error(f"Execution Error: Mapping features failed. Details: {e}")

# ==========================================
# TAB 3: BATCH PROCESSING (PLOTLY SORT FIXED)
# ==========================================
with tab3:
    st.subheader("📦 Institutional New Batch Operations Engine")
    st.markdown("Upload entire student cohorts to instantly run bulk risk predictions based on the official UM grading guidelines.")
    
    st.markdown("#### 📥 Step 1: Upload New Semester Student Cohort")
    uploaded_file = st.file_uploader("Choose a CSV Batch File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df_batch)} student rows from the current semester!")
            
            motivation_map = {"Low": 1, "Medium": 2, "High": 3}
            resource_map = {"Low": 1, "Medium": 2, "High": 3}
            parental_map = {"Low": 1, "Medium": 2, "High": 3}
            teacher_map = {"Low": 1, "Medium": 2, "High": 3}
            
            h_stud = df_batch['Hours_Studied'] if 'Hours_Studied' in df_batch.columns else df_batch.iloc[:, 0]
            atten = df_batch['Attendance'] if 'Attendance' in df_batch.columns else df_batch.iloc[:, 1]
            prev_s = df_batch['Previous_Scores'] if 'Previous_Scores' in df_batch.columns else df_batch.iloc[:, 5]
            tutoring = df_batch['Tutoring_Sessions'] if 'Tutoring_Sessions' in df_batch.columns else 0
            
            mot = df_batch['Motivation_Level'].map(motivation_map).fillna(2) if 'Motivation_Level' in df_batch.columns else 2
            res = df_batch['Access_to_Resources'].map(resource_map).fillna(2) if 'Access_to_Resources' in df_batch.columns else 2
            par = df_batch['Parental_Involvement'].map(parental_map).fillna(2) if 'Parental_Involvement' in df_batch.columns else 2
            tea = df_batch['Teacher_Quality'].map(teacher_map).fillna(2) if 'Teacher_Quality' in df_batch.columns else 2
            
            # Reconstruct matrix
            batch_features = np.zeros((len(df_batch), 19))
            batch_features[:, 0] = h_stud
            batch_features[:, 1] = atten
            batch_features[:, 2] = par
            batch_features[:, 3] = res
            batch_features[:, 4] = tutoring
            batch_features[:, 5] = prev_s
            batch_features[:, 6] = mot
            batch_features[:, 7] = tea
            
            # Predict
            scaled_batch = live_scaler.transform(batch_features)
            predictions = live_lr.predict(scaled_batch)
            df_batch['Predicted_Exam_Score'] = np.clip(predictions, 0.0, 100.0)
            
            def classify_risk_um(score):
                if score < 50.0: return "High Risk (<50)"
                elif score < 70.0: return "Moderate Risk (50-70)"
                else: return "Low Risk (>=70)"
                
            df_batch['Risk_Classification'] = df_batch['Predicted_Exam_Score'].apply(classify_risk_um)
            
            # Pre-sort entire dataframe by score descending for granular view
            df_batch = df_batch.sort_values('Predicted_Exam_Score', ascending=False)

            # --- VISUAL ANALYTICS SECTION ---
            st.markdown("#### 📊 Step 2: Cohort Risk Visual Analytics")
            
            high_risk_count = int(sum(df_batch['Risk_Classification'] == "High Risk (<50)"))
            mod_risk_count = int(sum(df_batch['Risk_Classification'] == "Moderate Risk (50-70)"))
            low_risk_count = int(sum(df_batch['Risk_Classification'] == "Low Risk (>=70)"))
            
            m1, m2, m3 = st.columns(3)
            m1.metric("🚨 High Risk (UM Fail Track)", f"{high_risk_count} students")
            m2.metric("⚠️ Moderate Risk (UM Pass/Good Track)", f"{mod_risk_count} students")
            m3.metric("✨ Low Risk (UM Distinction Track)", f"{low_risk_count} students")
            
            chart_col1, chart_col2 = st.columns(2)
            academic_order = ["Low Risk (>=70)", "Moderate Risk (50-70)", "High Risk (<50)"]
            
            with chart_col1:
                st.markdown("##### 📈 Cohort Risk Breakdown (UM Standards)")
                summary_df = pd.DataFrame({
                    "Risk Level": academic_order,
                    "Student Count": [low_risk_count, mod_risk_count, high_risk_count]
                })
                
                fig_bar = px.bar(
                    summary_df, 
                    x="Risk Level", 
                    y="Student Count",
                    color="Risk Level",
                    category_orders={"Risk Level": academic_order},
                    color_discrete_map={
                        "Low Risk (>=70)": "#2ecc71",
                        "Moderate Risk (50-70)": "#f1c40f",
                        "High Risk (<50)": "#e74c3c"
                    }
                )
                fig_bar.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_bar, use_container_width=True)
                st.caption("Figure 1: Aggregated distribution of student cohorts mapping to UM academic performance risk thresholds.")

            with chart_col2:
                st.markdown("##### 🔍 Behavioral Driver Mapping (Attendance vs. Study Hours)")
                fig_scatter = px.scatter(
                    df_batch,
                    x="Attendance",
                    y="Hours_Studied",
                    color="Risk_Classification",
                    category_orders={"Risk_Classification": academic_order},
                    color_discrete_map={
                        "Low Risk (>=70)": "#2ecc71",
                        "Moderate Risk (50-70)": "#f1c40f",
                        "High Risk (<50)": "#e74c3c"
                    }
                )
                fig_scatter.update_layout(margin=dict(l=20, r=20, t=20, b=20), legend_title_text='Risk Classification')
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.caption("Figure 2: Multi-dimensional clusters segmenting student status directly across primary behavioral variables.")
            
            # Data Table
            st.markdown("#### 📋 Step 3: Granular Batch Records Table")
            st.dataframe(df_batch[['Hours_Studied', 'Attendance', 'Previous_Scores', 'Predicted_Exam_Score', 'Risk_Classification']], use_container_width=True)
            
            csv_buffer = io.StringIO()
            df_batch.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Export Prediction Report to CSV/Excel",
                data=csv_data,
                file_name="um_semester_batch_predictions.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Batch Processing Error: Please check that your file headers line up correctly. Details: {e}")
            
    # Segment B: Retraining pipeline instructions
    st.markdown("---")
    st.markdown("#### 🔄 Step 4: End-of-Semester Automated Model Retraining")
    st.markdown(
        """
        **How the Automated Retraining Loop works:**
        1. Append the new semester data to your historical master dataset (`Master_Student_Performance.csv`).
        2. Run your connected **Google Colab Notebook Pipeline** from top to bottom.
        3. The notebook will automatically execute Section 4 (Model Selection), retest all non-linear ensembles against Linear Regression, and pick the fresh math champion.
        4. Section 6 will automatically write the newly trained `.joblib` model assets over your existing files.
        5. Push the updated files to GitHub, click **Rerun** on this dashboard, and your system updates instantly without breaking any front-end architecture.
        """
    )
    st.success("🔄 MLOps Pipeline Ready for Next Semester Lifecycle Integration.")
