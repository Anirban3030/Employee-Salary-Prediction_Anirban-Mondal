import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Salary Prediction", layout="centered")

# Title
st.title("💼 Employee Salary Prediction App")

# Upload CSV dataset
uploaded_file = st.file_uploader("📂 Upload Dataset (CSV format)", type=["csv"])

# Load model
@st.cache_resource
def load_model():
    return joblib.load("salary_prediction_model.pkl")

model = load_model()

# Once file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # Extract dropdown options from the dataset
    gender_options = sorted(df["Gender"].dropna().unique())
    education_options = sorted(df["Education Level"].dropna().unique())
    job_title_options = sorted(df["Job Title"].dropna().unique())

    st.subheader("📥 Enter Employee Details")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, step=1)
            gender = st.selectbox("Gender", options=gender_options)
            education = st.selectbox("Education Level", options=education_options)

        with col2:
            years_exp = st.number_input("Years of Experience", min_value=0, step=1)
            job_title = st.selectbox("Job Title", options=job_title_options)

        submit_btn = st.form_submit_button("🔮 Predict Salary")

    if submit_btn:
        # Create a DataFrame for prediction
        input_df = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Education Level": [education],
            "Job Title": [job_title],
            "Years of Experience": [years_exp]
        })

        # Ensure same preprocessing is applied if needed
        prediction = model.predict(input_df)[0]

        st.success(f"💰 Predicted Salary: ₹ {prediction:,.2f}")

else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
