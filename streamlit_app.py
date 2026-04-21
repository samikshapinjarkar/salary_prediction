
import streamlit as st
import pandas as pd
import joblib
# from sklearn.preprocessing import LabelEncoder # Not using directly due to original notebook's LabelEncoder usage

# Load the trained model
try:
    model = joblib.load('best_salary_prediction_model.pkl')
except FileNotFoundError:
    st.error("Model file 'best_salary_prediction_model.pkl' not found. Please ensure the model is trained and saved correctly.")
    st.stop()

st.title('Salary Prediction App')
st.write('Enter the employee details to predict their salary.')

# --- Input Fields ---
age = st.number_input('Age', min_value=18, max_value=100, value=30, step=1)
years_experience = st.number_input('Years of Experience', min_value=0.0, max_value=50.0, value=5.0, step=0.5)

# Gender - Inferring mapping from notebook's df.head() output after encoding
# Male: 1, Female: 0
gender_map = {'Male': 1, 'Female': 0}
selected_gender = st.selectbox('Gender', list(gender_map.keys()))
gender_encoded = gender_map[selected_gender]

# Education Level - Inferring partial mapping from notebook's df.head() output after encoding
# Bachelor's Degree: 0, Master's Degree: 3, PhD: 5
# WARNING: This mapping is incomplete and inferred from a partial view of the original data.
# For a robust app, the exact LabelEncoder mapping from training should be saved and loaded.
education_map = {
    'Bachelor\'s Degree': 0,
    'Master\'s Degree': 3,
    'PhD': 5
    # Add other education levels with their correct encoded values if known from training
    # e.g., 'High School': X, 'Some College': Y - these were not clearly mapped in the available df.head()
}
education_level_display = st.selectbox('Education Level', list(education_map.keys()))
education_level_encoded = education_map[education_level_display]


# Job Title - This is problematic due to high cardinality and the way LabelEncoder was used in training.
# The original notebook used a single LabelEncoder instance for multiple columns,
# meaning the exact mapping for 'Job Title' is not easily recoverable without saving the
# specific LabelEncoder instance or its 'classes_' attribute. For a production app,
# the original fitted LabelEncoder for Job Title should be saved.
st.subheader("Job Title Input (Critical Limitation)")
st.warning("Due to the way LabelEncoder was applied in the training notebook, the exact string-to-integer mapping for 'Job Title' is not directly recoverable. For a robust deployment, the original fitted LabelEncoder for 'Job Title' (or its 'classes_' attribute) must be saved and loaded. For this demo, please enter the encoded integer value.")
st.write("Examples from training data (Job Title -> Encoded Value):")
st.write("- Software Engineer -> 177")
st.write("- Data Analyst -> 18")
st.write("- Senior Manager -> 145")
st.write("- Sales Associate -> 116")
st.write("- Director -> 26")
job_title_encoded = st.number_input('Job Title (Encoded Integer)', min_value=0, value=177, step=1)


if st.button('Predict Salary'):
    # Create a DataFrame for the new input
    input_data = pd.DataFrame([[
        age,
        gender_encoded,
        education_level_encoded,
        job_title_encoded,
        years_experience
    ]],
    columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f'Predicted Salary: ${prediction:,.2f}')

