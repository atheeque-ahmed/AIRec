# Standard library imports
import pickle

# Third party imports
import numpy as np
import pandas as pd
import streamlit as st

# Constants
MODEL_FILE = './models/model.pkl'
SCALER_FILE = './models/scaler.pkl'
USERS_FILE = './data/users.csv'

# Load the model and scaler
with open(MODEL_FILE, 'rb') as file:
    model = pickle.load(file)

with open(SCALER_FILE, 'rb') as file:
    scaler = pickle.load(file)


def sigmoid(x):
    """
    Sigmoid function for transforming data.

    Parameters:
    x (float): The value to be transformed.

    Returns:
    float: The transformed value.
    """
    return 1 / (1 + np.exp(-x))


def dwpa_prediction(student_data, model, scaler):
    """
    Predicts user preferences using the Direct Weighted Product Aggregation approach.

    Parameters:
    student_data (array-like): The user features.
    model (Model object): The trained model.
    scaler (Scaler object): The scaler object used for scaling features.

    Returns:
    array: The predicted user preferences.
    """
    student_data_scaled = scaler.transform(student_data)
    linear_output = np.dot(student_data_scaled, model.coef_.T) + model.intercept_
    return sigmoid(linear_output)


def main():
    st.title('Add New User')
    st.markdown("## Model Documentation")
    st.markdown("""
    Our model uses a Direct Weighted Product Aggregation (DWPA) approach to predict user preferences for video, audio, and text content.
    The model is trained using user data including user's hearing, vision, focus issues and language proficiency, and then applies the sigmoid function 
    to the weighted sum of these features to generate the predictions. The sigmoid function is defined as follows:
    """)
    st.latex(r"S(x) = \\frac{1}{{1+e^{-x}}}")

    user_id = st.text_input("Enter User ID")
    name = st.text_input("Enter User Name")
    hearing_issue = st.number_input("Hearing Issue (0-1)", min_value=0, max_value=1)
    vision_issue = st.number_input("Vision Issue (0-1)", min_value=0, max_value=1)
    focus_issue = st.number_input("Focus Issue (0-1)", min_value=0, max_value=1)

    direct_input = st.checkbox("Enter Language Proficiency Directly")
    if not direct_input:
        language_proficiency = st.number_input("Language Proficiency (0-1)", min_value=0.0, max_value=1.0, step=0.02)
    else:
        st.markdown("## Language Proficiency Quiz")
        quiz_questions = {
            # same as in your code
        }
        language_proficiency = 0
        for i, (question, answers) in enumerate(quiz_questions.items(), start=1):
            answer = st.selectbox(question, answers)
            points = 0.1 - 0.02 * answers.index(answer)
            language_proficiency += points
        language_proficiency /= len(quiz_questions)  # Normalize the score to 0-1 range

    if st.button("Add User"):
        input_data = np.array([[hearing_issue, vision_issue, focus_issue, language_proficiency]])
        predictions = dwpa_prediction(input_data, model, scaler)
        video_percentage, audio_percentage, text_percentage = predictions[0]

        new_user = pd.DataFrame({
            'user_id': [user_id],
            'name': [name],
            'video_percentage': [video_percentage],
            'audio_percentage': [audio_percentage],
            'text_percentage': [text_percentage]
        })

        users = pd.read_csv(USERS_FILE)
        users = users.append(new_user, ignore_index=True)
        users.to_csv(USERS_FILE, index=False)

        st.success('User added successfully!')


if __name__ == '__main__':
    main()
