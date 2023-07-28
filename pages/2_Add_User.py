import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import json

# Load the model and the label encoder
with open('./models/ensemble_model.joblib', 'rb') as file:
    model = load(file)

with open('./models/label_encoder.joblib', 'rb') as file:
    le = load(file)


# Define the prediction function
def predict_preferred_content_type(hearing_issue, vision_issue, focus_issue, language_proficiency):
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame({
        'Hearing_Issue': [hearing_issue],
        'Vision_Issue': [vision_issue],
        'Focus_Issue': [focus_issue],
        'Language_Proficiency': [language_proficiency]
    })

    # Make a prediction
    prediction_proba = model.predict_proba(input_data)

    # Create a DataFrame with the probabilities
    proba_df = pd.DataFrame(prediction_proba, columns=le.classes_)

    # Return the probabilities
    return proba_df.iloc[0]


st.title('Add New User')

# Display Documentation
st.markdown("## Model Documentation")
st.markdown("""
Our model uses an ensemble approach to predict user preferences for video, audio, and text content.
The model is trained using user data including user's hearing, vision, focus issues and language proficiency.
""")

# Input fields
user_id = st.text_input("Enter User ID")
name = st.text_input("Enter User Name")
hearing_issue = st.number_input("Hearing Issue (0-1)", min_value=0, max_value=1)
vision_issue = st.number_input("Vision Issue (0-1)", min_value=0, max_value=1)
focus_issue = st.number_input("Focus Issue (0-1)", min_value=0, max_value=1)

# Language proficiency input
direct_input = st.checkbox("Enter Language Proficiency Directly")
if not direct_input:
    language_proficiency = st.number_input("Language Proficiency (0-1)", min_value=0.0, max_value=1.0, step=0.02)
else:
    # Language proficiency quiz
    st.markdown("## Language Proficiency Quiz")

    with open('./data/quiz_questions.json', 'r') as file:
        quiz_questions = json.load(file)

    language_proficiency = 0
    for i, (question, answers) in enumerate(quiz_questions.items(), start=1):
        answer = st.selectbox(question, answers)
        points = 0.1 - 0.02 * answers.index(answer)
        language_proficiency += points
    language_proficiency /= len(quiz_questions)  # Normalize the score to 0-1 range

# Button to make predictions and save the user
if st.button("Add User"):
    # Make predictions
    predictions = predict_preferred_content_type(hearing_issue, vision_issue, focus_issue, language_proficiency)
    video_percentage, audio_percentage, text_percentage = predictions['Video'], predictions['Audio'], predictions[
        'Text']

    # Create a DataFrame for the new user
    new_user = pd.DataFrame({
        'user_id': [user_id],
        'name': [name],
        'video_percentage': [video_percentage],
        'audio_percentage': [audio_percentage],
        'text_percentage': [text_percentage]
    })

    # Load the existing users
    users = pd.read_csv('./data/users.csv')

    print(type(users))
    print(type(new_user))

    # Append the new user
    users = users.append(new_user, ignore_index=True)

    # Save the updated users
    users.to_csv('./data/users.csv', index=False)

    st.success('User added successfully!')
