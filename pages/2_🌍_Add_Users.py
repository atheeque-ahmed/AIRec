import pickle

import numpy as np
import pandas as pd
import streamlit as st

# Load the model and scaler
with open('./models/model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('./models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define the prediction function
def dwpa_prediction(student_data, model, scaler):
    student_data_scaled = scaler.transform(student_data)
    linear_output = np.dot(student_data_scaled, model.coef_.T) + model.intercept_
    return sigmoid(linear_output)


st.title('Add New User')

# Display Documentation
st.markdown("## Model Documentation")
st.markdown("""
Our model uses a Direct Weighted Product Aggregation (DWPA) approach to predict user preferences for video, audio, and text content.
The model is trained using user data including user's hearing, vision, focus issues and language proficiency, and then applies the sigmoid function 
to the weighted sum of these features to generate the predictions. The sigmoid function is defined as follows:
""")
st.latex(r"S(x) = \frac{1}{{1+e^{-x}}}")

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
    quiz_questions = {
        'Question 1. Select the adjective sequence that aligns best with the natural English language order: "She has a ________ car.': ['beautiful red', 'red beautiful', 'stunning red', 'red stunning', 'gorgeous red'],
        'Question 2. Choose the grammatically correct verb form to complete the sentence: "If I ________ you, I would take the job.': ['were', 'am', 'is', 'was', 'be'],
        'Question 3. Select the word or phrase that accurately completes this common idiom: "A bird in the hand is worth ________ in the bush.': ['two', 'more than one', 'better than nothing', 'a lot', 'nothing'],
        'Question 4. Identify the sentence with the correct use of the relative pronoun:': ['The dog, which is brown, chased its tail.', 'The book that is on the table is mine.', 'The girl, who is wearing a blue dress, is my sister.', 'The car that is parked outside belongs to John.', 'The tree, which is in the backyard, is tall.'],
        'Question 5. Choose the grammatically correct verb form to complete the sentence: "She ________ her keys yesterday.': ['lost', 'loses', 'lose', 'has lost', 'was losing'],
        'Question 6. Identify the sentence that correctly employs the past perfect tense:': ['She had arrived home before I did.', 'They had finished their dinner when we arrived.', 'He had left for work before we woke up.', 'We had gone to the beach the day before.', 'The movie had started at 7 p.m.'],
        'Question 7. Choose the correct word to complete the analogy: "Book is to reading as fork is to ________.': ['eating', 'writing', 'cooking', 'cutting', 'holding'],
        'Question 8. Identify the sentence that appropriately uses a past participle:': ['The broken glass needs to be cleaned up.', 'She had gone to the store.', 'They have been running in the park.', 'The movie will start soon.', 'I have been studying all day.'],
        'Question 9. Choose the correct verb form to complete the sentence: "The sun ________ in the east.': ['rises', 'raises', 'rose', 'has risen', 'is rising'],
        'Question 10. Identify the sentence that correctly employs a comparative adjective:': ['This book is better than that one.', 'She is the most beautiful girl in the class.', 'He is taller than I am.', 'The dog ran faster than its friend.', 'It is the worst movie I\'ve ever seen.'],
    }
    language_proficiency = 0
    for i, (question, answers) in enumerate(quiz_questions.items(), start=1):
        answer = st.selectbox(question, answers)
        points = 0.1 - 0.02 * answers.index(answer)
        language_proficiency += points
    language_proficiency /= len(quiz_questions)  # Normalize the score to 0-1 range

# Button to make predictions and save the user
if st.button("Add User"):
    # Prepare the input data
    input_data = np.array([[hearing_issue, vision_issue, focus_issue, language_proficiency]])

    # Make predictions
    predictions = dwpa_prediction(input_data, model, scaler)
    video_percentage, audio_percentage, text_percentage = predictions[0]

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

    # Append the new user
    users = users.append(new_user, ignore_index=True)

    # Save the updated users
    users.to_csv('./data/users.csv', index=False)

    st.success('User added successfully!')
