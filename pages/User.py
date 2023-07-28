import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import backend as backend


st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

def load_users():
    return backend.load_users()


def load_courses():
    return backend.load_courses()


users_df = load_users()
courses_df = load_courses()

selected_user = st.sidebar.selectbox('Select a User:', users_df['user_id'].unique())

# Find the user in the DataFrame
selected_user_row = users_df[users_df['user_id'] == selected_user]

# Display the user's video, audio, and text percentages
video_percentage = selected_user_row['video_percentage'].values[0]
audio_percentage = selected_user_row['audio_percentage'].values[0]
text_percentage = selected_user_row['text_percentage'].values[0]

st.title(f"Selected User: {selected_user}")

st.header("Preferences")
st.subheader("Video Percentage")
st.progress(video_percentage)
st.write(f'Percentage: {video_percentage * 100}%')

st.subheader("Audio Percentage")
st.progress(audio_percentage)
st.write(f'Percentage: {audio_percentage * 100}%')

st.subheader("Text Percentage")
st.progress(text_percentage)
st.write(f'Percentage: {text_percentage * 100}%')

# Calculate average percentages for all users
avg_video = users_df['video_percentage'].mean()
avg_audio = users_df['audio_percentage'].mean()
avg_text = users_df['text_percentage'].mean()

# Create a DataFrame for the chart
data = pd.DataFrame({
    'Preference': ['Video', 'Audio', 'Text'],
    'Average': [avg_video, avg_audio, avg_text],
    selected_user: [video_percentage, audio_percentage, text_percentage]
})

# Create a new figure
fig, ax = plt.subplots()

# Line plot for each preference type
ax.plot(data['Preference'], data['Average'], marker='o', label='Average')
ax.plot(data['Preference'], data[selected_user], marker='o', label=selected_user)

ax.set_xlabel('Preferences')
ax.set_ylabel('Percentage')
ax.legend()

st.pyplot(fig)

completed_courses = users_df.loc[users_df['user_id'] == selected_user, 'completed_courses'].values[0]

if pd.notna(completed_courses):
    completed_courses = completed_courses.split(';')
    completed_courses_df = courses_df[courses_df['COURSE_ID'].isin(completed_courses)][
        ['COURSE_ID', 'TITLE', 'FORMAT']].drop_duplicates()
else:
    completed_courses_df = pd.DataFrame(columns=['COURSE_ID', 'TITLE', 'FORMAT'])

st.header("Completed Courses")
st.table(completed_courses_df)
