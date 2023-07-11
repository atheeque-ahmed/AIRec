# Standard library imports
import matplotlib.pyplot as plt

# Third party imports
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import OneHotEncoder

# Constants
USERS_FILE = './data/users.csv'
DATASET_FILE = './data/preferred_learning_strategies_dataset.csv'


def load_data():
    """
    Load the users data and add a new column for the count of completed courses.

    Returns:
    DataFrame: The loaded data.
    """
    df = pd.read_csv(USERS_FILE)
    df['completed_courses_count'] = df['completed_courses'].str.split(';').str.len()
    return df


def load_dataset_data():
    """
    Load the preferred learning strategies dataset.

    Returns:
    DataFrame: The loaded data.
    """
    df = pd.read_csv(DATASET_FILE)
    return df


def plot_preferences(df):
    """
    Plot the average preference percentages.

    Parameters:
    df (DataFrame): The data.
    """
    avg_pref = df[['video_percentage', 'audio_percentage', 'text_percentage']].mean()
    st.header('Average User Preferences')
    st.bar_chart(avg_pref)


def plot_completed_courses(df):
    """
    Plot the histogram of the number of completed courses.

    Parameters:
    df (DataFrame): The data.
    """
    st.header('Distribution of Completed Courses')
    st.bar_chart(df['completed_courses_count'])


def plot_heatmap(df):
    """
    Plot a heatmap of the correlation between user preferences.

    Parameters:
    df (DataFrame): The data.
    """
    corr = df[['video_percentage', 'audio_percentage', 'text_percentage']].corr()
    st.header('Heatmap of User Preferences')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot(plt)


def plot_dataset_data_heatmap(df):
    """
    Plot a heatmap of the correlation between features in the initial dataset.

    Parameters:
    df (DataFrame): The data.
    """
    enc = OneHotEncoder(sparse=False)
    encoded_strategy = enc.fit_transform(df[['Preferred_Learning_Strategy']])
    df_encoded = pd.DataFrame(encoded_strategy, columns=enc.get_feature_names_out(['Preferred_Learning_Strategy']))
    df = pd.concat([df.drop('Preferred_Learning_Strategy', axis=1), df_encoded], axis=1)
    corr = df.corr()
    st.header('Heatmap of Initial Dataset')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot(plt)


def main():
    st.title('User Analysis Dashboard')
    df = load_data()
    df2 = load_dataset_data()
    st.header('User Data')
    st.write(df)

    plot_preferences(df)
    plot_completed_courses(df)
    plot_heatmap(df)

    for feature in ['Hearing_Issue', 'Vision_Issue', 'Focus_Issue']:
        st.header(f'Preferred Learning Strategy distribution for different {feature} values')
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, hue='Preferred_Learning_Strategy', data=df2)
        st.pyplot(plt)

    st.header('Preferred Learning Strategy distribution for different Language Proficiency levels')
    plt.figure(figsize=(10, 6))
    df2['Language_Proficiency_Bin'] = pd.cut(df2['Language_Proficiency'], bins=3, labels=['Low', 'Medium', 'High'])
    sns.countplot(x='Language_Proficiency_Bin', hue='Preferred_Learning_Strategy', data=df2)
    st.pyplot(plt)
    plot_dataset_data_heatmap(df2)


if __name__ == '__main__':
    main()
