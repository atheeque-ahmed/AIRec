import pandas as pd
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid import GridUpdateMode, DataReturnMode
from st_aggrid.grid_options_builder import GridOptionsBuilder

import backend as backend

# Basic webpage setup
st.set_page_config(
    page_title="Course Recommender System",
    layout="wide",
    initial_sidebar_state="expanded",
)

weights = {
    'model1': 0.2,
    'model2': 0.2,
    'model3': 0.2,
    'model4': 0.2,
    'model5': 0.2,
}


# ------- Functions ------
# Load datasets


def load_ratings():
    return backend.load_ratings()


def load_course_sims():
    return backend.load_course_sims()


def load_courses():
    return backend.load_courses()


def load_bow():
    return backend.load_bow()


def load_genre():
    return backend.load_course_genres()


def load_user_profiles():
    return backend.load_user_profiles()


def load_users():
    return backend.load_users()


def init__recommender_app(user_id):
    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        user_df = backend.load_users()
        course_bow_df = load_bow()

    # Select courses
    st.success('Datasets loaded successfully...')

    st.markdown("""---""")
    st.subheader("Select courses that you have completed: ")

    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # Create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        height=300,
        width='100%',
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=True,
    )

    if response["selected_rows"]:
        # If courses selected, update them in CSV
        for course in response["selected_rows"]:
            backend.update_completed_courses(user_id, course['COURSE_ID'])

    # Refresh the completed courses table
    completed_courses = user_df.loc[user_df['user_id'] == user_id, 'completed_courses'].values[0]
    if pd.notna(completed_courses):
        completed_courses = completed_courses.split(';')
        completed_courses_df = course_df[course_df['COURSE_ID'].isin(completed_courses)][
            ['COURSE_ID', 'TITLE']].drop_duplicates()
    else:
        completed_courses_df = pd.DataFrame(columns=['COURSE_ID', 'TITLE'])

    st.subheader("Your courses: ")
    st.table(completed_courses_df)

    st.subheader("Your Rating: ")

    course_to_rate = st.selectbox("Select a course to rate:", completed_courses_df['COURSE_ID'])
    rating = st.slider("Rate this course:", min_value=1, max_value=5)
    if st.button("Submit rating"):
        backend.update_course_rating(user_id, course_to_rate, rating)

    return completed_courses_df


def train(model_name, params):
    # Start training course similarity model
    with st.spinner('Training...'):
        backend.train(model_name, params, selected_courses_df.COURSE_ID)
        st.success('Done!')


def predict(model_name, active_user, params, user):
    res = None
    # Start making predictions based on model name, test user ids, and parameters
    with st.spinner('Generating course recommendations: '):
        # time.sleep(0.5)
        res = backend.predict(model_name, active_user, params, user)
    st.success('Recommendations generated!')
    return res


# ------ UI ------

# Sidebar
st.sidebar.title('Personalized Learning Recommender')
# Initialize the app
# Add a dropdown to select users - Modification
users_df = load_users()
selected_user = st.sidebar.selectbox('Select a User:', users_df['user_id'].unique())
print("selected user", selected_user)
selected_courses_df = init__recommender_app(selected_user)

# Model selection select box
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
    "Select model:",
    backend.models
)

# Hyperparameters for each model
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters: ')

# Course similarity model
if model_selection == backend.models[0]:
    n_rec_sim = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
    t_rec_sim = st.sidebar.slider('Course Similarity Threshold %', min_value=0, max_value=100, value=50, step=10)
    params['n_rec_sim'] = n_rec_sim
    params['t_rec_sim'] = t_rec_sim

# User profile model
elif model_selection == backend.models[1]:
    n_rec_profile = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
    t_rec_profile = st.sidebar.slider('User Profile Similarity Threshold %', min_value=0, max_value=100, value=50,
                                      step=10)
    params['n_rec_profile'] = n_rec_profile
    params['t_rec_profile'] = t_rec_profile

# Clustering model
elif model_selection == backend.models[2]:
    n_rec_clu = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
    n_clu = st.sidebar.slider('Number of Clusters', min_value=1, max_value=50, value=20, step=1)
    params['n_rec_clu'] = n_rec_clu
    params['n_clu'] = n_clu
    params['exp_var'] = 0

# Clustering + PCA model
elif model_selection == backend.models[3]:
    n_rec_clu_pca = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
    n_clu_pca = st.sidebar.slider('Number of Clusters', min_value=1, max_value=50, value=20, step=1)
    exp_var = st.sidebar.slider('Explained Variance', min_value=1, max_value=100, value=80, step=1)
    params['n_rec_clu'] = n_rec_clu_pca
    params['n_clu'] = n_clu_pca
    params['exp_var'] = exp_var

# KNN
elif model_selection == backend.models[4]:
    n_rec_knn = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
    n_neigh = st.sidebar.slider('Number of Neighbors', min_value=1, max_value=50, value=20, step=1)
    params['n_rec_knn'] = n_rec_knn
    params['n_neigh'] = n_neigh

# NMF model
elif model_selection == backend.models[5]:
    n_rec_nmf = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
    nmf_factors = st.sidebar.slider('NMF Factors', min_value=1, max_value=140, value=70, step=1)
    nmf_epochs = st.sidebar.slider('SGD Epochs', min_value=1, max_value=100, value=50, step=1)
    params['n_rec_nmf'] = n_rec_nmf
    params['nmf_factors'] = nmf_factors
    params['nmf_epochs'] = nmf_epochs

# Neural Net
elif model_selection == backend.models[6]:
    n_rec_ncf = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
    params['ncf_val_split'] = 0.1
    params['ncf_batch_size'] = 512
    params['ncf_epochs'] = 20
    with st.sidebar.expander('Advanced options.', False):
        st.caption(
            'Changing these options will cause the app to train a new model, instead of using the pretrained one.')
        st.info('Depending on your settings, training can take a significant amount of time (minutes to  many hours).')
        st.caption('Training dataset generation')
        ncf_val_split = st.slider('Validation Split', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        ncf_batch_size = st.slider('Batch size', min_value=0, max_value=1024, value=512, step=64)
        st.caption('Model training')
        ncf_epochs = st.slider('Epochs', min_value=1, max_value=100, value=20, step=1)
    params['ncf_val_split'] = ncf_val_split
    params['ncf_batch_size'] = ncf_batch_size
    params['ncf_epochs'] = ncf_epochs
    params['n_rec_ncf'] = n_rec_ncf

elif model_selection == backend.models[7]:
    course_slider = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
    t_rec_sim = st.sidebar.slider('Course Similarity Threshold %', min_value=0, max_value=100, value=50, step=10)
    params['n_rec_sim'] = course_slider
    params['t_rec_sim'] = t_rec_sim

    t_rec_profile = st.sidebar.slider('User Profile Similarity Threshold %', min_value=0, max_value=100, value=50,
                                      step=10)
    params['n_rec_profile'] = course_slider
    params['t_rec_profile'] = t_rec_profile

    clusters_slider = st.sidebar.slider('Number of Clusters', min_value=1, max_value=50, value=20, step=1)
    params['n_rec_clu'] = course_slider
    params['n_clu'] = clusters_slider
    params['exp_var'] = 0

    exp_var = st.sidebar.slider('Explained Variance', min_value=1, max_value=100, value=80, step=1)
    params['n_rec_clu'] = course_slider
    params['n_clu'] = clusters_slider
    params['exp_var'] = exp_var

    n_neigh = st.sidebar.slider('Number of Neighbors', min_value=1, max_value=50, value=20, step=1)
    params['n_rec_knn'] = course_slider
    params['n_neigh'] = n_neigh

    nmf_factors = st.sidebar.slider('NMF Factors', min_value=1, max_value=140, value=70, step=1)
    nmf_epochs = st.sidebar.slider('SGD Epochs', min_value=1, max_value=100, value=50, step=1)
    params['n_rec_nmf'] = course_slider
    params['nmf_factors'] = nmf_factors
    params['nmf_epochs'] = nmf_epochs

    params['ncf_val_split'] = 0.1
    params['ncf_batch_size'] = 512
    params['ncf_epochs'] = 20
    with st.sidebar.expander('Advanced options.', False):
        st.caption(
            'Changing these options will cause the app to train a new model, instead of using the pretrained one.')
        st.info('Depending on your settings, training can take a significant amount of time (minutes to  many hours).')
        st.caption('Training dataset generation')
        ncf_val_split = st.slider('Validation Split', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        ncf_batch_size = st.slider('Batch size', min_value=0, max_value=1024, value=512, step=64)
        st.caption('Model training')
        ncf_epochs = st.slider('Epochs', min_value=1, max_value=100, value=20, step=1)
    params['ncf_val_split'] = ncf_val_split
    params['ncf_batch_size'] = ncf_batch_size
    params['ncf_epochs'] = ncf_epochs
    params['n_rec_ncf'] = course_slider

else:
    pass

# Training
st.sidebar.subheader('3. Training: ')
training_button = st.sidebar.button("Train Model")
training_text = ''
# Start training process
if training_button:
    train(model_name=model_selection, params=params)

# Prediction
st.sidebar.subheader('4. Prediction')
# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button and selected_courses_df.shape[0] > 0:
    # Create a new id for current user session
    active_user = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)

    # testing -modified
    filtered_df = users_df[users_df['name'] == selected_user]
    print("filtered_df", filtered_df)
    if not filtered_df.empty:
        user = filtered_df.iloc[0]
    else:
        user = None  # or some other default value

    user_ids = [active_user]
    # Get and display predictions
    # - call predict with necessary parameters
    # - format received dataframe to give only 2 decimals instead of 4.
    res_df = predict(model_selection, active_user, params, selected_user)
    st.table(res_df.style.format({"SCORE": "{:.2f}"}))
