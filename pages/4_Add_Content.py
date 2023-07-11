# Standard library imports
from collections import Counter

# Third party imports
import nltk
import numpy as np
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pandas import read_csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from textblob import TextBlob

# Constants
DATA_PATH = './data/'
COURSES_BOWS_FILE = DATA_PATH + 'courses_bows.csv'
COURSE_PROCESSED_FILE = DATA_PATH + 'course_processed.csv'
COURSE_GENRES_FILE = DATA_PATH + 'course_genres.csv'
SIM_FILE = DATA_PATH + 'sim.csv'

# Download NLTK packages
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# Setting Streamlit page config
st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

# Genre Keywords Dictionary
genre_keywords = {
    'Database': ['database', 'sql', 'nosql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'dbms', 'data management',
                 'relational database', 'sql server', 'sqlite', 'query', 'schema', 'db2', 'mariadb', 'firebase',
                 'data store'],
    'Python': ['python', 'pandas', 'numpy', 'scipy', 'matplotlib', 'pytorch', 'tensorflow', 'scikit-learn', 'keras',
               'flask', 'django', 'jupyter', 'ipython', 'pydata', 'seaborn', 'plotly', 'bokeh', 'beautifulsoup',
               'requests'],
    'CloudComputing': ['cloud', 'aws', 'amazon web services', 'gcp', 'google cloud', 'azure', 'microsoft azure',
                       'docker', 'kubernetes', 'devops', 'iaas', 'paas', 'saas', 'cloud storage', 'cloud computing',
                       'serverless', 'microservices', 'virtual machine', 'vpc', 'ec2', 's3', 'lambda', 'eks', 'gke',
                       'akka'],
    'DataAnalysis': ['data', 'analysis', 'statistics', 'data analysis', 'regression', 'probability',
                     'statistical analysis', 'data mining', 'data exploration', 'data visualization', 'data cleaning',
                     'pandas', 'numpy', 'data processing', 'time series', 'data transformation', 'data manipulation'],
    'Containers': ['docker', 'kubernetes', 'container', 'microservices', 'dockerfile', 'containerization',
                   'docker-compose', 'container orchestration', 'docker swarm', 'helm', 'openshift', 'rancher',
                   'container runtime', 'container cluster', 'pod', 'docker image', 'container registry', 'docker hub'],
    'MachineLearning': ['machine learning', 'ai', 'artificial intelligence', 'neural network', 'deep learning', 'ml',
                        'supervised learning', 'unsupervised learning', 'reinforcement learning', 'regression',
                        'classification', 'clustering', 'svm', 'support vector machine', 'knn', 'k-nearest neighbors',
                        'naive bayes', 'decision tree', 'random forest', 'gradient boosting', 'xgboost', 'lightgbm',
                        'perceptron', 'cnn', 'convolutional neural network', 'rnn', 'recurrent neural network', 'lstm',
                        'long short-term memory', 'gan', 'generative adversarial network'],
    'ComputerVision': ['computer vision', 'image processing', 'opencv', 'image recognition', 'object detection',
                       'object recognition', 'image segmentation', 'image analysis', 'cnn',
                       'convolutional neural network', 'deep learning', 'image classification', 'face recognition',
                       'ocr', 'optical character recognition', 'augmented reality', 'virtual reality', 'ar', 'vr'],
    'DataScience': ['data science', 'machine learning', 'statistics', 'data analysis', 'statistical analysis',
                    'data mining', 'data visualization', 'predictive modeling', 'data modeling', 'big data',
                    'r programming', 'python', 'pandas', 'numpy', 'data exploration', 'data cleaning',
                    'machine learning algorithms', 'data processing', 'time series', 'data transformation',
                    'data manipulation'],
    'BigData': ['big data', 'hadoop', 'spark', 'mapreduce', 'hive', 'pig', 'bigtable', 'hbase', 'cloudera', 'flume',
                'sqoop', 'zookeeper', 'oozie', 'hdfs', 'yarn', 'kafka', 'storm', 'samza', 'flink', 'beam', 'bigquery',
                'data lake', 'data warehouse', 'distributed computing', 'parallel computing', 'data processing',
                'data ingestion', 'data storage', 'data streaming'],
    'Chatbot': ['chatbot', 'nlp', 'natural language processing', 'dialogflow', 'lex', 'alexa', 'siri',
                'google assistant', 'bot', 'messaging', 'conversation', 'conversational interface', 'voice assistant',
                'voice recognition', 'text-to-speech', 'speak recognition', 'language understanding',
                'intent recognition', 'speech synthesis', 'turing test', 'alexa skills kit', 'actions on google'],
    'R': ['r programming', 'r language', 'r', 'tidyverse', 'ggplot2', 'dplyr', 'tidyr', 'shiny', 'rstudio', 'cran',
          'data.frame', 'lattice', 'knitr', 'rmarkdown', 'bioconductor', 'statistical computing', 'r script',
          'r package'],
    'BackendDev': ['backend', 'node.js', 'django', 'flask', 'api', 'rest', 'graphql', 'server', 'database',
                   'web server', 'express.js', 'ruby on rails', 'laravel', 'spring boot', '.net', 'php', 'java', 'ruby',
                   'python', 'go', 'c#', 'rust', 'sql', 'nosql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'dbms',
                   'data management', 'relational database', 'sql server', 'sqlite', 'query', 'schema', 'db2',
                   'mariadb', 'firebase', 'data store'],
    'FrontendDev': ['frontend', 'javascript', 'html', 'css', 'react', 'vue.js', 'angular', 'svelte', 'jquery',
                    'bootstrap', 'web design', 'responsive design', 'web development', 'browser', 'dom',
                    'document object model', 'ajax', 'json', 'xml', 'http', 'web page', 'website', 'web application',
                    'ui', 'user interface', 'ux', 'user experience', 'web form', 'navigation', 'layout', 'typography',
                    'color', 'icon', 'image', 'graphic', 'animation', 'motion', 'transition', 'transform', 'filter',
                    'gradient', 'flexbox', 'grid'],
    'Blockchain': ['blockchain', 'bitcoin', 'ethereum', 'cryptocurrency', 'crypto', 'block', 'chain',
                   'distributed ledger', 'ledger', 'decentralized', 'consensus', 'proof of work', 'proof of stake',
                   'hash', 'transaction', 'wallet', 'public key', 'private key', 'smart contract', 'dapp',
                   'decentralized application', 'token', 'coin', 'mining', 'bitcoin mining', 'btc', 'eth', 'xrp',
                   'ripple', 'litecoin', 'ltc', 'bitcoincash', 'bch', 'eos', 'stellar', 'xlm', 'tron', 'trx', 'cardano',
                   'ada', 'monero', 'xmr', 'dash', 'iota', 'miota', 'binance coin', 'bnb']
}



def process_text(text):
    """
    Process the given text by converting to lower case, tokenizing, removing stop words, and lemmatizing.

    Parameters:
    text (str): The text to process.

    Returns:
    str: The processed text.
    """
    text = text.lower()  # Convert text to lower case
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(words)
    return text


def analyze_sentiment(text):
    """
    Analyze the sentiment of the given text.

    Parameters:
    text (str): The text to analyze.

    Returns:
    float: The sentiment polarity (-1 to 1, where -1 is negative sentiment, 1 is positive sentiment).
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity


def calculate_genres(course_id, title, description):
    """
    Calculate the genres of a course based on its title and description.

    Parameters:
    course_id (str): The ID of the course.
    title (str): The title of the course.
    description (str): The description of the course.

    Returns:
    dict: A dictionary where the keys are the genres and the values are 1 if the genre is present, 0 otherwise.
    """
    new_genres = {genre: 0 for genre in genre_keywords.keys()}
    text = title.lower() + " " + description.lower()
    for genre, keywords in genre_keywords.items():
        if any(keyword in text for keyword in keywords):
            new_genres[genre] = 1
    return new_genres


def update_bow(course_id, processed_transcript):
    """
    Update the bag of words with the given course transcript.

    Parameters:
    course_id (str): The ID of the course.
    processed_transcript (str): The processed transcript of the course.
    """
    tokens = nltk.word_tokenize(processed_transcript)
    token_counts = Counter(tokens)
    df_bows = read_csv(COURSES_BOWS_FILE)
    new_rows = []
    for token, count in token_counts.items():
        new_rows.append({'doc_index': len(df_bows), 'doc_id': course_id, 'token': token, 'bow': count})
    df_bows = df_bows.append(new_rows, ignore_index=True)
    df_bows.to_csv(COURSES_BOWS_FILE, index=False)


def add_course(course_id, title, description, file_format, transcript):
    """
    Add a new course to the dataset.

    Parameters:
    course_id (str): The ID of the course.
    title (str): The title of the course.
    description (str): The description of the course.
    file_format (str): The format of the course (VIDEO, AUDIO, TEXT).
    transcript (str): The transcript or content of the course.
    """
    processed_text = process_text(transcript)
    sentiment_score = analyze_sentiment(processed_text)
    new_data = {
        'COURSE_ID': course_id,
        'TITLE': title,
        'DESCRIPTION': description,
        'FORMAT': file_format,
        'TRANSCRIPT': processed_text,
        'SENTIMENT': sentiment_score
    }
    df = read_csv(COURSE_PROCESSED_FILE)
    df = df.append(new_data, ignore_index=True)
    df.to_csv(COURSE_PROCESSED_FILE, index=False)
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['TITLE'] + ' ' + df['DESCRIPTION'])
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    np.savetxt(SIM_FILE, cosine_similarities, delimiter=",")
    new_genres = calculate_genres(course_id, title, description)
    df_genres = read_csv(COURSE_GENRES_FILE)
    df_genres = df_genres.append({**{'COURSE_ID': course_id, 'TITLE': title}, **new_genres}, ignore_index=True)
    df_genres.to_csv(COURSE_GENRES_FILE, index=False)
    update_bow(course_id, processed_text)


def main():
    st.title('Course Information Form')
    course_id = st.text_input('Course ID')
    title = st.text_input('Title')
    description = st.text_input('description')
    file_format = st.selectbox('Format', options=['VIDEO', 'AUDIO', 'TEXT'])
    if file_format in ['VIDEO', 'AUDIO']:
        transcript = st.text_area('Transcript')
    else:
        transcript = st.text_area('Content')
    if st.button('Submit'):
        add_course(course_id, title, description, file_format, transcript)
        st.success('Successfully added new course data!')


if __name__ == '__main__':
    main()
