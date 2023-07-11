import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from textblob import TextBlob
from collections import Counter

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

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


def update_bow(course_id, processed_transcript):
    # Tokenize the processed transcript
    tokens = nltk.word_tokenize(processed_transcript)

    # Calculate the frequency of each token
    token_counts = Counter(tokens)

    # Load the existing bag of words data
    df_bows = pd.read_csv('./data/courses_bows.csv')

    # For each unique token, add a new row to the dataframe
    for token, count in token_counts.items():
        df_bows = df_bows.append({'doc_index': len(df_bows), 'doc_id': course_id, 'token': token, 'bow': count},
                                 ignore_index=True)

    # Save the updated dataframe to CSV
    df_bows.to_csv('./data/courses_bows.csv', index=False)


def calculate_genres(course_id, title, description):
    # Initialize a dictionary to hold the genres of the new course
    new_genres = {genre: 0 for genre in genre_keywords.keys()}

    # Combine the title and description into a single string
    text = title.lower() + " " + description.lower()

    # Check if any of the keywords for each genre appear in the text
    for genre, keywords in genre_keywords.items():
        if any(keyword in text for keyword in keywords):
            new_genres[genre] = 1

    return new_genres


def process_text(text):
    text = text.lower()  # Convert text to lower case

    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join words back into a single string
    text = ' '.join(words)

    return text


def analyze_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)

    # Return the sentiment polarity (-1 to 1, where -1 is negative sentiment, 1 is positive sentiment)
    return blob.sentiment.polarity


def add_course(course_id, title, description, file_format, transcript):
    # Process the text
    processed_text = process_text(transcript)

    # Perform sentiment analysis
    sentiment_score = analyze_sentiment(processed_text)

    new_data = {'COURSE_ID': course_id,
                'TITLE': title,
                'DESCRIPTION': description,
                'FORMAT': file_format,
                'TRANSCRIPT': processed_text,
                'SENTIMENT': sentiment_score}

    # Load existing data
    df = pd.read_csv('./data/course_processed.csv')
    # Append new data
    df = df.append(new_data, ignore_index=True)
    # Save to CSV
    df.to_csv('./data/course_processed.csv', index=False)

    # Compute similarity matrix
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['TITLE'] + ' ' + df['DESCRIPTION'])  # Concatenate title and description

    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Save similarity matrix to CSV
    np.savetxt("./data/sim.csv", cosine_similarities, delimiter=",")

    # Calculate genres for the new course
    new_genres = calculate_genres(course_id, title, description)

    # Load existing genres data
    df_genres = pd.read_csv('./data/course_genres.csv')
    # Append new genres data
    df_genres = df_genres.append({**{'COURSE_ID': course_id, 'TITLE': title}, **new_genres}, ignore_index=True)
    # Save to CSV
    df_genres.to_csv('./data/course_genres.csv', index=False)

    # Update the bag of words
    update_bow(course_id, processed_text)


def main():
    st.title('Course Information Form')
    # Get user inputs
    course_id = st.text_input('Course ID')
    title = st.text_input('Title')
    description = st.text_input('description')
    file_format = st.selectbox('Format', options=['VIDEO', 'AUDIO', 'TEXT'])

    # Get transcript if the format is video or audio, get content if the format is text
    if file_format in ['VIDEO', 'AUDIO']:
        transcript = st.text_area('Transcript')
    else:
        transcript = st.text_area('Content')

    # On form submission, add the new data to the CSV
    if st.button('Submit'):
        add_course(course_id, title, description, file_format, transcript)
        st.success('Successfully added new course data!')


if __name__ == '__main__':
    main()
