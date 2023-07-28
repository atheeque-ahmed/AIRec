# Pull base image
FROM python:3.8

# Set work directory in the container
WORKDIR /app

# Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION to python
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Copy requirements.txt and install dependencies, excluding tensorflow
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port
EXPOSE 8501

# CMD streamlit run app.py
CMD streamlit run --server.port $PORT Recommender_Engine.py
