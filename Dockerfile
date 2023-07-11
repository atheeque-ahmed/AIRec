# Pull base image
FROM tensorflow/tensorflow:2.10.0

# Set work directory in the container
WORKDIR /app

# Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION to python
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Copy requirements.txt and install dependencies, excluding tensorflow
COPY requirements.txt .
RUN sed -i '/tensorflow/d' requirements.txt && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "recommender_app.py"]
