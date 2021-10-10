# Python version 3.9
FROM python:3.9-slim
# Set the working directory
WORKDIR .
# Copy requirements.txt  to the working directory
COPY requirements.txt .
# Install all requirements
RUN pip install --no-cache-dir -r requirements.txt
# Copy all files to the working directory
COPY . .
# Expose the port that our app runs in
EXPOSE 7274
# Define the necessary environment variables
ENV MODEL_REPO=models
ENV TESTDB_PREPROCESSED=http://preprocessing:7271/preprocessing/test
# Run the application
CMD ["python3", "model_evaluation.py"]
