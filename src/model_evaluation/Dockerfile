# Using python 3.9 full image as the base image
FROM python:3.9-slim
# Defining working directory and copy the requirements file. We will run the commands inside this new directory
WORKDIR .
# Copy requirements.txt  to the working directory
COPY requirements.txt .
# Install required python packages
RUN pip install --no-cache-dir -r requirements.txt
# Copy all files in model evaluation local host directory to /usr/src/myapp in Docker container
COPY . .
# Expose the port that our app runs in
EXPOSE 7274
# Environment Variables
ENV MODEL_REPO=models
ENV TESTDB_PREPROCESSED=http://preprocessing:7271/preprocessing/test
# Run our App
CMD ["python3", "model_evaluation.py"]
