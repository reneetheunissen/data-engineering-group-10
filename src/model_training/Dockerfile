# Using python 3.9 full image as the base image
FROM python:3.9-slim
# Defining working directory and copy the requirements file. We will run the commands inside this new directory
WORKDIR /usr/src/myapp
# Copy requirements.txt  to the working directory
COPY requirements.txt .
# Install required python packages
RUN pip install --no-cache-dir -r requirements.txt
# Copy all files in model training local host directory to /usr/src/myapp in Docker container
COPY . .
# Expose the port that our app runs in
EXPOSE 5000
# Environment Variables
ENV MODEL_REPO=/usr/src/myapp/models
ENV TRAININGDB_PREPROCESSED=http://training-db:5000/data/train_preprocessed
# Run our App
CMD ["python3", "model_training.py"]
