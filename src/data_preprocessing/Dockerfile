# Using python 3.9 full image as the base image
FROM python:3.9-slim
# Defining working directory and copy the requirements file. We will run the commands inside this new directory
WORKDIR .
# Copy requirements.txt  to the working directory
COPY requirements.txt .
# Install required python packages
RUN pip install --no-cache-dir -r requirements.txt
# Copy all files in data preprocessing local host directory to /usr/src/myapp in Docker container
COPY . .
# Expose the port that our app runs in
EXPOSE 7271
# Environment Variables
ENV DB_URL=sqlite:///data.db
ENV TRAININGDB_API=http://ingestion:7270/data/train
ENV TESTDB_API=http://ingestion:7270/data/test
ENV TRUTHDB_API=http://ingestion:7270/data/truth
# Run our App
CMD ["python3", "data_preprocessing.py"]
