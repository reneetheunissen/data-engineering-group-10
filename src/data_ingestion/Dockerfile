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
EXPOSE 7270
# Define the necessary environment variables
ENV DB_URL=sqlite:///data.db
# Run the application
CMD ["python3", "data_ingestion.py"]
