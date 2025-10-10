# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the main working directory in the container
WORKDIR /app

# Copy ONLY the requirements file first for better layer caching
COPY api/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code and your models into the container
COPY api/ .
COPY Final-Models/ ./Final-Models/

# Expose the port that the app will run on
EXPOSE 7860

# Define the command to run your app
# It knows to find main.py because we copied the 'api' contents into the current directory
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]