# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code to the working directory
COPY . .

# Expose the port that the app will run on. Hugging Face uses 7860 by default.
EXPOSE 7860

# Define the command to run your app. We use uvicorn directly.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]