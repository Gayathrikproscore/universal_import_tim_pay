# Simple Dockerfile for Document Processing API
# This creates a container that runs your FastAPI application

# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system packages needed for your app
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    libjpeg62-turbo-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first (this helps with caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Create a directory for logs
RUN mkdir -p logs

# Tell Docker which port your app uses
EXPOSE 8000

# Run your FastAPI application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]