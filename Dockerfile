FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system packages needed for your app
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    libjpeg62-turbo-dev \
    libpng-dev \
    libpoppler-cpp-dev \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first (this helps with caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Create necessary directories
RUN mkdir -p logs static

# Copy the HTML frontend to static directory
COPY static/ ./static/

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Tell Docker which port your app uses
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run your FastAPI application
CMD ["python", "-m", "uvicorn", "main2:app", "--host", "0.0.0.0", "--port", "8000"]