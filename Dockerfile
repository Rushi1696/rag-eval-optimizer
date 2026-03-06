# Simple Dockerfile for RAG API
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Expose port
EXPOSE 8000

# Run the simple API
CMD ["python", "-m", "uvicorn", "app.simple_api:app", "--host", "0.0.0.0", "--port", "8000"]