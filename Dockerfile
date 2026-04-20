FROM python:3.11-slim

# Install system dependencies for OpenCV and rembg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Set PYTHONPATH for package imports
ENV PYTHONPATH=/app/src

# Create data directory
RUN mkdir -p /app/data

# Expose API port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "scanario.api:app", "--host", "0.0.0.0", "--port", "8000"]
