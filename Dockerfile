# Multi-stage build for efficient image size
FROM python:3.9-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/app/.local

# Copy application code
COPY main.py .
COPY client_example.py .

# Create directories for temporary files and output
RUN mkdir -p /tmp/audio_processing /app/output && \
    chown -R app:app /app /tmp/audio_processing

# Switch to non-root user
USER app

# Add local Python packages to PATH
ENV PATH=/home/app/.local/bin:$PATH
ENV PYTHONPATH=/home/app/.local/lib/python3.9/site-packages:$PYTHONPATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Docker build and run instructions:
# 
# Build the image:
# docker build -t speaker-diarization-api .
#
# Run with environment variables:
# docker run -p 8000:8000 \
#   -e HUGGINGFACE_HUB_TOKEN=your_token_here \
#   -v $(pwd)/audio_files:/app/audio_files \
#   -v $(pwd)/output:/app/output \
#   speaker-diarization-api
#
# For GPU support, add: --gpus all
# docker run --gpus all -p 8000:8000 \
#   -e HUGGINGFACE_HUB_TOKEN=your_token_here \
#   speaker-diarization-api
