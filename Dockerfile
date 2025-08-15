# Flight Delay Analysis - Docker Container
FROM python:3.12-slim

# Set metadata
LABEL maintainer="Flight Delay Analysis Team"
LABEL description="Containerized flight delay prediction and analysis application"
LABEL version="1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn plotly streamlit requests python-dotenv imbalanced-learn jupyter ipykernel dask folium

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed logs

# Set proper permissions
RUN chmod +x run_exploration.py explore_data.py

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose ports (for Streamlit and Jupyter)
EXPOSE 8501 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pandas, numpy, matplotlib, seaborn, streamlit" || exit 1

# Default command - can be overridden
CMD ["python", "run_exploration.py"]