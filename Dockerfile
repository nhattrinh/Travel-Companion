# Multi-stage Dockerfile for Menu Translation + Travel Companion Backend
# Optimized for production deployment with minimal image size

# Stage 1: Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG ENVIRONMENT=production

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /build

# Copy base requirements first for caching
COPY requirements.txt .
# Copy travel companion extra requirements
COPY requirements-travel.txt ./

# Install Python dependencies (base + optional travel companion extras)
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ -f requirements-travel.txt ]; then pip install --no-cache-dir -r requirements-travel.txt; fi

# Copy application code
COPY app/ ./app/
COPY .env.* ./
COPY run.py .

# Stage 2: Runtime stage
FROM python:3.11-slim as runtime

# Set runtime arguments
ARG ENVIRONMENT=production
ARG APP_VERSION=1.0.0

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=${ENVIRONMENT} \
    APP_VERSION=${APP_VERSION} \
    PATH="/app/.local/bin:$PATH"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    # Image processing & OCR libraries
    libjpeg62-turbo \
    libpng16-16 \
    libwebp7 \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    # Networking and SSL
    ca-certificates \
    curl \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create application directories
RUN mkdir -p /app/logs /app/temp /app/models && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code from builder stage
COPY --from=builder --chown=appuser:appuser /build/app/ ./app/
COPY --from=builder --chown=appuser:appuser /build/requirements-travel.txt* ./
COPY --from=builder --chown=appuser:appuser /build/.env.* ./
COPY --from=builder --chown=appuser:appuser /build/run.py ./

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "run.py", "--env", "production"]

# Stage 3: Development stage (for development builds)
FROM runtime as development

# Switch back to root for development tools installation
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipython \
    debugpy

# Switch back to appuser
USER appuser

# Override environment for development
ENV ENVIRONMENT=development

# Development command with reload
CMD ["python", "run.py", "--env", "development", "--reload"]

# Stage 4: Testing stage
FROM development as testing

# Override environment for testing
ENV ENVIRONMENT=testing

# Copy test files
COPY --chown=appuser:appuser test_*.py ./

# Testing command
CMD ["python", "-m", "pytest", "-v", "--cov=app"]