# syntax=docker/dockerfile:1.4
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (cached layer)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    postgresql-client

# Copy only requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies with cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy application code (this layer changes most often)
COPY src/ ./src/
COPY entrypoint.sh .

# Make entrypoint.sh executable
RUN chmod +x entrypoint.sh

# Expose the port the app runs on
EXPOSE 8000

# Entry point
ENTRYPOINT ["./entrypoint.sh"]
