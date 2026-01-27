#!/bin/bash
set -e

# Wait for database to be ready
echo "Waiting for database..."
# Simple wait-for-it equivalent
until pg_isready -h "$DATABASE_HOST" -p "$DATABASE_PORT" -U "$DATABASE_USER"; do
  echo "Database is unavailable - sleeping"
  sleep 1
done

echo "Database is up - executing migrations"

# Run alembic migrations
alembic upgrade head

# Start the application
echo "Starting application..."
exec uvicorn src.main:app --host 0.0.0.0 --port 8000
