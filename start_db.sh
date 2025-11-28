#!/bin/bash

# Builds and starts the PostgreSQL database container defined in docker-compose.yml

echo "Starting PostgreSQL database using Docker Compose..."
docker compose up -d

if [ $? -eq 0 ]; then
    echo "✅ Database 'ml_eval_postgres' started successfully on port 5432."
    echo "Check status with: docker ps"
else
    echo "❌ ERROR: Failed to start the database container."
fi