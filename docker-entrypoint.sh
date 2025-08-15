#!/bin/bash
set -e

# Docker entrypoint script for flight delay analysis

echo "Starting Flight Delay Analysis Container..."
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"

# Create directories if they don't exist
mkdir -p data/raw data/processed logs

# Check if data exists, if not download it
if [ ! -f "data/processed/airline_exploration.csv" ]; then
    echo "No processed data found. Running data download and exploration..."
    python run_exploration.py
    python explore_data.py
    echo "Data processing completed."
else
    echo "Processed data found. Skipping download."
fi

# Execute the main command
exec "$@"