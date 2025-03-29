#!/bin/bash

# HR Analytics Dashboard Start Script
echo "Starting HR Analytics Dashboard..."

# Activate virtual environment
source .venv/bin/activate

# Check if the virtual environment was activated successfully
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment. Please check if it exists."
    exit 1
fi

# Set environment variables if needed
export PYTHONPATH=$(pwd)
export FLASK_ENV=production

# Start the application
echo "Launching dashboard application..."
python dashboard/app.py &

# Save the process ID to a file
echo $! > app.pid

echo "Dashboard started with PID $(cat app.pid)"
echo "You can access the dashboard at http://127.0.0.1:8050/"