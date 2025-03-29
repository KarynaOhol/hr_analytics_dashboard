#!/bin/bash

# HR Analytics Dashboard Stop Script
echo "Stopping HR Analytics Dashboard..."

# Check if PID file exists
if [ ! -f app.pid ]; then
    echo "PID file not found. The application may not be running."
    exit 1
fi

# Read the PID
PID=$(cat app.pid)

# Check if the process is still running
if ps -p $PID > /dev/null; then
    echo "Stopping application process (PID: $PID)..."
    kill $PID

    # Wait for the process to terminate
    sleep 2

    # Check if the process was terminated
    if ps -p $PID > /dev/null; then
        echo "Process did not terminate gracefully. Forcing termination..."
        kill -9 $PID
    fi

    echo "Application stopped successfully."
else
    echo "Process (PID: $PID) not found. The application may have already been stopped."
fi

# Remove the PID file
rm app.pid