#!/bin/bash

echo "========================================"
echo " PDF Question Answering System - Demo"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
else
    PYTHON_CMD="python"
    PIP_CMD="pip"
fi

echo "Python version:"
$PYTHON_CMD --version
echo

# Check if pip is available
if ! command -v $PIP_CMD &> /dev/null; then
    echo "ERROR: pip is not installed"
    exit 1
fi

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    $PIP_CMD install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
    echo
else
    echo "WARNING: requirements.txt not found"
    echo
fi

# Create necessary directories
mkdir -p uploads
mkdir -p vector_db
mkdir -p demo_data/sample_pdfs

echo "Starting PDF Question Answering System..."
echo
echo "The application will be available at:"
echo "  http://localhost:8000"
echo
echo "To stop the application, press Ctrl+C"
echo

# Start the application
$PYTHON_CMD main.py

# If we get here, the application has stopped
echo
echo "Application stopped."