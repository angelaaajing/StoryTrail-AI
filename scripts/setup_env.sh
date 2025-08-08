#!/bin/bash

echo "Creating and activating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Upgrading pip and installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Starting Redis server with Docker..."
docker run -d -p 6379:6379 --name redis-server redis:7-alpine

echo "Setup complete! Use 'source venv/bin/activate' to activate environment."
