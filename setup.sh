#!/bin/bash
# Setup script for Legal Document Summarizer

echo "Setting up Legal Document Summarizer..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Please install from https://ollama.ai"
    exit 1
fi

# Pull required models
echo "Pulling required Ollama models..."
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# Install Python requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Setup complete! Run with: streamlit run app.py"
