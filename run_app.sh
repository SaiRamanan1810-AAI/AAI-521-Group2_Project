#!/bin/bash
# Quick start script for the Streamlit web application

echo "🌿 Starting Multi-Crop Disease Classifier Web App..."
echo ""
echo "The app will open in your browser automatically."
echo "If it doesn't, navigate to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

# Check if models exist
if [ ! -f "models/plant_checkpoint.pth" ]; then
    echo "⚠️  Warning: Plant model checkpoint not found at models/plant_checkpoint.pth"
    echo "   Please train the models first or place checkpoints in the models/ directory."
    echo ""
fi

# Set environment variable to fix macOS OpenMP issue
export KMP_DUPLICATE_LIB_OK=True

# Run streamlit app
conda run -n plant-pest streamlit run app.py
