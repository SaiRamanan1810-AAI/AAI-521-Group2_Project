#!/usr/bin/env python3
"""
Streamlit web application for Multi-Crop Disease Classification.
Two-stage classification: Plant Species → Disease Detection
"""
import os
import sys

# Fix macOS OpenMP conflict - MUST be before any other imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import streamlit as st
from PIL import Image
import torch
import numpy as np
from io import BytesIO

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference import InferencePipeline

# Page configuration
st.set_page_config(
    page_title="Multi-Crop Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        font-size: 18px;
        border-radius: 10px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(plant_checkpoint, models_dir, device, threshold):
    """Load the inference pipeline (cached)."""
    try:
        import json
        
        # Load plant species names
        plant_meta_path = plant_checkpoint + '.meta.json'
        if os.path.exists(plant_meta_path):
            with open(plant_meta_path, 'r') as f:
                plant_meta = json.load(f)
                species_names = plant_meta.get('species', ['Cashew', 'Cassava', 'Maize', 'Tomato'])
        else:
            species_names = ['Cashew', 'Cassava', 'Maize', 'Tomato']
        
        # Build disease checkpoints dictionary and load disease names
        # Note: InferencePipeline expects keys as string indices "0", "1", "2", "3"
        disease_checkpoints = {}
        disease_names_map = {}
        
        for idx, sp in enumerate(species_names):
            ckpt_path = os.path.join(models_dir, f'{sp}_checkpoint.pth')
            if os.path.exists(ckpt_path):
                # Use string index as key (matching InferencePipeline's expectation)
                disease_checkpoints[str(idx)] = ckpt_path
                
                # Load disease class names (map by species name for display)
                meta_path = ckpt_path + '.meta.json'
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                        disease_names_map[sp] = meta.get('classes', [])
        
        pipeline = InferencePipeline(
            plant_checkpoint=plant_checkpoint,
            disease_checkpoints=disease_checkpoints,
            device=device,
            threshold=threshold
        )
        return pipeline, species_names, disease_names_map, None
    except Exception as e:
        return None, None, None, str(e)


def format_confidence(confidence):
    """Format confidence as percentage with color."""
    pct = confidence * 100
    if pct >= 80:
        color = "#28a745"  # green
    elif pct >= 60:
        color = "#ffc107"  # yellow
    else:
        color = "#dc3545"  # red
    return f'<span style="color: {color}; font-weight: bold;">{pct:.1f}%</span>'


def main():
    # Header
    st.title("🌿 Multi-Crop Disease Classifier")
    st.markdown("""
    ### Two-Stage Plant Disease Detection System
    Upload an image of a plant leaf to identify the species and detect diseases.
    """)
    
    # Sidebar - Configuration
    st.sidebar.header("⚙️ Configuration")
    
    plant_checkpoint = st.sidebar.text_input(
        "Plant Model Checkpoint",
        value="models/plant_checkpoint.pth",
        help="Path to the plant species classifier model"
    )
    
    models_dir = st.sidebar.text_input(
        "Disease Models Directory",
        value="models",
        help="Directory containing disease classifier models"
    )
    
    device_option = st.sidebar.selectbox(
        "Device",
        options=["cpu", "cuda", "mps"],
        index=0,
        help="Select computation device"
    )
    
    threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for disease classification"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### How it works:
    1. **Stage 1**: Identifies plant species (Cashew, Cassava, Maize, or Tomato)
    2. **Stage 2**: Routes to species-specific disease classifier
    3. **Result**: Shows plant species and disease with confidence scores
    """)
    
    # Load model
    with st.spinner("Loading models..."):
        pipeline, species_names, disease_names_map, error = load_model(
            plant_checkpoint, models_dir, device_option, threshold
        )
    
    if error:
        st.error(f"❌ Error loading models: {error}")
        st.info("Please ensure model checkpoints are available in the specified paths.")
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a plant leaf image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload a clear image of a plant leaf"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.caption(f"Image size: {image.size[0]} × {image.size[1]} pixels")
    
    with col2:
        st.subheader("🔍 Classification Results")
        
        if uploaded_file is not None:
            # Run inference button
            if st.button("🚀 Run Classification", key="classify_btn"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Save uploaded file temporarily
                        temp_path = "temp_upload.jpg"
                        image.save(temp_path)
                        
                        # Run inference
                        result = pipeline.predict(temp_path)
                        
                        # Clean up
                        os.remove(temp_path)
                        
                        # Get predictions with human-readable names
                        plant_idx = result.get('plant_prediction', 0)
                        species = species_names[plant_idx] if plant_idx < len(species_names) else f'Unknown ({plant_idx})'
                        species_conf = result.get('plant_confidence', 0.0)
                        
                        disease_idx = result.get('disease_prediction')
                        disease_conf = result.get('disease_confidence', 0.0)
                        disease = 'N/A'
                        if disease_idx is not None and species in disease_names_map:
                            disease_classes = disease_names_map[species]
                            disease = disease_classes[disease_idx] if disease_idx < len(disease_classes) else f'Unknown ({disease_idx})'
                        
                        above_threshold = species_conf >= threshold
                        note = result.get('note', '')
                        
                        # Display results
                        st.markdown("---")
                        
                        # Stage 1 - Plant Species
                        st.markdown("### 🌱 Stage 1: Plant Species")
                        
                        st.markdown(f"""
                        <div class="result-box info-box">
                            <h3 style="margin: 0; color: #17a2b8;">Species: {species}</h3>
                            <p style="margin: 5px 0 0 0; font-size: 18px;">
                                Confidence: {format_confidence(species_conf)}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Progress bar for species confidence
                        st.progress(species_conf)
                        
                        st.markdown("---")
                        
                        # Stage 2 - Disease Detection
                        st.markdown("### 🦠 Stage 2: Disease Detection")
                        
                        if above_threshold and disease_idx is not None:
                            
                            st.markdown(f"""
                            <div class="result-box success-box">
                                <h3 style="margin: 0; color: #28a745;">Disease: {disease}</h3>
                                <p style="margin: 5px 0 0 0; font-size: 18px;">
                                    Confidence: {format_confidence(disease_conf)}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Progress bar for disease confidence
                            st.progress(disease_conf)
                            
                        else:
                            reason = note if note else f"Species confidence ({format_confidence(species_conf)}) is below threshold ({threshold * 100:.0f}%)"
                            st.markdown(f"""
                            <div class="result-box warning-box">
                                <h3 style="margin: 0; color: #856404;">⚠️ Low Confidence</h3>
                                <p style="margin: 5px 0 0 0;">
                                    {reason}
                                </p>
                                <p style="margin: 5px 0 0 0; font-size: 14px;">
                                    Disease classification skipped. Try uploading a clearer image.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Summary
                        st.markdown("---")
                        st.markdown("### 📊 Summary")
                        
                        summary_data = {
                            "Metric": ["Plant Species", "Species Confidence", "Disease", "Disease Confidence", "Above Threshold"],
                            "Value": [
                                species,
                                f"{species_conf * 100:.1f}%",
                                disease if disease_idx is not None else 'N/A',
                                f"{disease_conf * 100:.1f}%" if disease_idx is not None else "N/A",
                                "✓ Yes" if above_threshold else "✗ No"
                            ]
                        }
                        
                        st.table(summary_data)
                        
                        # Download results as JSON
                        import json
                        download_result = {
                            'plant_species': species,
                            'plant_confidence': float(species_conf),
                            'disease': disease if disease_idx is not None else None,
                            'disease_confidence': float(disease_conf) if disease_idx is not None else None,
                            'above_threshold': above_threshold,
                            'threshold_used': threshold,
                            'raw_result': result
                        }
                        json_str = json.dumps(download_result, indent=2)
                        st.download_button(
                            label="📥 Download Results (JSON)",
                            data=json_str,
                            file_name="classification_result.json",
                            mime="application/json"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error during classification: {str(e)}")
                        st.info("Please check the image and try again.")
        else:
            st.info("👆 Upload an image to start classification")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>Multi-Crop Disease Classifier (CCMT)</strong></p>
        <p>Two-stage classification using EfficientNet-B0</p>
        <p>Species: Cashew, Cassava, Maize, Tomato</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Set OpenMP environment variable to avoid conflicts on macOS
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()
