import os
# Fix for OMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import io

# --- Configuration ---
CHECKPOINT_PATH = "../sam2/checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

st.set_page_config(page_title="SAM2 Auto Cutout", layout="wide")

@st.cache_resource
def load_model():
    """Load model once and cache it."""
    if not os.path.exists(CHECKPOINT_PATH):
        st.error(f"Checkpoint not found at: {CHECKPOINT_PATH}")
        return None
    
    try:
        # Force CPU for now
        sam2_model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device="cpu")
        predictor = SAM2ImagePredictor(sam2_model)
        return predictor
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def process_sam2(predictor, image_pil):
    """Run SAM2 inference and return cutout."""
    # Convert to RGB numpy for SAM2
    image_np = np.array(image_pil.convert("RGB"))
    predictor.set_image(image_np)
    
    # Simple Center Point Prompt
    h, w = image_np.shape[:2]
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])
    
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    
    # Create Cutout (Alpha Channel)
    best_mask = masks[0] 
    alpha_mask = (best_mask * 255).astype(np.uint8)
    alpha_image = Image.fromarray(alpha_mask, mode='L')
    
    # Apply to original (converted to RGBA)
    result_pil = image_pil.convert("RGBA")
    result_pil.putalpha(alpha_image)
    
    return result_pil, scores[0]

# --- UI Layout ---
st.title("✂️ SAM2 Auto Cutout Web")
st.markdown("Upload an image, and SAM2 will automatically remove the background.")

# Sidebar for controls or info
with st.sidebar:
    st.info("System Status: Local (CPU)")
    st.text(f"Model: {os.path.basename(CHECKPOINT_PATH)}")

# Main Area
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Original
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_column_width=True)

    # Load button
    if st.button("🚀 Run Segmentation (누끼 따기)", type="primary"):
        with st.spinner("Processing... (This may take a moment on CPU)"):
            predictor = load_model()
            if predictor:
                result_image, score = process_sam2(predictor, image)
                
                with col2:
                    st.subheader("Result")
                    st.image(result_image, use_column_width=True)
                    st.success(f"Segmentation Score: {score:.2f}")
                
                # Download Button
                buf = io.BytesIO()
                result_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Cutout PNG",
                    data=byte_im,
                    file_name="cutout_result.png",
                    mime="image/png"
                )
