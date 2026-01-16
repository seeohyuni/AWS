import streamlit as st
import requests
from PIL import Image
import io

# --- Configuration ---
# CHANGE THIS TO YOUR AI SERVER IP
AI_SERVER_URL = "http://54.212.247.253:8000/segment"

st.set_page_config(page_title="SAM2 Web Client", layout="wide")
st.title("☁️ SAM2 Cloud Segmentation")
st.caption("Web Server (Streamlit) -> AI Server (FastAPI)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Original
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption="Original", width=400)

    if st.button("🚀 Send to AI Server"):
        with st.spinner("Connecting to AI Server..."):
            try:
                files = {"file": ("image.png", image_bytes, "image/png")}
                response = requests.post(AI_SERVER_URL, files=files, timeout=120)
                
                if response.status_code == 200:
                    result_image = Image.open(io.BytesIO(response.content))
                    st.image(result_image, caption="Result from AI Server", width=400)
                else:
                    st.error(f"Server Error: {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"Connection Failed: {e}")
