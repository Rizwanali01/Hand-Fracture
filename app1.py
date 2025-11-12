import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
import io
import pandas as pd

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="Hand Fracture Detection",
    page_icon="🩹",
    layout="wide",
    initial_sidebar_state="expanded"
)
page_bg = """
<style>
/* Set a soothing medical-style background */
[data-testid="stAppViewContainer"] {
    background-color: #2c2c2c;  /* Light blue background */
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Main content area with transparent overlay */
.main {
    background-color: rgba(255, 255, 255, 0.9); /* Transparent white overlay */
    border-radius: 15px;
    padding: 2rem;
    margin: 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}

/* Title */
h1 {
    color: #007BFF; /* Medical blue */
    text-align: center;
    padding-bottom: 1rem;
    font-weight: 700;
}

/* Buttons */
.stButton>button {
    width: 100%;
    background-color: #007BFF;
    color: white;
    font-weight: bold;
    padding: 0.6rem;
    border-radius: 0.5rem;
    transition: 0.3s ease;
    border: none;
}

.stButton>button:hover {
    background-color: #339CFF;
    transform: scale(1.02);
}

/* Upload text */
.upload-text {
    text-align: center;
    color: #444;
    font-size: 1.1rem;
}

/* Footer */
.footer {
    text-align: center;
    color: #555;
    font-size: 0.9rem;
    margin-top: 2rem;
    padding-bottom: 1rem;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------- Session Initialization --------------------
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None

# -------------------- Title and Description --------------------
st.title("🩹 Hand Fracture Detection System")
st.markdown("<p class='upload-text'>Upload an X-ray image to detect and locate hand fractures using AI</p>",
            unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.subheader("ℹ️ About")
    st.info("""
    This app uses a YOLOv8 model to detect hand fractures in X-ray images.

    **Steps to use:**
    1. Adjust detection confidence  
    2. Upload an X-ray image  
    3. Click 'Detect Fractures'
    """)

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for displaying detections"
    )


# -------------------- Model Loader --------------------
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


if not st.session_state.model_loaded:
    model_path = "best.pt"
    if os.path.exists(model_path):
        with st.spinner("Loading YOLO model..."):
            st.session_state.model = load_model(model_path)
            if st.session_state.model is not None:
                st.session_state.model_loaded = True
    else:
        st.error("⚠️ Model file 'best.pt' not found. Please ensure it's in the same directory.")

# -------------------- Main Content --------------------
if not st.session_state.model_loaded:
    st.warning("⚠️ Please upload your model file (best.pt) to begin.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Upload X-ray Image")
        uploaded_file = st.file_uploader(
            "Choose an X-ray image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a hand X-ray image for fracture detection"
        )

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Original X-ray Image", use_container_width=True)
                st.caption(f"Image size: {image.size}, Format: {image.format}")

                if st.button("🔍 Detect Fractures", type="primary"):
                    with st.spinner("Analyzing image..."):
                        img_array = np.array(image)
                        results = st.session_state.model.predict(img_array, verbose=False)
                        st.session_state.results = results
                        st.session_state.detection_done = True

            except Exception as e:
                st.error(f"Error loading image: {e}")

    with col2:
        st.subheader("📊 Detection Results")

        if 'detection_done' in st.session_state and st.session_state.detection_done:
            results = st.session_state.results
            annotated_img = results[0].plot()
            annotated_img = annotated_img[:, :, ::-1]
            st.image(annotated_img, caption="Detected Fractures", use_container_width=True)

            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            filtered_boxes = [box for box in results[0].boxes if float(box.conf[0]) >= confidence_threshold]

            if filtered_boxes:
                st.success(f"✅ **{len(filtered_boxes)} fracture(s) detected**")

                data = []
                for box in filtered_boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = results[0].names[cls]
                    bbox = box.xyxy[0].cpu().numpy()
                    data.append({
                        "Class": class_name,
                        "Confidence": f"{conf:.2%}",
                        "X1": int(bbox[0]),
                        "Y1": int(bbox[1]),
                        "X2": int(bbox[2]),
                        "Y2": int(bbox[3])
                    })

                st.subheader("Detection Summary")
                st.dataframe(pd.DataFrame(data))
            else:
                st.info("ℹ️ No fractures detected above the confidence threshold.")
                st.write("Try lowering the threshold or using a different image.")

            st.markdown("</div>", unsafe_allow_html=True)

            annotated_pil = Image.fromarray(annotated_img)
            buf = io.BytesIO()
            annotated_pil.save(buf, format='PNG')
            byte_im = buf.getvalue()

            st.download_button(
                label="📥 Download Annotated Image",
                data=byte_im,
                file_name="fracture_detection_result.png",
                mime="image/png"
            )
        else:
            st.info("👈 Upload an image and click 'Detect Fractures' to see results")

# -------------------- Footer --------------------
st.markdown("<div class='footer'>Hand Fracture Detection System | Powered by YOLOv8 & Streamlit</div>",
            unsafe_allow_html=True)
