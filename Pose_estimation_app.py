import streamlit as st  # type: ignore
from PIL import Image
import numpy as np
import cv2

DEMO_IMAGE = "stand.jpg"

BODY_PARTS = { 
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

width, height = 368, 368
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# Main App Design
st.markdown(
    """
    <style>
    body {
        background-color: #f8f9fa;
        color: #212529;
    }
    .main-header {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
    }
    .sub-header {
        font-size: 18px;
        font-weight: normal;
        text-align: center;
        color: #6c757d;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🎨 Creative Human Pose Estimation App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload an image and visualize estimated poses in a sleek design</div>', unsafe_allow_html=True)

# Tabs for Navigation
tab1, tab2, tab3 = st.tabs(["📂 Upload & Settings", "🕺 Pose Estimation", "ℹ️ About"])

with tab1:
    st.markdown('<div class="upload-box">Upload an image with clearly visible body parts for better pose detection.</div>', unsafe_allow_html=True)

    # File Upload
    img_file_buffer = st.file_uploader("Drag & drop an image, or click to browse.", type=["jpg", "jpeg", "png"])
    thres = st.slider(
        "Detection Threshold",
        min_value=0,
        value=20,
        max_value=100,
        step=5
    ) / 100

    st.info("ℹ️ Adjust the threshold for better pose estimation.")

    # Default Image Handling
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
    else:
        st.warning("No image uploaded! Using demo image.")
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.subheader("Preview: Uploaded Image")
    st.image(image, caption="Your Uploaded Image", use_container_width=True)

@st.cache_resource
def poseDetector(frame):
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()[:, :19, :, :]
    points = []

    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x, y = (frameWidth * point[0]) / out.shape[3], (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thres else None)

    for pair in POSE_PAIRS:
        partFrom, partTo = pair[0], pair[1]
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    return frame

with tab2:
    st.markdown('<div class="sub-header">Analyzing the Image...</div>', unsafe_allow_html=True)
    
    with st.spinner("🔍 Processing... This might take a moment."):
        output = poseDetector(image)
    
    st.success("✅ Pose estimation complete! See the results below:")
    st.image(output, caption="Pose Estimated", use_container_width=True)

with tab3:
    st.header("About the App")
    st.write("""
        This Human Pose Estimation app uses OpenCV's deep learning model to detect key points on the human body.
        - **Input:** Upload an image with visible body parts.
        - **Output:** The app highlights joints and connections.
        - **Tools:** OpenCV, TensorFlow, Streamlit.
    """)
    st.markdown("### How to Use:")
    st.markdown("""
    1. Navigate to the **Upload & Settings** tab.
    2. Upload an image and adjust the detection threshold.
    3. View the results in the **Pose Estimation** tab.
    """)
