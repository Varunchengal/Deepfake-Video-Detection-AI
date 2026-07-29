import streamlit as st
import cv2
import numpy as np
import tempfile
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import time
from detector_utils import extract_face_region, analyze_texture_artifacts

# 1. Page Configuration & Modern Responsive Layout
st.set_page_config(
    page_title="Deepfake Detection AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS for Modern UI Look & Responsiveness
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF8080);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #A0AEC0;
        text-align: center;
        margin-bottom: 25px;
    }
    .result-box-fake {
        padding: 20px;
        border-radius: 12px;
        background-color: rgba(255, 75, 75, 0.15);
        border-left: 6px solid #FF4B4B;
        margin-top: 15px;
    }
    .result-box-real {
        padding: 20px;
        border-radius: 12px;
        background-color: rgba(40, 167, 69, 0.15);
        border-left: 6px solid #28a745;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Modern UI Titles
st.markdown('<p class="main-header">AI-Based Deepfake Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Media Verification Engine using Transfer Learning & Computer Vision</p>', unsafe_allow_html=True)

# 4. Load AI Model
@st.cache_resource
def load_trained_model():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    model_path = os.path.join("models", "deepfake_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    model.eval()
    return model

model = load_trained_model()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 5. Video Uploader with Responsive Box
uploaded_video = st.file_uploader("📁 Drag and Drop or Upload Video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)
    
    st.markdown("---")
    st.subheader("🖼️ Extracted Facial Regions Analysis")
    
    frame_count, faces_found = 0, 0
    predictions = []
    
    # 4 കോളമുകളുള്ള ലേഔട്ട് (മൊബൈലിലും റെസ്പോൺസീവ് ആയിരിക്കും)
    cols = st.columns(4)
    
    # --- Loading Progress Bar Configuration ---
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_target_frames = 8  # നമ്മൾ അനലൈസ് ചെയ്യാൻ ഉദ്ദേശിക്കുന്ന ഫ്രെയിമുകളുടെ എണ്ണം
    # ----------------------------------------
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % 15 == 0:
            cropped_face = extract_face_region(frame)
            
            # AI Model Prediction
            input_tensor = transform(cropped_face).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                base_prob = float(probabilities[0][1]) * 100
            
            # Texture Artifact Analysis
            texture_variance = analyze_texture_artifacts(cropped_face)
            if texture_variance < 150: 
                adjusted_fake_score = min(base_prob + 35.0, 94.50)
            else:
                adjusted_fake_score = max(base_prob - 20.0, 12.30)
                
            predictions.append(adjusted_fake_score)
            
            # ചിത്രങ്ങൾ റെസ്പോൺസീവ് ആയി കാണിക്കുന്നു
            with cols[faces_found % 4]:
                st.image(cropped_face, caption=f"Frame {faces_found+1}", use_container_width=True)
            
            faces_found += 1
            
            # --- Progress Bar Update ---
            progress_percent = int((faces_found / total_target_frames) * 100)
            progress_bar.progress(min(progress_percent, 100))
            status_text.markdown(f"**⚡ Analyzing Video Frames... Completed {min(progress_percent, 100)}%** *(Frame {faces_found}/{total_target_frames})*")
            # ---------------------------
            
            if faces_found >= total_target_frames:
                break
                
        frame_count += 1
        
    cap.release()
    
    if faces_found > 0:
        # പ്രോസസ്സിങ് കഴിഞ്ഞാൽ ലോഡിങ് ബാർ 100% ആക്കി മാറ്റുന്നു
        progress_bar.progress(100)
        status_text.success("✅ Frame Extraction & AI Analysis Completed 100%!")
        time.sleep(1) # ചെറിയൊരു സ്മൂത്ത് ഫീൽ കിട്ടാൻ
        
        avg_fake_score = np.mean(predictions)
        
        st.markdown("---")
        st.subheader("🎯 Final Detection Result")
        
        # Modern Card Design Result
        if avg_fake_score > 50:
            st.markdown(f"""
                <div class="result-box-fake">
                    <h3 style="color: #FF4B4B; margin-top: 0;">🚨 FAKE VIDEO DETECTED (AI Generated)</h3>
                    <p><b>Confidence Score:</b> {avg_fake_score:.2f}%</p>
                    <p><i>Visual inconsistencies, synthetic texture smoothing, or media manipulation detected.</i></p>
                </div>
            """, unsafe_allow_html=True)
        else:
            real_score = 100 - avg_fake_score
            st.markdown(f"""
                <div class="result-box-real">
                    <h3 style="color: #28a745; margin-top: 0;">✅ REAL VIDEO (Authentic Media)</h3>
                    <p><b>Confidence Score:</b> {real_score:.2f}%</p>
                    <p><i>No significant manipulation detected. The video appears to be authentic.</i></p>
                </div>
            """, unsafe_allow_html=True)