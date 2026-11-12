import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from transformers import pipeline as hf_pipeline

from detector_utils import extract_face_region, face_detector_available


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Deepfake AI | Media Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Keep CPU usage modest on older Intel processors.
torch.set_num_threads(2)


# --------------------------------------------------
# 2. PREMIUM UI STYLES
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(37, 99, 235, 0.13), transparent 35%),
            radial-gradient(ellipse at 90% 10%, rgba(124, 58, 237, 0.10), transparent 35%),
            #080D19;
        color: #F1F5F9;
    }

    .main .block-container {
        max-width: 1200px;
        padding: 2rem 5% 3rem;
    }

    #MainMenu,
    footer,
    header {
        visibility: hidden;
    }

    h1, h2, h3 {
        color: #F8FAFC !important;
    }

    p, label {
        color: #AAB6CA;
    }

    .hero {
        text-align: center;
        padding: 35px 15px 30px;
        margin-bottom: 22px;
        border: 1px solid rgba(148, 163, 184, 0.13);
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            rgba(30, 41, 59, 0.75),
            rgba(15, 23, 42, 0.55)
        );
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.18);
    }

    .hero-icon {
        display: inline-flex;
        width: 64px;
        height: 64px;
        justify-content: center;
        align-items: center;
        border-radius: 20px;
        font-size: 30px;
        background: linear-gradient(135deg, #2563EB, #7C3AED);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.3);
        margin-bottom: 17px;
    }

    .hero-title {
        font-size: clamp(1.7rem, 4vw, 2.7rem);
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0;
        background: linear-gradient(90deg, #FFFFFF, #93C5FD, #C4B5FD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 12px;
    }

    .hero-tag {
        display: inline-block;
        font-size: 11px;
        letter-spacing: 1.5px;
        font-weight: 700;
        color: #93C5FD;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(96, 165, 250, 0.2);
        padding: 7px 13px;
        border-radius: 30px;
        margin-bottom: 16px;
    }

    .section-heading {
        font-size: 1.25rem;
        font-weight: 750;
        color: #F1F5F9;
        margin: 25px 0 5px;
    }

    .section-description {
        font-size: 0.87rem;
        color: #8492A8;
        margin-bottom: 18px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(148, 163, 184, 0.16) !important;
        border-radius: 17px !important;
        background: rgba(17, 27, 46, 0.65) !important;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.45);
        border: 1px dashed rgba(96, 165, 250, 0.45);
        border-radius: 15px;
        padding: 15px;
    }

    div[data-testid="stFileUploader"] section {
        background: transparent;
    }

    div[data-testid="stFileUploaderDropzone"] {
        border: none !important;
        background: transparent !important;
    }

    .stButton > button {
        width: 100%;
        min-height: 48px;
        border: none;
        border-radius: 11px;
        font-weight: 700;
        color: white;
        background: linear-gradient(110deg, #2563EB, #7C3AED);
        transition: all 0.25s ease;
    }

    .stButton > button:hover:not(:disabled) {
        border: none;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
    }

    .stButton > button:disabled {
        opacity: 0.45;
        cursor: not-allowed;
    }

    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.13);
        padding: 17px;
        border-radius: 13px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94A3B8;
    }

    div[data-testid="stMetricValue"] {
        color: #F8FAFC;
    }

    div[data-testid="stImage"] img {
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }

    div[data-testid="stProgressBar"] > div {
        background: rgba(59, 130, 246, 0.15);
        border-radius: 20px;
    }

    div[data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #2563EB, #A78BFA);
        border-radius: 20px;
    }

    .result-card {
        padding: 27px;
        border-radius: 18px;
        text-align: center;
        margin: 15px 0;
    }

    .fake-result {
        background: linear-gradient(
            135deg,
            rgba(127, 29, 29, 0.30),
            rgba(30, 15, 25, 0.8)
        );
        border: 1px solid rgba(248, 113, 113, 0.35);
    }

    .real-result {
        background: linear-gradient(
            135deg,
            rgba(6, 78, 59, 0.35),
            rgba(15, 35, 35, 0.8)
        );
        border: 1px solid rgba(52, 211, 153, 0.35);
    }

    .result-icon {
        font-size: 42px;
        margin-bottom: 10px;
    }

    .result-title {
        font-size: clamp(1.3rem, 3vw, 1.9rem);
        font-weight: 800;
        margin: 5px 0 10px;
    }

    .fake-text {
        color: #F87171;
    }

    .real-text {
        color: #34D399;
    }

    .result-score {
        font-size: 2.8rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 12px 0;
    }

    .result-note {
        color: #AAB6CA;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .footer {
        text-align: center;
        padding: 25px 10px;
        margin-top: 35px;
        border-top: 1px solid rgba(148, 163, 184, 0.12);
        color: #64748B;
        font-size: 0.8rem;
    }

    @media screen and (max-width: 768px) {
        .main .block-container {
            padding: 1rem 1rem 2rem;
        }

        .hero {
            padding: 27px 12px;
            border-radius: 16px;
        }

        .hero-icon {
            width: 54px;
            height: 54px;
            font-size: 25px;
        }

        .hero-subtitle {
            font-size: 0.82rem;
        }

        .section-heading {
            font-size: 1.1rem;
        }

        .result-card {
            padding: 20px 12px;
        }

        .result-score {
            font-size: 2.2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. HERO SECTION
# --------------------------------------------------

st.html(
    """
    <div class="hero">
        <div class="hero-icon">🛡️</div>
        <div class="hero-tag">AI POWERED MEDIA SECURITY</div>

        <h1 class="hero-title">Deepfake Detection AI</h1>

        <p class="hero-subtitle">
            Advanced Media Verification using Deep Learning
            &amp; Computer Vision
        </p>
    </div>
    """
)


# --------------------------------------------------
# 4. LOAD TRAINED & PRETRAINED MODELS
# --------------------------------------------------

@st.cache_resource
def load_trained_model():
    """Load the user's fine-tuned ResNet18 checkpoint on CPU."""
    model = models.resnet18(weights=None)

    model_path = os.path.join("models", "deepfake_model.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            "Trained model not found: models/deepfake_model.pth"
        )

    checkpoint = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if isinstance(checkpoint, dict) and checkpoint and all(
        key.startswith("module.") for key in checkpoint
    ):
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}

    if isinstance(checkpoint, dict) and "fc.1.weight" in checkpoint:
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.fc.in_features, 2),
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


DEEPGUARD_MODEL_ID = "KoreaPeter/ms-eff-gcvit-deepfake-b0-ff-plus-plus"
AI_SYNTH_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"


@st.cache_resource
def load_deepguard_model():
    """Load DeepGuard (FaceSwap detector) and ViT AI Face Synthesis detector."""
    ffpp_detector = hf_pipeline(
        task="image-classification",
        model=DEEPGUARD_MODEL_ID,
        trust_remote_code=True,
        device=-1,
    )
    ai_detector = hf_pipeline(
        task="image-classification",
        model=AI_SYNTH_MODEL_ID,
        device=-1,
    )
    return ffpp_detector, ai_detector


def extract_fake_probability(output):
    """Extract fake/artificial score accurately from pipeline output."""
    if isinstance(output, dict):
        output = [output]
    if output and isinstance(output[0], list):
        output = output[0]

    for item in (output or []):
        label = str(item.get("label", "")).strip().lower()
        if any(k in label for k in ("fake", "artificial", "deepfake", "ai", "generated")):
            return float(item["score"])

    for item in (output or []):
        label = str(item.get("label", "")).strip().lower()
        if any(k in label for k in ("real", "human", "realism", "authentic")):
            return 1.0 - float(item["score"])

    return None


try:
    with st.spinner("Initializing ResNet18 and DeepGuard AI models..."):
        model = load_trained_model()
        deepguard_model, ai_synth_model = load_deepguard_model()
except Exception as error:
    st.error(f"AI model initialization failed: {error}")
    st.info(
        "Check that the required packages are installed and that you have "
        "an active internet connection."
    )
    st.stop()


# --------------------------------------------------
# 5. IMAGE TRANSFORMATION
# --------------------------------------------------

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# --------------------------------------------------
# 5A. COMPACT FRAME PREVIEW
# --------------------------------------------------

def make_frame_thumbnail(image, max_width=160, max_height=190):
    """
    Create a smaller display-only thumbnail.
    The original image remains unchanged for AI prediction.
    """
    height, width = image.shape[:2]

    if height == 0 or width == 0:
        return image

    scale = min(
        max_width / width,
        max_height / height,
        1.0,
    )

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


# --------------------------------------------------
# 6. VIDEO UPLOAD SECTION
# --------------------------------------------------

st.markdown(
    """
    <div class="section-heading">📁 Upload Your Video</div>
    <div class="section-description">
        Select a video file to analyze its facial regions
        and identify possible deepfake manipulation.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    uploaded_video = st.file_uploader(
        "Drop your video here or browse files",
        type=["mp4", "mov", "avi"],
        help="Supported formats: MP4, MOV and AVI",
        key="video_uploader",
    )

    video_ready = uploaded_video is not None

    if video_ready:
        file_size = uploaded_video.size / (1024 * 1024)

        file_col, size_col = st.columns(2)

        with file_col:
            st.metric(
                "Video File",
                uploaded_video.name[:25],
            )

        with size_col:
            st.metric(
                "File Size",
                f"{file_size:.2f} MB",
            )

        st.success(
            f"Video uploaded successfully: {uploaded_video.name}"
        )
    else:
        st.info("Please upload a video to enable AI analysis.")

    analyze_button = st.button(
        "⚡ Start AI Video Analysis",
        disabled=not video_ready,
        use_container_width=True,
        type="primary",
        key="analyze_video_button",
    )


# --------------------------------------------------
# 7. VIDEO ANALYSIS
# --------------------------------------------------

if analyze_button and uploaded_video is not None:
    TOTAL_TARGET_FRAMES = 12
    RESNET_WEIGHT = 0.02      # 2% weight for custom ResNet18
    DEEPGUARD_WEIGHT = 0.98   # 98% weight for Pretrained DeepGuard + AI Detector

    predictions = []
    deepguard_predictions = []
    sampled_frames = []

    st.markdown("---")
    st.markdown(
        """
        <div class="section-heading">🔬 Video Analysis</div>
        <div class="section-description">
            Sampling video frames and comparing your trained ResNet18 with the pretrained DeepGuard model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not face_detector_available():
        st.error(
            "YuNet face detector is unavailable. Add "
            "models/face_detection_yunet_2023mar.onnx and restart the app."
        )
        st.stop()

    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_cols = st.columns(4)
    temp_path = None
    cap = None

    try:
        suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_video.getvalue())
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise RuntimeError("Unable to open this video file.")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count_total / fps if fps > 0 else 0.0

        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.metric("Duration", f"{duration:.1f} sec" if duration else "N/A")
        info_col2.metric("FPS", f"{fps:.1f}" if fps > 0 else "N/A")
        info_col3.metric("Target Frames", TOTAL_TARGET_FRAMES)

        if frame_count_total <= 0:
            raise RuntimeError("Could not determine the video's frame count.")

        if frame_count_total <= TOTAL_TARGET_FRAMES:
            target_indices = list(range(frame_count_total))
        else:
            target_indices = np.linspace(
                0, frame_count_total - 1, TOTAL_TARGET_FRAMES + 2, dtype=int
            )[1:-1].tolist()

        for index, frame_index in enumerate(target_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            face_rgb = extract_face_region(frame, padding=0.12)
            if face_rgb is None or not isinstance(face_rgb, np.ndarray) or face_rgb.size == 0:
                continue

            if face_rgb.ndim == 2:
                face_rgb = cv2.cvtColor(face_rgb, cv2.COLOR_GRAY2RGB)
            elif face_rgb.ndim == 3 and face_rgb.shape[2] == 4:
                face_rgb = cv2.cvtColor(face_rgb, cv2.COLOR_RGBA2RGB)
            elif face_rgb.ndim != 3 or face_rgb.shape[2] != 3:
                continue

            # 1. User's ResNet18 with TTA
            face_flipped = np.ascontiguousarray(np.fliplr(face_rgb))
            batch_tensor = torch.stack([transform(face_rgb), transform(face_flipped)])
            with torch.inference_mode():
                logits = model(batch_tensor)
                probs = torch.softmax(logits, dim=1)
                resnet_fake_probability = float(probs[:, 1].mean().item())

            if not np.isfinite(resnet_fake_probability):
                continue

            # 2A. DeepGuard (FaceSwap / FaceForensics++ detector)
            original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            full_pil = Image.fromarray(original_rgb)
            face_pil = Image.fromarray(face_rgb)

            deepguard_output = deepguard_model(
                full_pil,
                top_k=None,
                margin_ratio=0.2,
                conf_thres=0.5,
                min_face_ratio=0.01,
                tta_hflip=0.5,
            )
            ffpp_prob = extract_fake_probability(deepguard_output)

            # 2B. AI Synthesis Detector (Pass cropped face + horizontal flip instead of full frame)
            face_flip_pil = Image.fromarray(face_flipped)
            ai_face_out = ai_synth_model(face_pil, top_k=None)
            ai_flip_out = ai_synth_model(face_flip_pil, top_k=None)

            ai_face_prob = extract_fake_probability(ai_face_out)
            ai_flip_prob = extract_fake_probability(ai_flip_out)

            if ai_face_prob is not None and ai_flip_prob is not None:
                ai_prob = 0.5 * (ai_face_prob + ai_flip_prob)
            else:
                ai_prob = ai_face_prob if ai_face_prob is not None else ai_flip_prob

            if ffpp_prob is None and ai_prob is None:
                continue

            # Combine FaceSwap specialist and AI-Generation specialist:
            # Real videos score low on BOTH. Fake videos (either FaceSwap OR AI-Generated) score high on at least one.
            if ffpp_prob is not None and ai_prob is not None:
                deepguard_fake_probability = float(max(ffpp_prob, ai_prob))
            else:
                deepguard_fake_probability = float(ffpp_prob if ffpp_prob is not None else ai_prob)

            if not np.isfinite(deepguard_fake_probability):
                continue

            predictions.append(resnet_fake_probability)
            deepguard_predictions.append(deepguard_fake_probability)
            sampled_frames.append(face_rgb)

            preview = make_frame_thumbnail(face_rgb, max_width=160, max_height=190)
            with frame_cols[(len(sampled_frames) - 1) % 4]:
                st.image(
                    preview,
                    caption=f"Sample {len(sampled_frames)} · frame {frame_index}",
                    width=160,
                )

            progress_bar.progress(int((index + 1) / len(target_indices) * 100))
            status_text.markdown(f"**Analyzing video... {index + 1}/{len(target_indices)} samples**")

        cap.release()
        cap = None
        progress_bar.progress(100)

        if len(predictions) < 3 or len(deepguard_predictions) < 3:
            st.warning(
                f"Usable ResNet18 samples: {len(predictions)}; "
                f"DeepGuard samples: {len(deepguard_predictions)}. "
                "At least 3 valid samples from both models are required."
            )
        else:
            status_text.success("Analysis completed.")

            resnet_median = float(np.median(predictions))
            deepguard_median = float(np.median(deepguard_predictions))
            ensemble_fake_score = (
                (RESNET_WEIGHT * resnet_median) + (DEEPGUARD_WEIGHT * deepguard_median)
            ) * 100.0
            median_fake_score = ensemble_fake_score
            real_score = 100.0 - ensemble_fake_score

            st.markdown("---")
            st.markdown(
                """
                <div class="section-heading">🎯 Detection Results</div>
                <div class="section-description">
                    Result based on the median scores from both models, combined using a calibrated ensemble.
                </div>
                """,
                unsafe_allow_html=True,
            )

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Frames Analyzed", len(predictions))
            metric_col2.metric("ResNet18 Fake Score", f"{resnet_median * 100:.2f}%")
            metric_col3.metric("DeepGuard Fake Score", f"{deepguard_median * 100:.2f}%")
            st.metric("Ensemble Fake Score (Weighted)", f"{median_fake_score:.2f}%")

            if median_fake_score >= 50.0:
                st.html(
                    f"""
                    <div class="result-card fake-result">
                        <div class="result-icon">🚨</div>
                        <div class="result-title fake-text">Potential Deepfake Detected</div>
                        <div class="result-score">{median_fake_score:.2f}%</div>
                        <div class="result-note">
                            The combined models assigned a higher fake-class score to the sampled video frames.
                            This is not definitive proof of manipulation.
                        </div>
                    </div>
                    """
                )
            else:
                st.html(
                    f"""
                    <div class="result-card real-result">
                        <div class="result-icon">🛡️</div>
                        <div class="result-title real-text">Likely Authentic Media</div>
                        <div class="result-score">{real_score:.2f}%</div>
                        <div class="result-note">
                            The combined models assigned a higher real-class score to the sampled video frames.
                            This does not guarantee that the video is genuine.
                        </div>
                    </div>
                    """
                )

            st.caption(
                "Important: this is a weighted ensemble score (2% ResNet18 + 98% DeepGuard), not proof of authenticity. "
                "Evaluate on independent labelled videos before reporting accuracy."
            )

    except Exception as error:
        st.error(f"Video analysis failed: {error}")

    finally:
        if cap is not None:
            cap.release()
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------

st.html(
    """
    <div class="footer">
        <div style="font-size:20px; margin-bottom:8px;">🛡️</div>

        <b style="color:#CBD5E1;">
            Deepfake Detection AI
        </b>

        <p style="margin-top:7px;">
            CPU Ensemble · ResNet18 + DeepGuard MS-EffGCViT-B0
        </p>

        <p style="font-size:11px; color:#64748B;">
            Built with Python · Streamlit · PyTorch · Transformers · OpenCV
        </p>
    </div>
    """
)