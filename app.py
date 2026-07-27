"""
app.py
------
Streamlit UI for Sign Language Recognition.

Two modes:
    1. 📁 Image Upload  -> predict a single uploaded photo
    2. 📷 Live Webcam   -> real-time recognition using streamlit-webrtc

Run with:
    streamlit run app.py
"""

import numpy as np
import cv2
import streamlit as st
from PIL import Image

from utils.predict import load_sign_model, predict_frame

# ----------------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .pred-box {
        padding: 1.2rem;
        border-radius: 14px;
        background: #f7f7fb;
        border: 1px solid #eaeaf2;
        text-align: center;
    }
    .pred-letter {
        font-size: 4rem;
        font-weight: 900;
        color: #2575fc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🤟 Sign Language Recognition</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">CNN-powered ASL alphabet recognition — upload an image or use your webcam live</div>',
    unsafe_allow_html=True,
)
st.caption("Note: this model recognizes static letter signs (A-Z except J and Z, which need hand motion).")


# ----------------------------------------------------------------------------
# Load model (cached so it only loads once per session)
# ----------------------------------------------------------------------------
@st.cache_resource
def get_model():
    return load_sign_model(
        model_path="model/sign_language_model.h5",
        class_map_path="model/class_indices.json",
    )


try:
    model, idx_to_class = get_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(
        "⚠️ Could not load the trained model. Run `train_model.py` first to "
        "generate `model/sign_language_model.h5` and `model/class_indices.json`.\n\n"
        f"Details: {e}"
    )

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.60, 0.05)
mode = st.sidebar.radio("Choose mode", ["📁 Image Upload", "📷 Live Webcam"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Tip:** Keep your hand centered, well-lit, and against a plain "
    "background for best accuracy."
)

# ----------------------------------------------------------------------------
# Mode 1: Image Upload
# ----------------------------------------------------------------------------
if mode == "📁 Image Upload" and model_loaded:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload an image")
        uploaded_file = st.file_uploader(
            "Choose a hand-sign image", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded image", use_column_width=True)

    with col2:
        st.subheader("Prediction")
        if uploaded_file is not None:
            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            label, confidence, all_preds = predict_frame(model, idx_to_class, frame_bgr)

            st.markdown(
                f"""
                <div class="pred-box">
                    <div class="pred-letter">{label.upper()}</div>
                    <div>Confidence: <b>{confidence*100:.1f}%</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if confidence < conf_threshold:
                st.warning(
                    "Low confidence — try a clearer photo, better lighting, "
                    "or center the hand more."
                )

            with st.expander("See top-5 class probabilities"):
                top5_idx = np.argsort(all_preds)[::-1][:5]
                for i in top5_idx:
                    st.write(f"**{idx_to_class[i]}** — {all_preds[i]*100:.1f}%")
        else:
            st.info("Upload an image on the left to see the prediction here.")

# ----------------------------------------------------------------------------
# Mode 2: Live Webcam (real-time, via streamlit-webrtc)
# ----------------------------------------------------------------------------
elif mode == "📷 Live Webcam" and model_loaded:
    st.subheader("Live webcam recognition")
    st.caption(
        "Allow camera access below. The predicted letter and confidence "
        "are overlaid directly on the video feed."
    )

    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import av

    class SignPredictor(VideoProcessorBase):
        def __init__(self):
            self.model = model
            self.idx_to_class = idx_to_class
            self.conf_threshold = conf_threshold

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Draw a guide box where the hand should be placed
            h, w, _ = img.shape
            box_size = min(h, w) // 2
            x1, y1 = w // 2 - box_size // 2, h // 2 - box_size // 2
            x2, y2 = x1 + box_size, y1 + box_size
            roi = img[max(y1, 0):y2, max(x1, 0):x2]

            if roi.size > 0:
                label, confidence, _ = predict_frame(self.model, self.idx_to_class, roi)
                color = (0, 200, 0) if confidence >= self.conf_threshold else (0, 165, 255)
                text = f"{label.upper()} ({confidence*100:.0f}%)"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img, text, (x1, max(y1 - 15, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
                )

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="sign-language-live",
        video_processor_factory=SignPredictor,
        media_stream_constraints={"video": True, "audio": False},
    )

    st.info(
        "Place your hand inside the green/orange box. Green = confident "
        "prediction, orange = below your confidence threshold."
    )

st.markdown("---")
st.caption(
    "Built with TensorFlow/Keras, OpenCV, and Streamlit • "
    "Trained on the Kaggle ASL Alphabet dataset"
)
