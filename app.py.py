import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import time
import matplotlib.pyplot as plt

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🕵️‍♂️",
    layout="centered"
)

# ====== CUSTOM CSS ======
st.markdown("""
<style>
    .stApp {
        background-color: #f0f7ff;
        color: #222222;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 2rem 1rem 4rem 1rem;
    }
    .header {
        background: linear-gradient(90deg, #4B8BBE, #306998);
        padding: 2rem 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 15px rgba(75,139,190,0.4);
        margin-bottom: 1.5rem;
    }
    div[data-testid="fileUploaderDropzone"] {
        background: #61a0af;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.5rem;
        border: none;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(97,160,175,0.5);
        transition: background-color 0.3s ease;
    }
    div[data-testid="fileUploaderDropzone"]:hover {
        background: #468a96;
        box-shadow: 0 6px 20px rgba(70,138,150,0.7);
    }
    img {
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.15);
        max-width: 350px;
        height: auto;
        display: block;
        margin: 0 auto 25px auto;
    }
    .result-box {
        background-color: #e9f0f7;
        border-radius: 15px;
        padding: 2rem;
        max-width: 450px;
        margin: 1.5rem auto;
        box-shadow: 0 6px 24px rgba(48,105,152,0.25);
        text-align: center;
        color: #222222;
        font-weight: 600;
        font-size: 1.2rem;
    }
    .pred-fake {
        color: #d32f2f;
        background-color: #ffebee;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        box-shadow: 0 3px 10px rgba(211,47,47,0.3);
        display: inline-block;
        font-weight: 700;
        font-size: 1.4rem;
        margin-left: 10px;
    }
    .pred-real {
        color: #388e3c;
        background-color: #e8f5e9;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        box-shadow: 0 3px 10px rgba(56,142,60,0.3);
        display: inline-block;
        font-weight: 700;
        font-size: 1.4rem;
        margin-left: 10px;
    }
    .tagline {
        text-align: center;
        color: #555555;
        font-style: italic;
        margin-top: -10px;
        margin-bottom: 25px;
        font-size: 1rem;
    }
    .footer {
        font-size: 0.85rem;
        text-align: center;
        margin-top: 3rem;
        color: #555555;
        font-style: italic;
    }
    .analyze-button {
        display: block;
        margin: 0 auto 20px auto;
        background-color:#ff5722;
        color: blue;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.7rem 2rem;
        border-radius: 10px;
        cursor: pointer;
        border: none;
        box-shadow: 0 5px 15px rgba(75,139,190,0.4);
        transition: background-color 0.3s ease;
    }
    .analyze-button:hover {
        background-color: #306998;
    }
</style>
""", unsafe_allow_html=True)

# ====== HEADER ======
st.markdown("""
<div class="header">
    <h1>🕵️‍♂️ DeepFake Detector</h1>
    <p>Upload an image and let AI detect if it's <strong>Real</strong> or <strong>Fake</strong>.</p>
</div>
""", unsafe_allow_html=True)

# ====== MODEL SELECTION ======
model_choice = st.selectbox(
    "🔍 Select a model",
    ["Fine-Tuned ShuffleNetV2", "ShuffleNetV2", "CNN"]
)

# ====== MODEL LOADING FUNCTIONS ======
@st.cache_resource
def load_finetuned_shufflenet():
    model = models.shufflenet_v2_x1_0(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load("best_shufflenet.pth", map_location=torch.device("cpu")))
    model.to(torch.device("cpu"))
    model.eval()
    return model

@st.cache_resource
def load_shufflenet():
    model = models.shufflenet_v2_x1_0(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Dummy 2-class output
    model.to(torch.device("cpu"))
    model.eval()
    return model

@st.cache_resource
def load_cnn():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Dummy 2-class output
    model.to(torch.device("cpu"))
    model.eval()
    return model

# ====== LOAD SELECTED MODEL ======
if model_choice == "Fine-Tuned ShuffleNetV2":
    model = load_finetuned_shufflenet()
elif model_choice == "ShuffleNetV2":
    model = load_shufflenet()
elif model_choice == "CNN":
    model = load_cnn()

# ====== IMAGE TRANSFORM ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== FILE UPLOAD + SESSION ======
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

uploaded_file = st.file_uploader(
    "📤 Choose an image file", type=["jpg", "jpeg", "png"],
    key=st.session_state.get("uploader_key", 0)
)

if uploaded_file is not None:
    st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")

# Tagline
st.markdown(
    '<p class="tagline">Upload a face image to detect deepfakes — stay aware!</p>',
    unsafe_allow_html=True
)

# Agar image available hai to dikhaye aur buttons show kare
if st.session_state.uploaded_image is not None:
    image = st.session_state.uploaded_image
    st.image(image, caption='🖼 Uploaded Image')

    col1, col2, col3 = st.columns(3)
    with col1:
        analyze = st.button("🔍 Analyze (Selected Model)")
    with col2:
        all_models = st.button("🧠 All Models Compare")
    with col3:
        clear = st.button("🗑️ Clear")

    # ====== ANALYZE BUTTON ======
    if analyze:
        with st.spinner(f"Analyzing picture with {model_choice}..."):
            progress_bar = st.progress(0)
            img_tensor = transform(image).unsqueeze(0).to("cpu")

            for percent in range(0, 101, 20):
                time.sleep(0.15)
                progress_bar.progress(percent)

            class_names = ['Fake', 'Real']
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                pred_idx = probs.argmax()
                pred_class = class_names[pred_idx]
                confidence = probs[pred_idx] * 100

            progress_bar.progress(100)
            time.sleep(0.2)

        st.markdown(
            f"""
            <div class="result-box">
                <span>🧠 Prediction:</span><br>
                **{model_choice}**: {pred_class} ({confidence:.2f}%)
            </div>
            """,
            unsafe_allow_html=True
        )

        # Graph
        fig, ax = plt.subplots()
        ax.bar(class_names, probs*100, color=['red', 'green'])
        ax.set_ylabel('Probability (%)')
        ax.set_title(f'{model_choice} Prediction')
        ax.set_ylim([0, 100])
        st.pyplot(fig)

    # ====== ALL MODELS BUTTON ======
    if all_models:
        with st.spinner("Analyzing picture with all models..."):
            progress_bar = st.progress(0)
            img_tensor = transform(image).unsqueeze(0).to("cpu")

            for percent in range(0, 101, 20):
                time.sleep(0.15)
                progress_bar.progress(percent)

            models_dict = {
                "Fine-Tuned ShuffleNetV2": load_finetuned_shufflenet(),
                "ShuffleNetV2": load_shufflenet(),
                "CNN": load_cnn()
            }

            class_names = ['Fake', 'Real']
            results = {}
            predictions_text = ""

            with torch.no_grad():
                for name, m in models_dict.items():
                    output = m(img_tensor)
                    probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                    results[name] = probs
                    pred_idx = probs.argmax()
                    pred_class = class_names[pred_idx]
                    confidence = probs[pred_idx] * 100
                    predictions_text += f"**{name}**: {pred_class} ({confidence:.2f}%)<br>"

            progress_bar.progress(100)
            time.sleep(0.2)

        st.markdown(
            f"""
            <div class="result-box">
                <span>🧠 Predictions:</span><br>{predictions_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Grouped chart
        import numpy as np
        fig, ax = plt.subplots()
        labels = class_names
        x = np.arange(len(labels))
        width = 0.25

        for i, (model_name, probs) in enumerate(results.items()):
            ax.bar(x + i*width, probs*100, width, label=model_name)

        ax.set_xticks(x + width)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Probability (%)')
        ax.set_title('All Models Prediction Comparison')
        ax.set_ylim([0, 100])
        ax.legend()
        st.pyplot(fig)

    # ====== CLEAR BUTTON ======
    if clear:
        st.session_state.uploaded_image = None
        st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
        st.rerun()

# ====== FOOTER ======
st.markdown(
    "<div class='footer'>🔍 This result is based on the uploaded image and may not be perfect. Always verify with additional tools.</div>",
    unsafe_allow_html=True
)





