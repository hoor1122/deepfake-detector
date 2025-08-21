import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import time

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# ====== CUSTOM CSS ======
st.markdown("""
<style>
.stApp { background-color: #f0f7ff; font-family: 'Segoe UI', sans-serif; padding: 1rem; }
.header { background: linear-gradient(90deg, #4B8BBE, #306998); padding: 1rem; border-radius: 15px; color: white; text-align: center; box-shadow: 0 6px 15px rgba(75,139,190,0.4); margin-bottom: 1rem; }
.stButton>button { background-color: #4B8BBE; color: white; font-weight: 600; font-size: 1rem; border-radius: 10px; padding: 0.5rem 1.5rem; margin: 0.5rem 0; }
.stButton>button:hover { background-color: #306998; }
.result-box { background-color: #e9f0f7; border-radius: 15px; padding: 1rem; text-align: center; font-weight: 600; margin-top: 1rem; }
.pred-fake { color: #d32f2f; }
.pred-real { color: #388e3c; }
</style>
""", unsafe_allow_html=True)

# ====== HEADER ======
st.markdown('<div class="header"><h1>🕵️‍♂️ DeepFake Detector</h1><p>Upload an image and let AI detect if it\'s Real or Fake.</p></div>', unsafe_allow_html=True)

# ====== LAYOUT ======
col1, col2 = st.columns([2, 1])

# ====== IMAGE UPLOAD ======
with col1:
    uploaded_file = st.file_uploader("📤 Choose an image file", type=["jpg","jpeg","png"])
    image = None
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

# ====== MODEL SELECTION ======
model_choice = None
if uploaded_file:
    with col1:
        model_choice = st.selectbox(
            "🔍 Select a model to analyze",
            ["Fine-Tuned ShuffleNetV2", "ShuffleNetV2", "CNN"]
        )

# ====== MODEL LOADERS ======
@st.cache_resource
def load_finetuned_shufflenet():
    model = models.shufflenet_v2_x1_0(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load("best_shufflenet.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_shufflenet():
    model = models.shufflenet_v2_x1_0(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.eval()
    return model

@st.cache_resource
def load_cnn():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.eval()
    return model

# ====== IMAGE TRANSFORM ======
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ====== ANALYZE BUTTON ======
prediction_result = None
probs_all_models = {}

if uploaded_file and model_choice:
    analyze = st.button("Analyze")
    if analyze:
        if model_choice == "Fine-Tuned ShuffleNetV2":
            model = load_finetuned_shufflenet()
        elif model_choice == "ShuffleNetV2":
            model = load_shufflenet()
        elif model_choice == "CNN":
            model = load_cnn()
        
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            _, predicted = torch.max(outputs, 1)
            class_names = ['Fake','Real']
            pred_class = class_names[predicted.item()]
            confidence = probs[predicted.item()] * 100

        prediction_result = (pred_class, confidence)
        probs_all_models[model_choice] = probs

# ====== CLEAR BUTTON ======
if uploaded_file:
    clear = st.button("Clear / Upload New Image")
    if clear:
        st.experimental_rerun()

# ====== DISPLAY RESULTS ======
with col1:
    if prediction_result:
        pred_class, confidence = prediction_result
        color_class = "pred-real" if pred_class=="Real" else "pred-fake"
        st.markdown(f'<div class="result-box">🧠 Prediction: <span class="{color_class}">{pred_class}</span><br>Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)

# ====== BAR GRAPHS (Sidebar) ======
with col2:
    st.subheader("Prediction Probabilities")
    for model_name in ["Fine-Tuned ShuffleNetV2","ShuffleNetV2","CNN"]:
        if model_name in probs_all_models:
            probs = probs_all_models[model_name]
            fig, ax = plt.subplots()
            bars = ax.bar(class_names, probs*100, color=['#d32f2f','#388e3c'])
            ax.set_ylim([0,100])
            ax.set_ylabel("Probability (%)")
            ax.set_title(model_name)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}%', xy=(bar.get_x()+bar.get_width()/2, height), xytext=(0,3), textcoords="offset points", ha='center')
            st.pyplot(fig)
