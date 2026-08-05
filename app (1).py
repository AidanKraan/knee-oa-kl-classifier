import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms, models
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Knee OA KL-Grade Classifier", layout="centered")

# ---------- Load the trained model from Hugging Face (cached) ----------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="AidanKraan/knee-oa-kl-model",
        filename="best_knee_model.pth"
    )
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ---------- Preprocessing (must match training) ----------
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

grade_names = {
    0: "Grade 0 — None", 1: "Grade 1 — Doubtful", 2: "Grade 2 — Minimal",
    3: "Grade 3 — Moderate", 4: "Grade 4 — Severe"
}

# ---------- Grad-CAM ----------
def make_gradcam(input_tensor, pred_class):
    grads, acts = {}, {}
    def fwd(m, i, o): acts["v"] = o
    def bwd(m, gi, go): grads["v"] = go[0]
    layer = model.layer4[-1]
    h1 = layer.register_forward_hook(fwd)
    h2 = layer.register_full_backward_hook(bwd)

    output = model(input_tensor)
    model.zero_grad()
    output[0, pred_class].backward()

    pooled = grads["v"].mean(dim=[0, 2, 3])
    a = acts["v"][0]
    for i in range(a.shape[0]):
        a[i] *= pooled[i]
    hm = a.mean(0).detach().numpy()
    hm = np.maximum(hm, 0)
    hm = hm / (hm.max() + 1e-8)
    h1.remove(); h2.remove()
    return hm

# ---------- Interface ----------
st.title("Knee Osteoarthritis KL-Grade Classifier")
st.markdown(
    "Upload a knee X-ray to get a predicted **Kellgren–Lawrence grade (0–4)** "
    "and a **Grad-CAM heatmap** showing where the model looked."
)

st.warning(
    "⚠️ **Educational prototype only — NOT for clinical or diagnostic use.** "
    "This model was trained on a public research dataset by a student for learning purposes. "
    "It has not been clinically validated and must not inform any medical decision."
)

uploaded = st.file_uploader("Upload a knee X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)[0]
    pred_class = int(output.argmax(1).item())
    confidence = float(probs[pred_class])

    hm = make_gradcam(input_tensor, pred_class)
    disp = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    hm_resized = cv2.resize(hm, (224, 224))
    hm_color = cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB) / 255.0
    overlay = np.clip(0.5 * disp + 0.5 * hm_color, 0, 1)

    col1, col2 = st.columns(2)
    col1.image(disp, caption="Uploaded X-ray", use_container_width=True)
    col2.image(overlay, caption="Grad-CAM (where the model looked)", use_container_width=True)

    st.subheader(f"Prediction: {grade_names[pred_class]}")
    st.write(f"Confidence: **{confidence*100:.1f}%**")

    st.markdown("**Probability by grade:**")
    for g in range(5):
        st.write(f"{grade_names[g]}: {probs[g]*100:.1f}%")
