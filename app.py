import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms, models
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Knee OA KL-Grade Classifier", page_icon="", layout="wide")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="AidanKraan/knee-oa-kl-model", filename="best_knee_model.pth")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

# grade -> (label, description, severity color)
grade_info = {
    0: ("Grade 0", "None",     "#22c55e"),
    1: ("Grade 1", "Doubtful", "#84cc16"),
    2: ("Grade 2", "Minimal",  "#eab308"),
    3: ("Grade 3", "Moderate", "#f97316"),
    4: ("Grade 4", "Severe",   "#ef4444"),
}

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
    hm = np.maximum(hm, 0); hm = hm / (hm.max() + 1e-8)
    h1.remove(); h2.remove()
    return hm

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write(
        "An educational deep-learning demo that grades knee osteoarthritis severity "
        "from an X-ray on the **Kellgren–Lawrence (KL)** scale, with a **Grad-CAM** "
        "heatmap showing which regions drove the prediction."
    )
    st.subheader("KL grades")
    for g, (name, desc, color) in grade_info.items():
        st.markdown(f"<span style='color:{color};font-size:1.2rem'>●</span> **{name}** — {desc}",
                    unsafe_allow_html=True)
    st.subheader("Under the hood")
    st.write("ResNet-18, transfer-learned on an OAI-derived dataset (~8k radiographs). "
             "Test accuracy 66%; within ±1 grade 93.5%.")
    st.markdown("[🔗 View the code on GitHub](https://github.com/AidanKraan/knee-oa-kl-classifier)")

# ---------- Main ----------
st.title(" Knee Osteoarthritis KL-Grade Classifier")
st.caption("Upload a knee X-ray → predicted KL grade (0–4) + a Grad-CAM heatmap of where the model looked.")

st.warning(
    "⚠️ **Educational prototype only — NOT for clinical or diagnostic use.** "
    "Trained on a public research dataset by a student for learning purposes. "
    "Not clinically validated; must not inform any medical decision."
)

uploaded = st.file_uploader("Upload a knee X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)[0]
    pred_class = int(output.argmax(1).item())
    confidence = float(probs[pred_class])
    name, desc, color = grade_info[pred_class]

    hm = make_gradcam(input_tensor, pred_class)
    disp = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    hm_resized = cv2.resize(hm, (224, 224))
    hm_color = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET),
                            cv2.COLOR_BGR2RGB) / 255.0
    overlay = np.clip(0.5 * disp + 0.5 * hm_color, 0, 1)

    # Color-coded prediction banner
    st.markdown(
        f"<div style='background:{color}22;border-left:6px solid {color};"
        f"padding:16px 20px;border-radius:10px;margin:14px 0'>"
        f"<span style='font-size:1.5rem;font-weight:700;color:{color}'>{name} — {desc}</span><br>"
        f"<span style='opacity:0.85'>Model confidence: {confidence*100:.1f}%</span></div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    col1.image(disp, caption="Uploaded X-ray", use_container_width=True)
    col2.image(overlay, caption="Grad-CAM — where the model looked", use_container_width=True)

    st.subheader("Probability by grade")
    for g in range(5):
        gname, gdesc, gcolor = grade_info[g]
        marker = "  ⬅ predicted" if g == pred_class else ""
        st.write(f"**{gname}** — {gdesc}{marker}")
        st.progress(float(probs[g]))
        
