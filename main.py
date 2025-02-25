import os
import gdown
import torch
import streamlit as st
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

# Google Drive File ID
GDRIVE_FILE_ID = "1yLRjGY-2d3phiT0dXtVDPVYavmhVxXsf"
MODEL_PATH = "model/paddy_disease_classifier_final.pt"

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model... Please wait."):
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")

# Load the pre-trained model
@st.cache_resource
def load_model():
    try:
        model = ViTForImageClassification.from_pretrained(
            "google/vit-large-patch16-224-in21k", num_labels=10
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

model = load_model()
processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")

# Define class labels
index_to_disease = {
    0: "tungro", 1: "brown_spot", 2: "bacterial_leaf_blight",
    3: "bacterial_leaf_streak", 4: "blast", 5: "downy_mildew",
    6: "dead_heart", 7: "hispa", 8: "normal", 9: "bacterial_panicle_blight"
}

# Streamlit UI
st.title("🌾 Paddy Disease Classifier")
st.write("Upload an image of a paddy leaf to classify its disease.")

# Display instructions as HTML with custom styling
st.markdown("""
    <style>
        .instructions {
            font-size: 16px;
            color: #FF5733;
        }
        .note-icon {
            color: #FF5733;
            font-weight: bold;
        }
    </style>
    <p class="instructions">
        <span class="note-icon">⚠ Note</span>: Poor-quality images or images of other plants may lead to incorrect predictions, 
        as the model is trained specifically on paddy diseases.
    </p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process Image
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits[0], dim=0).tolist()

    # Show Prediction Results
    st.subheader("Prediction Results:")
    for i, disease in index_to_disease.items():
        st.write(f"**{disease}**: {round(probabilities[i], 4)}")
