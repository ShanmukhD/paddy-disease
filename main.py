import os
import gdown
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from flask import Flask, request, render_template, jsonify
from PIL import Image

app = Flask(__name__)

# Google Drive File ID (Updated)
GDRIVE_FILE_ID = "1yLRjGY-2d3phiT0dXtVDPVYavmhVxXsf"
MODEL_PATH = "model/paddy_disease_classifier_final.pt"

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)
    print("✅ Model downloaded successfully!")

# Load the pre-trained model
try:
    model = ViTForImageClassification.from_pretrained(
        "google/vit-large-patch16-224-in21k", num_labels=10
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Model loading failed:", str(e))
    raise SystemExit("Exiting due to model load failure.")

# Initialize image processor
processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")

# Define class labels
index_to_disease = {
    0: "tungro", 1: "brown_spot", 2: "bacterial_leaf_blight",
    3: "bacterial_leaf_streak", 4: "blast", 5: "downy_mildew",
    6: "dead_heart", 7: "hispa", 8: "normal", 9: "bacterial_panicle_blight"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    try:
        image = Image.open(file)
    except Exception as e:
        return jsonify({"error": "Invalid image file."})
    
    # Convert image to RGB
    image = image.convert('RGB')
    
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits[0], dim=0).tolist()
    
    result = {index_to_disease[i]: round(probabilities[i], 4) for i in range(len(index_to_disease))}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
