# from flask import Flask, request, render_template, jsonify
# import torch
# from transformers import ViTForImageClassification, ViTImageProcessor
# from PIL import Image
# import os

# app = Flask(__name__)

# # Model path (update based on your local path)
# model_path = os.path.join("model", "paddy_disease_classifier_final.pt")

# # Load the pre-trained model
# try:
#     model = ViTForImageClassification.from_pretrained(
#         "google/vit-large-patch16-224-in21k", num_labels=10
#     )
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#     model.eval()
#     print("✅ Model loaded successfully!")
# except Exception as e:
#     print("❌ Model loading failed:", str(e))
#     raise SystemExit("Exiting due to model load failure.")

# # Initialize image processor
# processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")

# # Define class labels
# index_to_disease = {
#     0: 'tungro', 1: 'brown_spot', 2: 'bacterial_leaf_blight',
#     3: 'bacterial_leaf_streak', 4: 'blast', 5: 'downy_mildew',
#     6: 'dead_heart', 7: 'hispa', 8: 'normal', 9: 'bacterial_panicle_blight'
# }

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"})
    
#     file = request.files['file']
#     image = Image.open(file).convert('RGB')
#     inputs = processor(images=image, return_tensors="pt")
    
#     with torch.no_grad():
#         outputs = model(**inputs)
#         probabilities = torch.nn.functional.softmax(outputs.logits[0], dim=0).tolist()
    
#     result = {index_to_disease[i]: round(probabilities[i], 4) for i in range(len(index_to_disease))}
#     return jsonify(result)

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, render_template, jsonify
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os
import streamlit


app = Flask(__name__)

# Load model and processor
MODEL_PATH = "model/paddy_disease_classifier_final.pt"
LABELS = [
    'tungro', 'brown_spot', 'bacterial_leaf_blight', 'bacterial_leaf_streak',
    'blast', 'downy_mildew', 'dead_heart', 'hispa', 'normal', 'bacterial_panicle_blight'
]

try:
    model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224-in21k", num_labels=len(LABELS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    raise SystemExit("Exiting due to model load failure.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"})
    
    try:
        image = Image.open(file).convert('RGB')
    except Exception:
        return jsonify({"error": "Invalid image file."})
    
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits[0], dim=0).tolist()
    
    return jsonify({LABELS[i]: round(probabilities[i], 4) for i in range(len(LABELS))})

if __name__ == '__main__':
    app.run(debug=True)
