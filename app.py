from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
import joblib
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
import io
import os

print("🚀 Starting Digital Library ML API on Render...")

app = Flask(__name__)
CORS(app)  # Allow your main app to call this API

# Load ML models
print("🔍 Loading ML models...")
try:
    vectorizer = joblib.load('vectorizer.joblib')
    kmeans = joblib.load('kmeans.joblib')
    print(f"✅ Models loaded! Clusters: {kmeans.n_clusters}")
    MODELS_LOADED = True
except Exception as e:
    print(f"❌ Error: {e}")
    MODELS_LOADED = False

CLUSTER_NAMES = ['BSCS', 'BSED-MATH', 'BSES', 'BSHM', 'BTLED-HE', 'BEED']

@app.route('/')
def home():
    return jsonify({
        "message": "Digital Library ML API",
        "status": "running",
        "models": MODELS_LOADED
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    """Main endpoint for your Render app"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file", "cluster": 0, "confidence": 0.5})
        
        file = request.files['file']
        pdf_bytes = file.read()
        
        # Extract text
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for i in range(min(2, len(reader.pages))):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text += page_text + " "
        
        if not text.strip():
            text = "academic research"
        
        # Predict
        X = vectorizer.transform([text])
        cluster = kmeans.predict(X)[0]
        
        # Confidence
        closest, distances = pairwise_distances_argmin_min(X, kmeans.cluster_centers_)
        distance = distances[0]
        confidence = max(0.4, min(0.95, 1.0 - (distance / 20.0)))
        
        cluster = max(0, min(5, cluster))
        
        return jsonify({
            "success": True,
            "cluster": int(cluster),
            "confidence": float(confidence),
            "cluster_name": CLUSTER_NAMES[cluster],
            "text_preview": text[:100]
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "cluster": 0,
            "confidence": 0.5
        })

@app.route('/predict-text', methods=['POST'])
def predict_text():
    """Test endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"success": False, "error": "No text", "cluster": 0, "confidence": 0.5})
        
        X = vectorizer.transform([text])
        cluster = kmeans.predict(X)[0]
        
        closest, distances = pairwise_distances_argmin_min(X, kmeans.cluster_centers_)
        distance = distances[0]
        confidence = max(0.4, min(0.95, 1.0 - (distance / 20.0)))
        
        cluster = max(0, min(5, cluster))
        
        return jsonify({
            "success": True,
            "cluster": int(cluster),
            "confidence": float(confidence),
            "cluster_name": CLUSTER_NAMES[cluster],
            "text": text[:50]
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "cluster": 0, "confidence": 0.5})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)