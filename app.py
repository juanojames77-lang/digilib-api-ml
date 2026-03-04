from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
import joblib
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
import io
import os
import re
import time

# KeyBERT imports
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

print("🚀 Starting Digital Library ML API on Render...")

app = Flask(__name__)
CORS(app)

# Track start time for uptime
START_TIME = time.time()

# Load ML models
print("🔍 Loading ML models...")
try:
    vectorizer = joblib.load('vectorizer.joblib')
    kmeans = joblib.load('kmeans.joblib')
    print(f"✅ Models loaded! Clusters: {kmeans.n_clusters}")
    MODELS_LOADED = True
except Exception as e:
    print(f"❌ Error loading models: {e}")
    MODELS_LOADED = False

# Initialize KeyBERT model
print("🔧 Initializing KeyBERT for topic extraction...")
try:
    # Use a lightweight but effective model
    # Options: 'all-MiniLM-L6-v2' (fast), 'all-mpnet-base-v2' (more accurate)
    kw_model = KeyBERT(model='all-MiniLM-L6-v2')
    print("✅ KeyBERT initialized successfully!")
    KEYBERT_LOADED = True
except Exception as e:
    print(f"❌ Error initializing KeyBERT: {e}")
    KEYBERT_LOADED = False
    kw_model = None

CLUSTER_NAMES = ['BSCS', 'BSED-MATH', 'BSES', 'BSHM', 'BTLED-HE', 'BEED']

# ========== KEYBERT TOPIC EXTRACTION ==========
def extract_topics_keybert(text, num_topics=8):
    """
    Extract topics using KeyBERT (BERT-based semantic understanding)
    Much better than TF-IDF for capturing meaningful topics
    """
    try:
        if not KEYBERT_LOADED or kw_model is None:
            print("⚠️ KeyBERT not available, falling back to TF-IDF")
            return extract_topics_tfidf(text, num_topics)
        
        # Clean text but preserve meaning
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        if len(text.split()) < 20:
            return [{"topic": "general", "score": 0.5}]
        
        # Extract keywords with diversity to get varied topics
        # keyphrase_ngram_range=(1, 3) captures 1-3 word phrases like "machine learning"
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=num_topics,
            use_mmr=True,  # Maximal Marginal Relevance for diversity
            diversity=0.7   # Balance between relevance and diversity
        )
        
        topics = []
        for keyword, score in keywords:
            # Capitalize each word for better display
            formatted_topic = ' '.join(word.capitalize() for word in keyword.split())
            topics.append({
                "topic": formatted_topic,
                "score": float(score)
            })
        
        return topics
        
    except Exception as e:
        print(f"❌ KeyBERT extraction error: {e}")
        return extract_topics_tfidf(text, num_topics)

# ========== FALLBACK TF-IDF TOPIC EXTRACTION ==========
def extract_topics_tfidf(text, num_topics=8):
    """Original TF-IDF based extraction as fallback"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\b\d+\b', '', text)
        
        words = text.split()
        if len(words) < 10:
            return [{"topic": "general", "score": 0.5}]
        
        academic_stop_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'were', 'be', 'been',
            'this', 'that', 'these', 'those', 'it', 'its', 'has', 'have', 'had',
            'study', 'research', 'thesis', 'paper', 'document', 'analysis',
            'chapter', 'section', 'figure', 'table', 'result', 'method'
        ]
        
        tfidf = TfidfVectorizer(
            max_features=30,
            stop_words=academic_stop_words,
            ngram_range=(1, 2),
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b'
        )
        
        tfidf_matrix = tfidf.fit_transform([text])
        feature_names = tfidf.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        top_indices = scores.argsort()[-num_topics:][::-1]
        
        topics = []
        for idx in top_indices:
            if scores[idx] > 0.1:
                topic_name = feature_names[idx]
                formatted_topic = ' '.join(word.capitalize() for word in topic_name.split())
                topics.append({
                    "topic": formatted_topic,
                    "score": float(scores[idx])
                })
        
        return topics
        
    except Exception as e:
        print(f"❌ TF-IDF extraction error: {e}")
        return []

# ========== EXTRACT TEXT FROM PDF ==========
def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF bytes - reading 10 pages"""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        text = ""
        
        pages_to_read = min(10, total_pages)
        print(f"📄 PDF has {total_pages} pages, reading {pages_to_read} pages")
        
        for i in range(pages_to_read):
            try:
                page_text = reader.pages[i].extract_text()
                if page_text and page_text.strip():
                    text += page_text + " "
            except Exception as page_err:
                print(f"⚠️ Error reading page {i+1}: {page_err}")
                continue
        
        return text.strip()
    except Exception as e:
        print(f"❌ PDF text extraction error: {e}")
        return ""

# ========== ROUTES ==========

@app.route('/')
def home():
    return jsonify({
        "message": "Digital Library ML API",
        "status": "running",
        "models_loaded": MODELS_LOADED,
        "topic_extraction": "KeyBERT (BERT-based)" if KEYBERT_LOADED else "TF-IDF (fallback)",
        "pages_scanned": 10,
        "endpoints": {
            "/predict": "Upload PDF for course prediction",
            "/predict-with-topics": "Upload PDF for course + topic extraction (KeyBERT)",
            "/extract-topics": "Extract topics only (KeyBERT)",
            "/predict-text": "Test with text input"
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy", 
        "models": MODELS_LOADED,
        "topic_extraction": "keybert" if KEYBERT_LOADED else "tfidf",
        "uptime": time.time() - START_TIME,
        "pages_scanned": 10
    })

@app.route('/info')
def info():
    """Get information about the API configuration"""
    return jsonify({
        "name": "DIGILIB ML API",
        "version": "3.0",
        "topic_extraction": "KeyBERT (BERT-based)",
        "model": "all-MiniLM-L6-v2",
        "pages_scanned": 10,
        "clusters": CLUSTER_NAMES,
        "models_loaded": MODELS_LOADED,
        "keybert_loaded": KEYBERT_LOADED
    })

# ========== PREDICT ENDPOINT ==========
@app.route('/predict', methods=['POST'])
def predict():
    """Returns only course prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file", "cluster": 0, "confidence": 0.5})
        
        file = request.files['file']
        pdf_bytes = file.read()
        
        text = extract_text_from_pdf(pdf_bytes)
        
        if not text.strip():
            text = "academic research thesis"
        
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
            "pages_scanned": min(10, len(PdfReader(io.BytesIO(pdf_bytes)).pages))
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "cluster": 0, "confidence": 0.5})

# ========== PREDICT WITH TOPICS (KEYBERT) ==========
@app.route('/predict-with-topics', methods=['POST'])
def predict_with_topics():
    """Returns course prediction + KeyBERT extracted topics"""
    try:
        if 'file' not in request.files:
            return jsonify({
                "success": False, 
                "error": "No file", 
                "cluster": 0, 
                "confidence": 0.5,
                "topics": []
            })
        
        file = request.files['file']
        pdf_bytes = file.read()
        
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        
        text = extract_text_from_pdf(pdf_bytes)
        
        if not text.strip():
            text = "academic research thesis"
        
        # 1. Course prediction
        X = vectorizer.transform([text])
        cluster = kmeans.predict(X)[0]
        
        closest, distances = pairwise_distances_argmin_min(X, kmeans.cluster_centers_)
        distance = distances[0]
        confidence = max(0.4, min(0.95, 1.0 - (distance / 20.0)))
        
        cluster = max(0, min(5, cluster))
        
        # 2. Topic extraction using KeyBERT
        topics = extract_topics_keybert(text, 8)
        
        return jsonify({
            "success": True,
            "cluster": int(cluster),
            "confidence": float(confidence),
            "cluster_name": CLUSTER_NAMES[cluster],
            "topics": topics,
            "text_length": len(text),
            "total_pages": total_pages,
            "pages_scanned": min(10, total_pages),
            "extraction_method": "keybert" if KEYBERT_LOADED else "tfidf"
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "cluster": 0,
            "confidence": 0.5,
            "topics": []
        })

# ========== EXTRACT TOPICS ONLY ==========
@app.route('/extract-topics', methods=['POST'])
def extract_topics_only():
    """Extract topics using KeyBERT"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file", "topics": []})
        
        file = request.files['file']
        pdf_bytes = file.read()
        
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        
        text = extract_text_from_pdf(pdf_bytes)
        
        if not text.strip():
            text = "academic research thesis"
        
        topics = extract_topics_keybert(text, 10)
        
        return jsonify({
            "success": True,
            "topics": topics,
            "text_length": len(text),
            "total_pages": total_pages,
            "pages_scanned": min(10, total_pages),
            "extraction_method": "keybert" if KEYBERT_LOADED else "tfidf"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "topics": []})

# ========== PREDICT TEXT ==========
@app.route('/predict-text', methods=['POST'])
def predict_text():
    """Test endpoint for text input"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({
                "success": False, 
                "error": "No text", 
                "cluster": 0, 
                "confidence": 0.5,
                "topics": []
            })
        
        # Course prediction
        X = vectorizer.transform([text])
        cluster = kmeans.predict(X)[0]
        
        closest, distances = pairwise_distances_argmin_min(X, kmeans.cluster_centers_)
        distance = distances[0]
        confidence = max(0.4, min(0.95, 1.0 - (distance / 20.0)))
        
        cluster = max(0, min(5, cluster))
        
        # Topic extraction
        topics = extract_topics_keybert(text, 8)
        
        return jsonify({
            "success": True,
            "cluster": int(cluster),
            "confidence": float(confidence),
            "cluster_name": CLUSTER_NAMES[cluster],
            "topics": topics,
            "extraction_method": "keybert" if KEYBERT_LOADED else "tfidf"
        })
        
    except Exception as e:
        return jsonify({
            "success": False, 
            "error": str(e), 
            "cluster": 0, 
            "confidence": 0.5,
            "topics": []
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)