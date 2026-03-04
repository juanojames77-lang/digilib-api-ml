from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
import joblib
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.feature_extraction.text import TfidfVectorizer
import io
import os
import re

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
    print(f"❌ Error loading models: {e}")
    MODELS_LOADED = False

CLUSTER_NAMES = ['BSCS', 'BSED-MATH', 'BSES', 'BSHM', 'BTLED-HE', 'BEED']

# ========== ENHANCED TOPIC EXTRACTION FUNCTION ==========
def extract_topics(text, num_topics=8):
    """
    Extract main topics/keywords from text using TF-IDF
    FIXED: Removes numbers, captures phrases, better filtering
    """
    try:
        # Clean text - remove special characters but keep words
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text)
        
        # Remove standalone numbers (like "2024", "123") but keep words with numbers (like "python3")
        text = re.sub(r'\b\d+\b', '', text)
        
        # If text is too short, return basic topics
        words = text.split()
        if len(words) < 10:
            return [{"topic": "general", "score": 0.5}]
        
        # Expanded academic stop words list
        academic_stop_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'were', 'be', 'been',
            'this', 'that', 'these', 'those', 'it', 'its', 'has', 'have', 'had',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'their', 'they', 'them', 'our', 'we', 'you', 'your', 'my', 'his', 'her',
            'study', 'research', 'thesis', 'paper', 'document', 'analysis',
            'chapter', 'section', 'figure', 'table', 'equation', 'result',
            'method', 'methodology', 'approach', 'conclusion', 'discussion',
            'introduction', 'background', 'literature', 'review', 'objective',
            'purpose', 'data', 'findings', 'based', 'used', 'using', 'also',
            'within', 'between', 'among', 'since', 'until', 'during', 'while'
        ]
        
        # Create TF-IDF vectorizer with improved settings
        tfidf = TfidfVectorizer(
            max_features=30,  # Increased to catch more topics
            stop_words=academic_stop_words,
            ngram_range=(1, 3),  # Capture up to 3-word phrases
            min_df=1,
            max_df=1.0,
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b'  # Only words with letters (no numbers)
        )
        
        # Transform text - this computes TF-IDF on the fly
        tfidf_matrix = tfidf.fit_transform([text])
        
        # Get feature names and scores
        feature_names = tfidf.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Sort by score and get top topics
        top_indices = scores.argsort()[-num_topics:][::-1]
        
        topics = []
        for idx in top_indices:
            if scores[idx] > 0.1:  # Only include meaningful scores
                topic_name = feature_names[idx]
                # Filter out any remaining number-like strings
                if not topic_name.replace('-', '').replace('_', '').isdigit():
                    # Capitalize first letter of each word for better display
                    formatted_topic = ' '.join(word.capitalize() for word in topic_name.split())
                    topics.append({
                        "topic": formatted_topic,
                        "score": float(scores[idx])
                    })
        
        # If we got topics, return them
        if len(topics) >= 3:
            return topics
        
        # Otherwise try with more permissive settings (single words)
        return fallback_topics_enhanced(text, num_topics)
        
    except Exception as e:
        print(f"❌ Topic extraction error: {e}")
        return fallback_topics_enhanced(text, num_topics)

def fallback_topics_enhanced(text, num_topics=5):
    """Enhanced fallback with better filtering"""
    words = text.lower().split()
    
    # Academic/technical words to exclude
    exclude_words = {'study', 'research', 'paper', 'thesis', 'analysis', 'method', 
                     'result', 'conclusion', 'figure', 'table', 'chapter', 'section',
                     'introduction', 'background', 'review', 'objective', 'purpose',
                     'data', 'findings', 'based', 'used', 'using'}
    
    # Common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'were', 'be', 'been',
                  'this', 'that', 'these', 'those', 'it', 'its', 'has', 'have', 'had'}
    
    # Count word frequencies (3+ letter words, excluding stop/exclude words)
    word_counts = {}
    for word in words:
        if (len(word) > 3 and 
            word not in exclude_words and
            word not in stop_words and
            not word.isdigit()):
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    topics = []
    for word, count in sorted_words[:num_topics]:
        # Capitalize the word for display
        formatted_word = word.capitalize()
        topics.append({
            "topic": formatted_word,
            "score": min(0.8, count / 10)  # Normalize score
        })
    
    return topics

# Keep original fallback for backward compatibility
def fallback_topics(text, num_topics=5):
    """Original fallback - kept for compatibility"""
    words = text.lower().split()
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'were', 'be', 'been',
                  'this', 'that', 'these', 'those', 'it', 'its'}
    
    word_counts = {}
    for word in words:
        if word not in stop_words and len(word) > 3:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    topics = []
    for word, count in sorted_words[:num_topics]:
        topics.append({
            "topic": word,
            "score": count / len(words)
        })
    
    return topics

# ========== OPTIONAL: KeyBERT Topic Extraction (Advanced) ==========
# Uncomment this if you install keybert: pip install keybert sentence-transformers
"""
def extract_topics_keybert(text, num_topics=8):
    '''Extract topics using KeyBERT (better quality)'''
    try:
        from keybert import KeyBERT
        
        # Initialize KeyBERT (loads once)
        if not hasattr(extract_topics_keybert, "kw_model"):
            extract_topics_keybert.kw_model = KeyBERT()
        
        # Extract keywords
        keywords = extract_topics_keybert.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=num_topics
        )
        
        topics = []
        for keyword, score in keywords:
            formatted_keyword = ' '.join(word.capitalize() for word in keyword.split())
            topics.append({
                "topic": formatted_keyword,
                "score": float(score)
            })
        
        return topics
        
    except ImportError:
        print("⚠️ KeyBERT not installed, falling back to TF-IDF")
        return extract_topics(text, num_topics)
    except Exception as e:
        print(f"❌ KeyBERT error: {e}")
        return extract_topics(text, num_topics)
"""

# ========== EXTRACT TEXT FROM PDF ==========
def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF bytes"""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        # Read first 3 pages
        for i in range(min(3, len(reader.pages))):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text += page_text + " "
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
        "endpoints": {
            "/predict": "Upload PDF for course prediction",
            "/predict-with-topics": "Upload PDF for course + topic extraction",
            "/extract-topics": "Extract topics only",
            "/predict-text": "Test with text input"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "models": MODELS_LOADED})

# ========== ORIGINAL PREDICT ENDPOINT (unchanged) ==========
@app.route('/predict', methods=['POST'])
def predict():
    """Original endpoint - returns only course prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({
                "success": False, 
                "error": "No file", 
                "cluster": 0, 
                "confidence": 0.5
            })
        
        file = request.files['file']
        pdf_bytes = file.read()
        
        # Extract text
        text = extract_text_from_pdf(pdf_bytes)
        
        if not text.strip():
            text = "academic research thesis"
        
        # Predict using your trained model
        X = vectorizer.transform([text])
        cluster = kmeans.predict(X)[0]
        
        # Calculate confidence
        closest, distances = pairwise_distances_argmin_min(X, kmeans.cluster_centers_)
        distance = distances[0]
        confidence = max(0.4, min(0.95, 1.0 - (distance / 20.0)))
        
        cluster = max(0, min(5, cluster))
        
        return jsonify({
            "success": True,
            "cluster": int(cluster),
            "confidence": float(confidence),
            "cluster_name": CLUSTER_NAMES[cluster]
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "cluster": 0,
            "confidence": 0.5
        })

# ========== PREDICT WITH TOPICS ENDPOINT (UPDATED) ==========
@app.route('/predict-with-topics', methods=['POST'])
def predict_with_topics():
    """Returns both course prediction AND extracted topics (fixed version)"""
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
        
        # Extract text
        text = extract_text_from_pdf(pdf_bytes)
        
        if not text.strip():
            text = "academic research thesis"
        
        # 1. Course prediction (using your existing model)
        X = vectorizer.transform([text])
        cluster = kmeans.predict(X)[0]
        
        closest, distances = pairwise_distances_argmin_min(X, kmeans.cluster_centers_)
        distance = distances[0]
        confidence = max(0.4, min(0.95, 1.0 - (distance / 20.0)))
        
        cluster = max(0, min(5, cluster))
        
        # 2. Topic extraction (UPDATED - no numbers, better phrases)
        topics = extract_topics(text, 8)  # Extract up to 8 topics
        
        return jsonify({
            "success": True,
            "cluster": int(cluster),
            "confidence": float(confidence),
            "cluster_name": CLUSTER_NAMES[cluster],
            "topics": topics,  # Array of {topic, score}
            "text_length": len(text),
            "pages_analyzed": min(3, len(PdfReader(io.BytesIO(pdf_bytes)).pages))
        })
        
    except Exception as e:
        print(f"❌ Error in predict-with-topics: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "cluster": 0,
            "confidence": 0.5,
            "topics": []
        })

# ========== EXTRACT TOPICS ONLY ENDPOINT ==========
@app.route('/extract-topics', methods=['POST'])
def extract_topics_only():
    """Extract topics without course prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file", "topics": []})
        
        file = request.files['file']
        pdf_bytes = file.read()
        
        # Extract text
        text = extract_text_from_pdf(pdf_bytes)
        
        if not text.strip():
            text = "academic research thesis"
        
        # Extract topics (UPDATED)
        topics = extract_topics(text, 10)
        
        return jsonify({
            "success": True,
            "topics": topics,
            "text_length": len(text)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "topics": []
        })

# ========== Test with Text Input ==========
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
        
        # Topic extraction (UPDATED)
        topics = extract_topics(text, 8)
        
        return jsonify({
            "success": True,
            "cluster": int(cluster),
            "confidence": float(confidence),
            "cluster_name": CLUSTER_NAMES[cluster],
            "topics": topics
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