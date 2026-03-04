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
import time

print("🚀 Starting Digital Library ML API on Render...")

app = Flask(__name__)
CORS(app)  # Allow your main app to call this API

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

CLUSTER_NAMES = ['BSCS', 'BSED-MATH', 'BSES', 'BSHM', 'BTLED-HE', 'BEED']

# ========== IMPROVED TOPIC EXTRACTION ==========
def extract_topics(text, num_topics=8):
    """
    Improved topic extraction with better filtering and phrase detection
    """
    try:
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\b\d+\b', '', text)  # Remove numbers
        
        words = text.split()
        if len(words) < 20:
            return [{"topic": "General", "score": 0.5}]
        
        # Academic stop words - expanded
        stop_words = [
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
            'within', 'between', 'among', 'since', 'until', 'during', 'while',
            'however', 'therefore', 'thus', 'hence', 'furthermore', 'moreover',
            'consequently', 'additionally', 'alternatively', 'conversely',
            'significant', 'important', 'necessary', 'potential', 'various',
            'including', 'regarding', 'according', 'following', 'previous'
        ]
        
        # Try with multiple n-gram ranges to catch different phrase types
        topic_sets = []
        
        # Try 2-word phrases first (often most meaningful)
        tfidf_bigram = TfidfVectorizer(
            max_features=25,
            stop_words=stop_words,
            ngram_range=(2, 2),
            min_df=1,
            max_df=1.0,
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b'
        )
        
        bigram_matrix = tfidf_bigram.fit_transform([text])
        bigram_features = tfidf_bigram.get_feature_names_out()
        bigram_scores = bigram_matrix.toarray()[0]
        
        for idx in bigram_scores.argsort()[-num_topics:][::-1]:
            if bigram_scores[idx] > 0.2:
                topic_sets.append({
                    "topic": ' '.join(w.capitalize() for w in bigram_features[idx].split()),
                    "score": float(bigram_scores[idx]),
                    "type": "phrase"
                })
        
        # Also try single words for completeness
        tfidf_unigram = TfidfVectorizer(
            max_features=15,
            stop_words=stop_words,
            ngram_range=(1, 1),
            min_df=1,
            max_df=1.0,
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b'
        )
        
        unigram_matrix = tfidf_unigram.fit_transform([text])
        unigram_features = tfidf_unigram.get_feature_names_out()
        unigram_scores = unigram_matrix.toarray()[0]
        
        for idx in unigram_scores.argsort()[-int(num_topics/2):][::-1]:
            if unigram_scores[idx] > 0.15 and len(unigram_features[idx]) > 3:
                topic_sets.append({
                    "topic": unigram_features[idx].capitalize(),
                    "score": float(unigram_scores[idx]),
                    "type": "word"
                })
        
        # Sort all topics by score and remove duplicates
        topic_sets.sort(key=lambda x: x['score'], reverse=True)
        
        # Deduplicate (if a word appears as part of a phrase, keep the phrase)
        final_topics = []
        seen_words = set()
        
        for topic in topic_sets:
            words_in_topic = set(topic['topic'].lower().split())
            if not any(words_in_topic.intersection(seen_words)):
                final_topics.append({
                    "topic": topic['topic'],
                    "score": round(topic['score'], 3)
                })
                seen_words.update(words_in_topic)
            
            if len(final_topics) >= num_topics:
                break
        
        return final_topics if final_topics else fallback_topics_enhanced(text, num_topics)
        
    except Exception as e:
        print(f"❌ Topic extraction error: {e}")
        return fallback_topics_enhanced(text, num_topics)

def fallback_topics_enhanced(text, num_topics=5):
    """Enhanced fallback with better filtering"""
    words = text.lower().split()
    
    # Common words to exclude
    exclude = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
               'of', 'with', 'by', 'from', 'as', 'is', 'was', 'were', 'be', 'been',
               'study', 'research', 'thesis', 'paper', 'chapter', 'figure', 'table',
               'analysis', 'method', 'result', 'conclusion', 'discussion'}
    
    # Count word frequencies
    word_counts = {}
    for word in words:
        if word not in exclude and len(word) > 3 and not word.isdigit():
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    topics = []
    for word, count in sorted_words[:num_topics]:
        topics.append({
            "topic": word.capitalize(),
            "score": round(min(0.8, count / 10), 3)
        })
    
    return topics

# ========== EXTRACT TEXT FROM PDF - UPDATED TO 10 PAGES ==========
def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF bytes - NOW READING 10 PAGES!"""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        text = ""
        
        # UPDATED: Read first 10 pages (was 3)
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
        
        # If still no text, try reading as plain text (fallback)
        if not text.strip():
            try:
                text = pdf_bytes.decode('utf-8', errors='ignore')[:5000]
                print("⚠️ Using fallback text decoding")
            except:
                text = ""
        
        return text.strip()
    except Exception as e:
        print(f"❌ PDF text extraction error: {e}")
        return ""

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

# ========== ROUTES ==========

@app.route('/')
def home():
    return jsonify({
        "message": "Digital Library ML API",
        "status": "running",
        "models_loaded": MODELS_LOADED,
        "pages_scanned": 10,  # Let users know we now scan 10 pages
        "endpoints": {
            "/predict": "Upload PDF for course prediction",
            "/predict-with-topics": "Upload PDF for course + topic extraction",
            "/extract-topics": "Extract topics only",
            "/predict-text": "Test with text input"
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy", 
        "models": MODELS_LOADED,
        "uptime": time.time() - START_TIME,
        "pages_scanned": 10
    })

# ========== ORIGINAL PREDICT ENDPOINT ==========
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
        
        # Extract text (now using 10 pages)
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
            "cluster_name": CLUSTER_NAMES[cluster],
            "pages_scanned": min(10, len(PdfReader(io.BytesIO(pdf_bytes)).pages))
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
    """Returns both course prediction AND extracted topics (now scanning 10 pages)"""
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
        
        # Get page count for response
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        
        # Extract text (now using 10 pages)
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
        
        # 2. Topic extraction
        topics = extract_topics(text, 8)
        
        return jsonify({
            "success": True,
            "cluster": int(cluster),
            "confidence": float(confidence),
            "cluster_name": CLUSTER_NAMES[cluster],
            "topics": topics,
            "text_length": len(text),
            "total_pages": total_pages,
            "pages_scanned": min(10, total_pages)
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
        
        # Get page count
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        
        # Extract text (now using 10 pages)
        text = extract_text_from_pdf(pdf_bytes)
        
        if not text.strip():
            text = "academic research thesis"
        
        # Extract topics
        topics = extract_topics(text, 10)
        
        return jsonify({
            "success": True,
            "topics": topics,
            "text_length": len(text),
            "total_pages": total_pages,
            "pages_scanned": min(10, total_pages)
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
        
        # Topic extraction
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

# ========== INFO ENDPOINT ==========
@app.route('/info')
def info():
    """Get information about the API configuration"""
    return jsonify({
        "name": "DIGILIB ML API",
        "version": "2.0",
        "pages_scanned": 10,
        "clusters": CLUSTER_NAMES,
        "models_loaded": MODELS_LOADED,
        "topic_extraction": "TF-IDF with 3-word phrases",
        "max_topics": 8
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)