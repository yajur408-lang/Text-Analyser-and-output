"""
Streamlit Interactive Tweet Viewer
Filter and visualize tweets by sentiment and target labels
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
from scipy.sparse import hstack, csr_matrix
from PIL import Image
import io

# Try to import pytesseract for OCR
OCR_AVAILABLE = False
TESSERACT_CONFIGURED = False

try:
    import pytesseract
    OCR_AVAILABLE = True
    
    # Manual Tesseract configuration (set your installation path here)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # Try to configure Tesseract path for Windows (common installation location)
    import platform
    if platform.system() == 'Windows':
        # First check if manual path exists
        if os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            TESSERACT_CONFIGURED = True
        else:
            # Try other common paths
            common_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
            ]
            for path in common_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    TESSERACT_CONFIGURED = True
                    break
    
    # Test if Tesseract is accessible
    if TESSERACT_CONFIGURED or platform.system() != 'Windows':
        try:
            pytesseract.get_tesseract_version()
            TESSERACT_CONFIGURED = True
        except:
            TESSERACT_CONFIGURED = False
except ImportError:
    OCR_AVAILABLE = False
except Exception:
    # pytesseract installed but Tesseract engine not found
    OCR_AVAILABLE = True
    TESSERACT_CONFIGURED = False

# ML model imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.base import BaseEstimator, TransformerMixin
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
    import xgboost as xgb
    SKLEARN_AVAILABLE = True
except ImportError as e:
    SKLEARN_AVAILABLE = False

# Google Generative AI imports
try:
    import google.generativeai as genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

# Try to import torch, but make it optional
try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError, RuntimeError) as e:
    TORCH_AVAILABLE = False
    # Don't show warning on initial load - only show when needed
    # The app works fine without PyTorch for dataset viewing

# Try to import transformers
try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError) as e:
    TRANSFORMERS_AVAILABLE = False
    # Don't show warning on initial load - only show when needed

# Set page config
st.set_page_config(
    page_title="Tweet Sentiment Viewer",
    page_icon="🐦",
    layout="wide"
)

# Color mapping for sentiments
SENTIMENT_COLORS = {
    'Positive': '#28a745',  # Green
    'Negative': '#dc3545',  # Red
    'Neutral': '#ffc107',   # Yellow
    'Irrelevant': '#6c757d' # Gray
}

TARGET_COLORS = {
    1: '#28a745',  # Green for positive
    0: '#dc3545'   # Red for negative/neutral
}

# Extended sentiment colors
EXTENDED_SENTIMENT_COLORS = {
    'Sarcastic': '#ff6b6b',  # Red-pink
    'Playful': '#4ecdc4',    # Teal
    'Funny': '#ffe66d',      # Yellow
    'Flirty': '#ff9ff3',     # Pink
    'Angry': '#ee5a6f',      # Dark red
    'Sadness': '#6c5ce7',    # Purple
    'Dark Humour': '#2d3436',  # Dark gray/black
    'Neutral': '#95a5a6',    # Gray
    'Positive': '#28a745',   # Green
    'Negative': '#dc3545'    # Red
}


def extract_text_from_quotes(text):
    """Extract text inside quotation marks"""
    if pd.isna(text):
        return ""
    text = str(text)
    quoted_text = re.findall(r'"([^"]*)"', text)
    if quoted_text:
        return ' '.join(quoted_text)
    return text


def clean_text(text):
    """Clean text for display"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    text = ' '.join(text.split())
    return text.strip()


def clean_text_for_ml(text):
    """Clean text for ML processing (same as training)"""
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^\w\s.,!?]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.lower().strip()


def extract_text_from_image(image):
    """Extract text from image using OCR"""
    if not OCR_AVAILABLE:
        return None, "pytesseract is not installed. Install with: pip install pytesseract"
    
    try:
        # Convert to PIL Image if needed
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif not isinstance(image, Image.Image):
            image = Image.open(image)
        
        # Extract text using pytesseract
        extracted_text = pytesseract.image_to_string(image)
        
        if not extracted_text or len(extracted_text.strip()) == 0:
            return None, "No text could be extracted from the image. Please ensure the image contains readable text."
        
        return extracted_text.strip(), None
    except Exception as e:
        return None, f"Error extracting text from image: {str(e)}"


# ML Feature Extractors (same as training script)
class Word2VecTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for Word2Vec embeddings"""
    
    def __init__(self, vector_size=100, window=5, min_count=2):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
    
    def fit(self, X, y=None):
        """Train Word2Vec model"""
        tokenized = [simple_preprocess(text, deacc=True) for text in X]
        self.model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            sg=0
        )
        return self
    
    def transform(self, X):
        """Generate embeddings"""
        tokenized = [simple_preprocess(text, deacc=True) for text in X]
        embeddings = []
        for tokens in tokenized:
            if tokens:
                word_vectors = [
                    self.model.wv[word] 
                    for word in tokens 
                    if word in self.model.wv
                ]
                if word_vectors:
                    embeddings.append(np.mean(word_vectors, axis=0))
                else:
                    embeddings.append(np.zeros(self.vector_size))
            else:
                embeddings.append(np.zeros(self.vector_size))
        return np.array(embeddings)


class SentimentFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract sentiment features using VADER and TextBlob"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract sentiment features"""
        features = []
        for text in X:
            # VADER scores
            vader_scores = self.vader.polarity_scores(str(text))
            # TextBlob scores
            blob = TextBlob(str(text))
            features.append([
                vader_scores['compound'],
                vader_scores['pos'],
                vader_scores['neu'],
                vader_scores['neg'],
                blob.sentiment.polarity,
                blob.sentiment.subjectivity
            ])
        return np.array(features)


def combine_features(tfidf_features, sentiment_features, w2v_features):
    """Combine different feature types"""
    # Convert sentiment and w2v to sparse matrices for efficient stacking
    sentiment_sparse = csr_matrix(sentiment_features)
    w2v_sparse = csr_matrix(w2v_features)
    
    # Combine all features
    combined = hstack([tfidf_features, sentiment_sparse, w2v_sparse]).toarray()
    return combined


@st.cache_resource
def load_sentiment_models():
    """Load sentiment analysis models"""
    models = {}
    
    if not TRANSFORMERS_AVAILABLE:
        return models  # Return empty dict, error will be shown in UI
    
    # Check if torch is actually available (transformers needs it)
    if not TORCH_AVAILABLE:
        st.warning("⚠️ PyTorch is required for sentiment models. Models will not load.")
        return models
    
    # Determine device (use CPU if torch not available)
    device = -1  # Default to CPU (-1 means CPU in transformers)
    if TORCH_AVAILABLE:
        try:
            # Import torch locally to avoid NameError
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except (NameError, ImportError, AttributeError):
            device = -1
    
    try:
        with st.spinner("Loading Hugging Face Twitter-RoBERTa model..."):
            models['twitter_roberta'] = hf_pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True,
                device=device
            )
    except Exception as e:
        st.warning(f"Could not load Twitter-RoBERTa: {e}")
        models['twitter_roberta'] = None
    
    try:
        with st.spinner("Loading FinBERT model..."):
            models['finbert'] = hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                return_all_scores=True,
                device=device
            )
    except Exception as e:
        st.warning(f"Could not load FinBERT: {e}")
        models['finbert'] = None
    
    try:
        with st.spinner("Loading BERT model..."):
            models['bert'] = hf_pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                return_all_scores=True,
                device=device
            )
    except Exception as e:
        st.warning(f"Could not load BERT: {e}")
        models['bert'] = None
    
    return models


def analyze_sentiment_with_model(text, model, model_name):
    """Analyze sentiment using a specific model"""
    if model is None:
        return None, None
    
    try:
        # Truncate text if too long
        max_length = 512
        text_truncated = text[:max_length] if len(text) > max_length else text
        
        results = model(text_truncated)
        
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], list):
                # Multiple scores format
                scores = {item['label']: item['score'] for item in results[0]}
            else:
                # Single result format
                scores = {results[0]['label']: results[0]['score']}
            
            return scores, results
        return None, None
    except Exception as e:
        st.error(f"Error with {model_name}: {e}")
        return None, None


def map_to_extended_sentiment(scores_dict, model_name):
    """Map model scores to extended sentiment categories"""
    if scores_dict is None:
        return "Neutral", 0.0
    
    # Normalize labels
    scores_lower = {k.lower(): v for k, v in scores_dict.items()}
    
    # Get dominant sentiment
    dominant_label = max(scores_dict.items(), key=lambda x: x[1])[0].lower()
    dominant_score = max(scores_dict.values())
    
    # Map based on model and scores
    if model_name == 'twitter_roberta':
        # Labels: LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
        neg_score = scores_lower.get('label_0', scores_lower.get('negative', 0))
        pos_score = scores_lower.get('label_2', scores_lower.get('positive', 0))
        neu_score = scores_lower.get('label_1', scores_lower.get('neutral', 0))
        
        if neg_score > 0.6:
            if neg_score > 0.85:
                return "Angry", neg_score
            elif neg_score > 0.75:
                return "Sadness", neg_score
            elif neg_score > 0.65:
                # Check if it might be dark humour (negative but with some neutral/positive elements)
                if neu_score > 0.2 or pos_score > 0.15:
                    return "Dark Humour", neg_score
                return "Sarcastic", neg_score
            else:
                return "Sarcastic", neg_score
        elif pos_score > 0.6:
            if pos_score > 0.8:
                return "Playful", pos_score
            else:
                return "Funny", pos_score
        else:
            return "Neutral", neu_score
    
    elif model_name == 'finbert':
        # FinBERT: positive, negative, neutral
        pos_score = scores_lower.get('positive', 0)
        neg_score = scores_lower.get('negative', 0)
        neu_score = scores_lower.get('neutral', 0)
        
        if pos_score > 0.6:
            if pos_score > 0.75:
                return "Playful", pos_score
            else:
                return "Funny", pos_score
        elif neg_score > 0.6:
            if neg_score > 0.8:
                return "Angry", neg_score
            elif neg_score > 0.7:
                return "Sadness", neg_score
            elif neg_score > 0.65:
                # Check if it might be dark humour (negative but with some neutral/positive elements)
                if neu_score > 0.2 or pos_score > 0.15:
                    return "Dark Humour", neg_score
                return "Sarcastic", neg_score
            else:
                return "Sarcastic", neg_score
        else:
            return "Neutral", neu_score
    
    elif model_name == 'bert':
        # BERT multilingual: 1-5 star ratings
        star_5 = scores_lower.get('5 stars', scores_lower.get('5', 0))
        star_4 = scores_lower.get('4 stars', scores_lower.get('4', 0))
        star_3 = scores_lower.get('3 stars', scores_lower.get('3', 0))
        star_2 = scores_lower.get('2 stars', scores_lower.get('2', 0))
        star_1 = scores_lower.get('1 star', scores_lower.get('1', 0))
        
        if star_5 > 0.4:
            return "Funny", star_5
        elif star_4 > 0.4:
            return "Playful", star_4
        elif star_1 > 0.4:
            return "Angry", star_1
        elif star_2 > 0.4:
            # 2-star ratings could be sadness or dark humour
            # If star_3 is also high, might be dark humour (mixed sentiment)
            if star_3 > 0.3:
                return "Dark Humour", star_2
            return "Sadness", star_2
        else:
            return "Neutral", star_3
    
    return "Neutral", dominant_score


def detect_sentiment_category(text, scores_dict):
    """Detect specific sentiment categories from text and scores"""
    text_lower = text.lower()
    
    # Extended keyword-based detection
    sarcastic_keywords = ['sarcasm', 'sarcastic', 'obviously', 'sure', 'right', 'yeah right', 
                         'totally', 'of course', 'clearly', 'obviously', 'duh']
    playful_keywords = ['haha', 'lol', 'hehe', 'fun', 'play', 'joke', 'joking', 'teasing',
                       'prank', 'giggling', 'wink', '😉', '😜']
    funny_keywords = ['funny', 'hilarious', 'laugh', 'comedy', 'humor', 'lmao', 'rofl',
                     'hysterical', 'comical', 'amusing', '😂', '🤣']
    flirty_keywords = ['flirt', 'cute', 'hot', 'sexy', 'attractive', 'beautiful', 'gorgeous',
                      'handsome', 'charming', 'seductive', 'wink', '😘', '😍', '💋']
    angry_keywords = ['angry', 'mad', 'furious', 'rage', 'hate', 'stupid', 'idiot', 'damn',
                     'annoyed', 'irritated', 'frustrated', 'pissed', '😠', '😡']
    sadness_keywords = ['sad', 'sadness', 'depressed', 'depression', 'unhappy', 'upset', 'down',
                       'melancholy', 'gloomy', 'sorrow', 'grief', 'heartbroken', 'crying',
                       'tears', 'lonely', 'loneliness', 'disappointed', 'hopeless', '😢', '😭', '💔']
    dark_humour_keywords = ['dark humor', 'dark humour', 'dark comedy', 'gallows humor', 'gallows humour',
                           'morbid', 'morbid humor', 'twisted', 'twisted humor', 'cynical', 'cynical humor',
                           'black comedy', 'black humor', 'edgy', 'edgy humor', 'deadpan', 'macabre',
                           'sick humor', 'sick humour', 'grim humor', 'bleak humor', 'ironic death',
                           'death joke', 'tragedy joke', 'morbid joke', 'dark joke', 'twisted joke']
    
    # Check for keywords with priority (Dark Humour first as it's most specific)
    if any(kw in text_lower for kw in dark_humour_keywords):
        return "Dark Humour"
    elif any(kw in text_lower for kw in angry_keywords):
        return "Angry"
    elif any(kw in text_lower for kw in sadness_keywords):
        return "Sadness"
    elif any(kw in text_lower for kw in sarcastic_keywords):
        return "Sarcastic"
    elif any(kw in text_lower for kw in funny_keywords):
        return "Funny"
    elif any(kw in text_lower for kw in flirty_keywords):
        return "Flirty"
    elif any(kw in text_lower for kw in playful_keywords):
        return "Playful"
    
    # Fallback to model-based detection
    return None  # Will use model mapping


def analyze_sentiment_with_google_genai(text, api_key=None, model_name="gemini-2.5-pro"):
    """Analyze sentiment using Google Generative AI"""
    if not GOOGLE_GENAI_AVAILABLE:
        return None, "Google Generative AI is not installed. Install with: pip install google-generativeai"
    
    try:
        # Use API key from environment or parameter
        if api_key:
            genai.configure(api_key=api_key)
        elif os.getenv('GOOGLE_API_KEY'):
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        else:
            return None, "Google API key not found. Set GOOGLE_API_KEY environment variable or provide api_key parameter."
        
        # Create the model with specified model name
        try:
            model = genai.GenerativeModel(model_name)
        except Exception as e:
            # If specified model fails, try alternatives
            fallback_models = ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro']
            model = None
            for fallback_name in fallback_models:
                if fallback_name != model_name:
                    try:
                        model = genai.GenerativeModel(fallback_name)
                        model_name = fallback_name  # Update to the working model
                        break
                    except:
                        continue
            
            if model is None:
                return None, f"Could not create model '{model_name}'. Error: {str(e)}. Please try a different model name (gemini-2.5-pro, gemini-2.5-flash, gemini-1.5-pro, or gemini-1.5-flash)."
        
        # Create prompt for sentiment analysis
        prompt = f"""Analyze the sentiment of the following text and classify it into one of these categories:
- Sarcastic: Sarcastic or ironic content
- Playful: Light-hearted, fun content
- Funny: Humorous, comedic content
- Flirty: Flirtatious or romantic content
- Angry: Angry, frustrated, or negative content
- Sadness: Sad, depressed, or melancholic content
- Dark Humour: Dark, morbid, or twisted humor about serious topics
- Neutral: Neutral or balanced content

Text: "{text}"

Respond with ONLY the category name and a confidence score (0.0 to 1.0) in this format:
Category: [category name]
Confidence: [score]
"""
        
        # Generate response
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Parse response
        category = "Neutral"
        confidence = 0.5
        
        lines = response_text.split('\n')
        for line in lines:
            if 'Category:' in line or 'category:' in line.lower():
                category = line.split(':')[-1].strip()
            elif 'Confidence:' in line or 'confidence:' in line.lower():
                try:
                    confidence = float(line.split(':')[-1].strip())
                except:
                    confidence = 0.5
        
        # Map to our sentiment categories
        category_map = {
            'sarcastic': 'Sarcastic',
            'playful': 'Playful',
            'funny': 'Funny',
            'flirty': 'Flirty',
            'angry': 'Angry',
            'sadness': 'Sadness',
            'sad': 'Sadness',
            'dark humour': 'Dark Humour',
            'dark humor': 'Dark Humour',
            'neutral': 'Neutral'
        }
        
        category = category_map.get(category.lower(), category)
        
        return {
            'sentiment': category,
            'confidence': confidence,
            'raw_response': response_text
        }, None
        
    except Exception as e:
        return None, f"Error analyzing with Google GenAI: {str(e)}"


@st.cache_resource
def load_ml_models():
    """Load or train ML models (Google GenAI, Random Forest, XGBoost)"""
    if not SKLEARN_AVAILABLE:
        return None, None, None, None, None, None
    
    # Try to load pre-trained models
    models = {}
    feature_extractors = {}
    
    try:
        # Check if models are saved
        if os.path.exists('ml_models.pkl'):
            with open('ml_models.pkl', 'rb') as f:
                saved_data = pickle.load(f)
                models = saved_data.get('models', {})
                feature_extractors = saved_data.get('extractors', {})
                if models and feature_extractors:
                    st.success("✅ Loaded pre-trained ML models")
                    return (
                        models.get('Random Forest'),
                        models.get('XGBoost'),
                        feature_extractors.get('tfidf'),
                        feature_extractors.get('sentiment'),
                        feature_extractors.get('w2v')
                    )
    except Exception as e:
        st.warning(f"Could not load saved models: {e}")
    
    # If models not found, train them on the fly
    st.info("Training ML models on training data... This may take a few minutes.")
    
    try:
        # Load training data (includes flirting datasets)
        train_df = pd.read_csv('twitter_training.csv', header=None,
                              names=['id', 'topic', 'sentiment', 'tweet'])
        
        # Add flirting_rated.csv
        try:
            flirting_rated = pd.read_csv('flirting_rated.csv')
            # ADJUST THIS MAPPING if labels are inaccurate
            polarity_mapping = {0: 'Neutral', 1: 'Positive'}  # Adjust if needed
            
            flirting_rated_processed = pd.DataFrame({
                'id': range(len(train_df), len(train_df) + len(flirting_rated)),
                'topic': 'Flirting',
                'sentiment': flirting_rated['polarity'].map(polarity_mapping),
                'tweet': flirting_rated['final_messages']
            })
            train_df = pd.concat([train_df, flirting_rated_processed], ignore_index=True)
        except Exception:
            pass
        
        # Add flirtation_dataset.csv
        try:
            flirtation_data = pd.read_csv('flirtation_dataset.csv')
            # ADJUST THIS THRESHOLD if labels are inaccurate
            def map_score_to_sentiment(score):
                if score >= 50:  # Adjust threshold (try 40, 50, 60, 70)
                    return 'Positive'
                else:
                    return 'Neutral'
            
            flirtation_processed = pd.DataFrame({
                'id': range(len(train_df), len(train_df) + len(flirtation_data)),
                'topic': 'Flirting',
                'sentiment': flirtation_data['score'].apply(map_score_to_sentiment),
                'tweet': flirtation_data['message']
            })
            train_df = pd.concat([train_df, flirtation_processed], ignore_index=True)
        except Exception:
            pass
        
        # Process tweets
        train_df['tweet_text'] = train_df['tweet'].apply(extract_text_from_quotes)
        train_df['cleaned_text'] = train_df['tweet_text'].apply(clean_text_for_ml)
        
        # Encode target
        train_df['target'] = (train_df['sentiment'] == 'Positive').astype(int)
        
        # Prepare data
        X_train_text = train_df['cleaned_text'].values
        y_train = train_df['target'].values
        
        # Extract features
        with st.spinner("Extracting features..."):
            # TF-IDF
            tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2, max_df=0.95)
            X_train_tfidf = tfidf.fit_transform(X_train_text)
            
            # Sentiment features
            sentiment_extractor = SentimentFeatureExtractor()
            X_train_sentiment = sentiment_extractor.fit_transform(X_train_text)
            
            # Word2Vec
            w2v = Word2VecTransformer(vector_size=100)
            X_train_w2v = w2v.fit_transform(X_train_text)
            
            # Combine features
            X_train_processed = combine_features(X_train_tfidf, X_train_sentiment, X_train_w2v)
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_processed)
        
        # Train models
        with st.spinner("Training models..."):
            # Random Forest
            rf_model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
            ])
            rf_model.fit(X_train_scaled, y_train)
            models['Random Forest'] = rf_model
            
            # XGBoost
            xgb_model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'))
            ])
            xgb_model.fit(X_train_scaled, y_train)
            models['XGBoost'] = xgb_model
        
        # Save models for future use
        try:
            with open('ml_models.pkl', 'wb') as f:
                pickle.dump({
                    'models': models,
                    'extractors': {
                        'tfidf': tfidf,
                        'sentiment': sentiment_extractor,
                        'w2v': w2v,
                        'scaler': scaler
                    }
                }, f)
        except Exception as e:
            st.warning(f"Could not save models: {e}")
        
        st.success("✅ ML models trained successfully!")
        return (
            models['Random Forest'],
            models['XGBoost'],
            tfidf,
            sentiment_extractor,
            w2v
        )
        
    except Exception as e:
        st.error(f"Error training ML models: {e}")
        return None, None, None, None, None


def predict_with_ml_models(text, rf_model, xgb_model, tfidf, sentiment_extractor, w2v):
    """Predict sentiment using ML models"""
    if not all([rf_model, xgb_model, tfidf, sentiment_extractor, w2v]):
        return {}
    
    # Clean text
    cleaned_text = clean_text_for_ml(text)
    
    # Extract features
    try:
        # TF-IDF
        text_tfidf = tfidf.transform([cleaned_text])
        
        # Sentiment features
        text_sentiment = sentiment_extractor.transform([cleaned_text])
        
        # Word2Vec
        text_w2v = w2v.transform([cleaned_text])
        
        # Combine features
        text_features = combine_features(text_tfidf, text_sentiment, text_w2v)
        
        # Scale (scaler is in the pipeline, but we need to scale before)
        # Actually, the scaler is in the pipeline, so we don't need to scale here
        
        results = {}
        
        # Predict with each model
        for name, model in [('Random Forest', rf_model), 
                           ('XGBoost', xgb_model)]:
            try:
                # Get prediction probability
                proba = model.predict_proba(text_features)[0]
                prediction = model.predict(text_features)[0]
                
                # Map to sentiment categories
                if prediction == 1:
                    sentiment = "Positive"
                    confidence = proba[1]
                    # Further categorize based on confidence
                    if confidence > 0.8:
                        sentiment = "Playful"
                    elif confidence > 0.6:
                        sentiment = "Funny"
                else:
                    sentiment = "Negative"
                    confidence = proba[0]
                    # Further categorize
                    if confidence > 0.85:
                        sentiment = "Angry"
                    elif confidence > 0.7:
                        sentiment = "Sadness"
                    elif confidence > 0.6:
                        # Could be dark humour if there's some ambiguity
                        sentiment = "Dark Humour"
                    elif confidence > 0.55:
                        sentiment = "Sarcastic"
                    else:
                        sentiment = "Neutral"
                
                # Check for keyword overrides
                keyword_sentiment = detect_sentiment_category(text, None)
                if keyword_sentiment:
                    sentiment = keyword_sentiment
                
                results[name] = {
                    'sentiment': sentiment,
                    'confidence': float(confidence),
                    'prediction': int(prediction),
                    'probabilities': {
                        'Negative': float(proba[0]),
                        'Positive': float(proba[1]) if len(proba) > 1 else 0.0
                    }
                }
            except Exception as e:
                st.warning(f"Error with {name}: {e}")
        
        return results
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return {}


def analyze_text_sentiment(text):
    """Analyze text sentiment using all available models"""
    if not text or len(text.strip()) == 0:
        return None
    
    results = {}
    
    # Transformer models
    if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
        models = load_sentiment_models()
        for model_name, model in models.items():
            if model is not None:
                scores, raw_results = analyze_sentiment_with_model(text, model, model_name)
                if scores:
                    sentiment, confidence = map_to_extended_sentiment(scores, model_name)
                    
                    # Override with keyword detection if applicable
                    keyword_sentiment = detect_sentiment_category(text, scores)
                    if keyword_sentiment:
                        sentiment = keyword_sentiment
                    
                    results[model_name] = {
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'scores': scores,
                        'raw': raw_results
                    }
    
    # ML models
    if SKLEARN_AVAILABLE:
        rf_model, xgb_model, tfidf, sentiment_extractor, w2v = load_ml_models()
        ml_results = predict_with_ml_models(text, rf_model, xgb_model, 
                                           tfidf, sentiment_extractor, w2v)
        results.update(ml_results)
    
    return results


@st.cache_data
def load_data():
    """Load and process data, including flirting datasets"""
    try:
        train_df = pd.read_csv('twitter_training.csv', header=None,
                              names=['id', 'topic', 'sentiment', 'tweet'])
        val_df = pd.read_csv('twitter_validation.csv', header=None,
                            names=['id', 'topic', 'sentiment', 'tweet'])
        
        # Load and add flirting_rated.csv
        try:
            flirting_rated = pd.read_csv('flirting_rated.csv')
            # ADJUST THIS MAPPING if labels are inaccurate
            # Current: 0 = Neutral, 1 = Positive
            # If reversed, change to: {0: 'Positive', 1: 'Neutral'}
            polarity_mapping = {0: 'Neutral', 1: 'Positive'}
            
            flirting_rated_processed = pd.DataFrame({
                'id': range(len(train_df), len(train_df) + len(flirting_rated)),
                'topic': 'Flirting',
                'sentiment': flirting_rated['polarity'].map(polarity_mapping),
                'tweet': flirting_rated['final_messages']
            })
            train_df = pd.concat([train_df, flirting_rated_processed], ignore_index=True)
        except Exception as e:
            st.warning(f"Could not load flirting_rated.csv: {e}")
        
        # Load and add flirtation_dataset.csv
        try:
            flirtation_data = pd.read_csv('flirtation_dataset.csv')
            # ADJUST THIS THRESHOLD if labels are inaccurate
            # Current: score >= 50 = Positive, < 50 = Neutral
            # Try different thresholds: 40, 50, 60, 70
            def map_score_to_sentiment(score):
                if score >= 50:  # Adjust this threshold
                    return 'Positive'
                else:
                    return 'Neutral'
            
            flirtation_processed = pd.DataFrame({
                'id': range(len(train_df), len(train_df) + len(flirtation_data)),
                'topic': 'Flirting',
                'sentiment': flirtation_data['score'].apply(map_score_to_sentiment),
                'tweet': flirtation_data['message']
            })
            train_df = pd.concat([train_df, flirtation_processed], ignore_index=True)
        except Exception as e:
            st.warning(f"Could not load flirtation_dataset.csv: {e}")
        
        # Process tweets
        train_df['tweet_text'] = train_df['tweet'].apply(extract_text_from_quotes)
        train_df['cleaned_text'] = train_df['tweet_text'].apply(clean_text)
        val_df['tweet_text'] = val_df['tweet'].apply(extract_text_from_quotes)
        val_df['cleaned_text'] = val_df['tweet_text'].apply(clean_text)
        
        # Add target column
        train_df['target'] = (train_df['sentiment'] == 'Positive').astype(int)
        val_df['target'] = (val_df['sentiment'] == 'Positive').astype(int)
        
        return train_df, val_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


def display_tweet_row(row, show_cleaned=True):
    """Display a single tweet row with color coding"""
    sentiment = row['sentiment']
    target = row['target']
    tweet_text = row['cleaned_text'] if show_cleaned else row['tweet_text']
    
    # Color based on sentiment
    bg_color = SENTIMENT_COLORS.get(sentiment, '#ffffff')
    text_color = '#ffffff' if sentiment in ['Positive', 'Negative'] else '#000000'
    
    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            color: {text_color};
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        ">
            <strong>Sentiment:</strong> {sentiment} | 
            <strong>Target:</strong> {target} | 
            <strong>Topic:</strong> {row.get('topic', 'N/A')}
            <br>
            <strong>Tweet:</strong> {tweet_text[:200]}{'...' if len(tweet_text) > 200 else ''}
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    """Main Streamlit app"""
    st.title("🐦 Tweet Sentiment Viewer")
    st.markdown("Interactive viewer for tweet sentiment analysis")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Dataset Viewer", "🔍 Sentiment Analyzer"])
    
    with tab1:
        dataset_viewer()
    
    with tab2:
        sentiment_analyzer()


def sentiment_analyzer():
    """Sentiment analysis tab for custom text input"""
    st.header("🔍 Multi-Model Sentiment Analysis")
    st.markdown("Analyze sentiment using Transformer models (Twitter-RoBERTa, FinBERT, BERT) and ML models (Google GenAI, Random Forest, XGBoost)")
    
    # Check if at least one type of model is available
    if not TRANSFORMERS_AVAILABLE and not SKLEARN_AVAILABLE:
        st.error("""
        **Transformers library is not available.**
        
        To use the sentiment analyzer, please install the required packages.
        """)
        
        with st.expander("📋 Installation Instructions"):
            st.markdown("""
            ### Step 1: Install Visual C++ Redistributables (Required for PyTorch)
            
            Download and install from: https://aka.ms/vs/17/release/vc_redist.x64.exe
            
            Then restart your computer.
            
            ### Step 2: Install PyTorch and Transformers
            
            Run these commands in your terminal:
            ```bash
            pip uninstall torch torchvision torchaudio -y
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            pip install transformers
            ```
            
            ### Step 3: Verify Installation
            
            ```bash
            python -c "import torch; print('PyTorch works!')"
            ```
            
            ### Alternative: Use Without PyTorch
            
            The **Dataset Viewer** tab works perfectly without PyTorch!
            You can view, filter, and export all your data.
            """)
        
        st.info("💡 **Tip**: The Dataset Viewer tab works great without PyTorch. You can use it to explore your data while fixing PyTorch separately.")
        return
    
    # Input method selection
    input_method = st.radio(
        "Input Method:",
        ["📝 Text Input", "📷 Image Upload (OCR)"],
        horizontal=True
    )
    
    text_input = ""
    
    if input_method == "📝 Text Input":
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste your text here...",
            help="Enter any text to get sentiment analysis from multiple models"
        )
    else:
        # Image upload with OCR
        st.subheader("📷 Upload Image for OCR")
        
        if not OCR_AVAILABLE:
            st.warning("⚠️ OCR package (pytesseract) is not installed.")
            st.code("pip install pytesseract Pillow")
            st.info("""
            **Note**: You also need to install Tesseract OCR engine:
            - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
            - **Mac**: `brew install tesseract`
            - **Linux**: `sudo apt-get install tesseract-ocr`
            """)
        elif not TESSERACT_CONFIGURED:
            st.error("⚠️ Tesseract OCR engine not found!")
            with st.expander("📋 How to Install Tesseract OCR"):
                st.markdown("""
                ### Windows Installation:
                
                1. **Download Tesseract:**
                   - Go to: https://github.com/UB-Mannheim/tesseract/wiki
                   - Download: `tesseract-ocr-w64-setup-5.x.x.exe` (64-bit)
                
                2. **Install Tesseract:**
                   - Run the installer
                   - **Note the installation path** (usually `C:\\Program Files\\Tesseract-OCR`)
                   - Or run: `powershell -ExecutionPolicy Bypass -File install_tesseract_windows.ps1`
                
                3. **Configure (if auto-detection fails):**
                   Add this to the top of viewer.py after the imports:
                   ```python
                   import pytesseract
                   pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
                   ```
                
                4. **Restart Streamlit app**
                
                ### Quick Fix:
                Run: `powershell -ExecutionPolicy Bypass -File install_tesseract_windows.ps1`
                """)
        else:
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'],
                help="Upload an image containing text to extract and analyze"
            )
            
            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Extract text from image
                with st.spinner("Extracting text from image using OCR..."):
                    extracted_text, error = extract_text_from_image(image)
                    
                    if error:
                        st.error(f"❌ {error}")
                    elif extracted_text:
                        st.success("✅ Text extracted successfully!")
                        st.text_area(
                            "Extracted Text:",
                            value=extracted_text,
                            height=150,
                            help="Review and edit the extracted text if needed"
                        )
                        text_input = extracted_text
                    else:
                        st.warning("⚠️ No text could be extracted from the image.")
    
    # Model selection
    st.sidebar.header("Model Selection")
    
    # Transformer models
    if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
        st.sidebar.subheader("Transformer Models")
        use_twitter_roberta = st.sidebar.checkbox("Twitter-RoBERTa", value=True)
        use_finbert = st.sidebar.checkbox("FinBERT", value=True)
        use_bert = st.sidebar.checkbox("BERT Multilingual", value=True)
    else:
        use_twitter_roberta = False
        use_finbert = False
        use_bert = False
    
    # ML models
    st.sidebar.subheader("ML Models")
    
    # Google GenAI
    if GOOGLE_GENAI_AVAILABLE:
        use_google_genai = st.sidebar.checkbox("Google GenAI", value=True)
        google_api_key = st.sidebar.text_input(
            "Google API Key (optional)",
            type="password",
            help="Leave empty to use GOOGLE_API_KEY environment variable"
        )
        google_model_name = st.sidebar.selectbox(
            "Model",
            ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
            index=0,
            help="Select the Google GenAI model to use"
        )
    else:
        use_google_genai = False
        google_api_key = None
        google_model_name = "gemini-2.5-pro"
        st.sidebar.warning("Google GenAI requires google-generativeai. Install with: pip install google-generativeai")
    
    # Traditional ML models
    if SKLEARN_AVAILABLE:
        use_rf = st.sidebar.checkbox("Random Forest", value=True)
        use_xgb = st.sidebar.checkbox("XGBoost", value=True)
    else:
        use_rf = False
        use_xgb = False
        st.sidebar.warning("ML models require scikit-learn. Install with: pip install scikit-learn")
    
    # Show OCR status
    if input_method == "📷 Image Upload (OCR)":
        if not OCR_AVAILABLE:
            st.error("⚠️ OCR package (pytesseract) is not installed.")
            st.code("pip install pytesseract Pillow")
        elif not TESSERACT_CONFIGURED:
            st.error("⚠️ Tesseract OCR engine not found!")
            with st.expander("📋 How to Install Tesseract OCR"):
                st.markdown("""
                ### Windows Installation:
                
                1. **Download Tesseract:**
                   - Go to: https://github.com/UB-Mannheim/tesseract/wiki
                   - Download: `tesseract-ocr-w64-setup-5.x.x.exe` (64-bit)
                
                2. **Install Tesseract:**
                   - Run the installer
                   - **Note the installation path** (usually `C:\\Program Files\\Tesseract-OCR`)
                   - Or run: `powershell -ExecutionPolicy Bypass -File install_tesseract_windows.ps1`
                
                3. **Configure (if auto-detection fails):**
                   Add this to the top of viewer.py:
                   ```python
                   import pytesseract
                   pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
                   ```
                
                4. **Restart Streamlit app**
                
                ### Alternative: Quick Fix Script
                Run: `powershell -ExecutionPolicy Bypass -File install_tesseract_windows.ps1`
                """)
        else:
            st.success("✅ OCR is ready! Upload an image to extract text.")
    
    if st.button("Analyze Sentiment", type="primary"):
        if not text_input or len(text_input.strip()) == 0:
            if input_method == "📷 Image Upload (OCR)":
                st.warning("Please upload an image with text or switch to text input.")
            else:
                st.warning("Please enter some text to analyze.")
            return
        
        with st.spinner("Analyzing sentiment..."):
            results = {}
            
            # Transformer models
            if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
                models = load_sentiment_models()
                
                if use_twitter_roberta and models.get('twitter_roberta'):
                    scores, raw = analyze_sentiment_with_model(
                        text_input, models['twitter_roberta'], 'twitter_roberta'
                    )
                    if scores:
                        sentiment, confidence = map_to_extended_sentiment(scores, 'twitter_roberta')
                        keyword_sentiment = detect_sentiment_category(text_input, scores)
                        if keyword_sentiment:
                            sentiment = keyword_sentiment
                        results['Twitter-RoBERTa'] = {
                            'sentiment': sentiment,
                            'confidence': confidence,
                            'scores': scores
                        }
                
                if use_finbert and models.get('finbert'):
                    scores, raw = analyze_sentiment_with_model(
                        text_input, models['finbert'], 'finbert'
                    )
                    if scores:
                        sentiment, confidence = map_to_extended_sentiment(scores, 'finbert')
                        keyword_sentiment = detect_sentiment_category(text_input, scores)
                        if keyword_sentiment:
                            sentiment = keyword_sentiment
                        results['FinBERT'] = {
                            'sentiment': sentiment,
                            'confidence': confidence,
                            'scores': scores
                        }
                
                if use_bert and models.get('bert'):
                    scores, raw = analyze_sentiment_with_model(
                        text_input, models['bert'], 'bert'
                    )
                    if scores:
                        sentiment, confidence = map_to_extended_sentiment(scores, 'bert')
                        keyword_sentiment = detect_sentiment_category(text_input, scores)
                        if keyword_sentiment:
                            sentiment = keyword_sentiment
                        results['BERT'] = {
                            'sentiment': sentiment,
                            'confidence': confidence,
                            'scores': scores
                        }
            
            # Google GenAI
            if use_google_genai and GOOGLE_GENAI_AVAILABLE:
                genai_result, error = analyze_sentiment_with_google_genai(text_input, google_api_key, google_model_name)
                if error:
                    st.warning(f"Google GenAI: {error}")
                elif genai_result:
                    # Check for keyword overrides
                    keyword_sentiment = detect_sentiment_category(text_input, None)
                    if keyword_sentiment:
                        genai_result['sentiment'] = keyword_sentiment
                    results['Google GenAI'] = genai_result
            
            # ML models
            if SKLEARN_AVAILABLE:
                rf_model, xgb_model, tfidf, sentiment_extractor, w2v = load_ml_models()
                ml_results = predict_with_ml_models(text_input, rf_model, xgb_model,
                                                   tfidf, sentiment_extractor, w2v)
                
                # Filter by selection
                if use_rf and 'Random Forest' in ml_results:
                    results['Random Forest'] = ml_results['Random Forest']
                if use_xgb and 'XGBoost' in ml_results:
                    results['XGBoost'] = ml_results['XGBoost']
        
        # Display results
        if results:
            st.success("Analysis complete!")
            
            # Display input text
            st.subheader("Input Text")
            st.info(text_input)
            
            # Display results for each model
            st.subheader("Sentiment Analysis Results")
            
            cols = st.columns(len(results))
            for idx, (model_name, result) in enumerate(results.items()):
                with cols[idx]:
                    sentiment = result['sentiment']
                    confidence = result['confidence']
                    color = EXTENDED_SENTIMENT_COLORS.get(sentiment, '#95a5a6')
                    
                    st.markdown(
                        f"""
                        <div style="
                            background-color: {color};
                            color: white;
                            padding: 15px;
                            border-radius: 10px;
                            text-align: center;
                            margin: 10px 0;
                        ">
                            <h3>{model_name}</h3>
                            <h2>{sentiment}</h2>
                            <p>Confidence: {confidence:.2%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Detailed scores
            with st.expander("View Detailed Scores"):
                for model_name, result in results.items():
                    st.write(f"**{model_name}**")
                    
                    # Handle different result formats (transformer vs ML)
                    if 'scores' in result:
                        # Transformer model
                        scores_df = pd.DataFrame([
                            {'Label': k, 'Score': v} 
                            for k, v in result['scores'].items()
                        ])
                        st.dataframe(scores_df, use_container_width=True)
                        st.bar_chart(scores_df.set_index('Label'))
                    elif 'probabilities' in result:
                        # ML model
                        prob_df = pd.DataFrame([
                            {'Label': k, 'Probability': v} 
                            for k, v in result['probabilities'].items()
                        ])
                        st.dataframe(prob_df, use_container_width=True)
                        st.bar_chart(prob_df.set_index('Label'))
                        st.write(f"Prediction: {result.get('prediction', 'N/A')}")
            
            # Consensus sentiment
            st.subheader("Consensus Sentiment")
            sentiments = [r['sentiment'] for r in results.values()]
            sentiment_counts = pd.Series(sentiments).value_counts()
            
            if len(sentiment_counts) > 0:
                consensus = sentiment_counts.index[0]
                consensus_color = EXTENDED_SENTIMENT_COLORS.get(consensus, '#95a5a6')
                agreement = sentiment_counts.iloc[0] / len(results) * 100
                
                st.markdown(
                    f"""
                    <div style="
                        background-color: {consensus_color};
                        color: white;
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                        font-size: 24px;
                    ">
                        <strong>{consensus}</strong>
                        <br>
                        <small>Agreement: {agreement:.1f}% ({sentiment_counts.iloc[0]}/{len(results)} models)</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.error("No models were able to analyze the text. Please check model availability.")


def dataset_viewer():
    """Original dataset viewer functionality"""
    # Load data
    train_df, val_df = load_data()
    
    if train_df is None or val_df is None:
        st.error("Failed to load data files. Please ensure twitter_training.csv and twitter_validation.csv are in the current directory.")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Dataset selection
    dataset = st.sidebar.selectbox(
        "Select Dataset",
        ["Training", "Validation", "Both"]
    )
    
    # Get selected dataset
    if dataset == "Training":
        df = train_df.copy()
    elif dataset == "Validation":
        df = val_df.copy()
    else:
        df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Sentiment filter
    sentiments = df['sentiment'].unique().tolist()
    selected_sentiments = st.sidebar.multiselect(
        "Filter by Sentiment",
        options=sentiments,
        default=sentiments
    )
    
    # Target filter
    targets = sorted(df['target'].unique().tolist())
    selected_targets = st.sidebar.multiselect(
        "Filter by Target",
        options=targets,
        default=targets,
        format_func=lambda x: "Positive" if x == 1 else "Negative/Neutral"
    )
    
    # Topic filter
    topics = sorted(df['topic'].dropna().unique().tolist())
    selected_topics = st.sidebar.multiselect(
        "Filter by Topic",
        options=topics,
        default=topics
    )
    
    # Text search
    search_text = st.sidebar.text_input("Search in tweets", "")
    
    # Display options
    show_cleaned = st.sidebar.checkbox("Show cleaned text", value=True)
    samples_per_page = st.sidebar.slider("Samples per page", 10, 100, 50)
    
    # Apply filters
    filtered_df = df[
        (df['sentiment'].isin(selected_sentiments)) &
        (df['target'].isin(selected_targets)) &
        (df['topic'].isin(selected_topics))
    ].copy()
    
    # Apply text search
    if search_text:
        filtered_df = filtered_df[
            filtered_df['cleaned_text'].str.contains(search_text, case=False, na=False)
        ]
    
    # Display statistics
    st.header("Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tweets", len(filtered_df))
    
    with col2:
        sentiment_counts = filtered_df['sentiment'].value_counts()
        st.metric("Unique Sentiments", len(sentiment_counts))
    
    with col3:
        positive_count = len(filtered_df[filtered_df['target'] == 1])
        st.metric("Positive Tweets", positive_count)
    
    with col4:
        negative_count = len(filtered_df[filtered_df['target'] == 0])
        st.metric("Negative/Neutral Tweets", negative_count)
    
    # Sentiment distribution chart
    st.subheader("Sentiment Distribution")
    sentiment_counts = filtered_df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)
    
    # Verify sample sizes
    st.subheader("Sample Size Verification")
    sample_sizes = filtered_df['sentiment'].value_counts()
    for sentiment, count in sample_sizes.items():
        status = "✅" if count >= 50 else "⚠️"
        st.write(f"{status} {sentiment}: {count} samples")
    
    # Display tweets
    st.header("Tweets")
    
    # Pagination
    total_pages = (len(filtered_df) - 1) // samples_per_page + 1
    if total_pages > 1:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        start_idx = (page - 1) * samples_per_page
        end_idx = start_idx + samples_per_page
        page_df = filtered_df.iloc[start_idx:end_idx]
    else:
        page_df = filtered_df
    
    # Display tweets
    for idx, row in page_df.iterrows():
        display_tweet_row(row, show_cleaned)
    
    # Download filtered data
    st.sidebar.header("Export")
    if st.sidebar.button("Download Filtered Data"):
        csv = filtered_df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="filtered_tweets.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
