"""
Streamlit Interactive Tweet Viewer
Filter and visualize tweets by sentiment and target labels
"""

import streamlit as st
import pandas as pd
import numpy as np
import re

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
            if neg_score > 0.8:
                return "Angry", neg_score
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
            if neg_score > 0.75:
                return "Angry", neg_score
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
            return "Sarcastic", star_2
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
    
    # Check for keywords with priority
    if any(kw in text_lower for kw in angry_keywords):
        return "Angry"
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


def analyze_text_sentiment(text):
    """Analyze text sentiment using all available models"""
    if not text or len(text.strip()) == 0:
        return None
    
    models = load_sentiment_models()
    results = {}
    
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
    
    return results


@st.cache_data
def load_data():
    """Load and process data"""
    try:
        train_df = pd.read_csv('twitter_training.csv', header=None,
                              names=['id', 'topic', 'sentiment', 'tweet'])
        val_df = pd.read_csv('twitter_validation.csv', header=None,
                            names=['id', 'topic', 'sentiment', 'tweet'])
        
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
    st.markdown("Analyze sentiment using Hugging Face Transformers, FinBERT, and BERT")
    
    # Check if transformers are available
    if not TRANSFORMERS_AVAILABLE:
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
    
    # Text input
    text_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Type or paste your text here...",
        help="Enter any text to get sentiment analysis from multiple models"
    )
    
    # Model selection
    st.sidebar.header("Model Selection")
    use_twitter_roberta = st.sidebar.checkbox("Twitter-RoBERTa", value=True)
    use_finbert = st.sidebar.checkbox("FinBERT", value=True)
    use_bert = st.sidebar.checkbox("BERT Multilingual", value=True)
    
    if st.button("Analyze Sentiment", type="primary"):
        if not text_input or len(text_input.strip()) == 0:
            st.warning("Please enter some text to analyze.")
            return
        
        with st.spinner("Analyzing sentiment..."):
            # Load models
            models = load_sentiment_models()
            results = {}
            
            # Analyze with selected models
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
                    scores_df = pd.DataFrame([
                        {'Label': k, 'Score': v} 
                        for k, v in result['scores'].items()
                    ])
                    st.dataframe(scores_df, use_container_width=True)
                    st.bar_chart(scores_df.set_index('Label'))
            
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

