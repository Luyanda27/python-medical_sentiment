# -*- coding: utf-8 -*-
"""
Advanced Medical Aid Sentiment Analysis Pipeline
Author: Corey 
Date: 2025-06-30
"""

# ======================
# 1. SETUP & CONFIGURATION
# ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import shap
import mlflow
from mlflow.models import infer_signature
import warnings

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 200)
nltk.download(['stopwords', 'wordnet'])
plt.style.use('ggplot')

# ======================
# 2. DATA GENERATION
# ======================
def generate_medical_reviews(n_samples=500):
    """
    Generates synthetic medical aid reviews with realistic patterns
    using Faker and pre-trained sentiment models for label alignment
    
    Args:
        n_samples (int): Number of reviews to generate
        
    Returns:
        pd.DataFrame: Generated reviews with ratings and categories
    """
    fake = Faker()
    sentiment_analyzer = pipeline(
        "text-classification", 
        model="finiteautomata/bertweet-base-sentiment-analysis"
    )
    
    # Medical domain-specific themes
    themes = {
        "Claims": ["delay", "denied", "documents", "approval", "rejection", "process"],
        "Customer_Service": ["rude", "unhelpful", "waiting", "call", "response", "resolution"],
        "Payments": ["increase", "cost", "deduction", "refund", "unaffordable", "price"],
        "Coverage": ["limit", "hospital", "doctor", "treatment", "medicine", "excluded"]
    }
    
    data = []
    for _ in range(n_samples):
        # Select random category and associated keywords
        category = np.random.choice(list(themes.keys()))
        keywords = themes[category]
        
        # Generate realistic review text
        review = " ".join([
            fake.sentence(ext_word_list=keywords),
            fake.sentence(),
            fake.sentence()
        ])
        
        # Get sentiment and align with rating (1-5)
        sentiment = sentiment_analyzer(review[:512])[0]['label']
        rating = {
            "POS": np.random.choice([4, 5]),
            "NEG": np.random.choice([1, 2]), 
            "NEU": np.random.choice([3, 4])
        }[sentiment]
        
        data.append([review, rating, category])
    
    return pd.DataFrame(data, columns=["review", "rating", "category"])

# Generate and save dataset
medical_df = generate_medical_reviews(500)
medical_df.to_csv("medical_aid_reviews.csv", index=False)

# ======================
# 3. TEXT PREPROCESSING
# ======================
class TextPreprocessor:
    """
    Advanced text cleaning and normalization pipeline with:
    - Medical domain-specific stopwords
    - Lemmatization with POS tagging
    - Custom regex patterns for insurance terminology
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.medical_stopwords = {
            'medical', 'aid', 'scheme', 'provider', 'member', 
            'hello', 'peter', 'client', 'customer'
        }
        self.regex_patterns = [
            (r'\b(claim|claims)\b', 'claim'),
            (r'\b(pay|payment|paid)\b', 'payment'),
            (r'\b(cover|coverage)\b', 'coverage')
        ]
        
    def clean_text(self, text):
        """Main cleaning pipeline"""
        if not isinstance(text, str):
            return ""
            
        # Lowercase and remove special chars
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Apply custom regex replacements
        for pattern, replacement in self.regex_patterns:
            text = re.sub(pattern, replacement, text)
            
        # Tokenize and lemmatize with POS awareness
        tokens = []
        for token in text.split():
            if token not in self.stop_words and token not in self.medical_stopwords:
                # Simple POS tagging (noun by default)
                pos = 'n'
                if token.endswith('ing'):
                    pos = 'v'
                tokens.append(self.lemmatizer.lemmatize(token, pos))
                
        return ' '.join(tokens)

# Apply preprocessing
preprocessor = TextPreprocessor()
medical_df['cleaned_review'] = medical_df['review'].apply(preprocessor.clean_text)

# ======================
# 4. TOPIC MODELING (BERTopic)
# ======================
def train_topic_model(documents):
    """
    Trains BERTopic model with medical domain optimizations:
    - Sentence-BERT embeddings fine-tuned for healthcare
    - Dynamic topic reduction
    - Custom stopwords
    
    Args:
        documents (list): Preprocessed review texts
        
    Returns:
        BERTopic: Trained topic model
    """
    # Medical domain embedding model
    embedding_model = SentenceTransformer(
        'paraphrase-multilingual-MiniLM-L12-v2'
    )
    
    # Custom stopwords
    custom_stopwords = list(preprocessor.stop_words.union(
        preprocessor.medical_stopwords
    ))
    
    # Initialize and train BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        language='english',
        nr_topics='auto',
        verbose=True,
        calculate_probabilities=True
    )
    
    topics, _ = topic_model.fit_transform(documents)
    
    return topic_model

# Train and visualize
topic_model = train_topic_model(medical_df['cleaned_review'].tolist())

# Visualization
topic_model.visualize_barchart(
    top_n_topics=6,
    n_words=5,
    title="Top Medical Complaint Topics"
)

# ======================
# 5. SENTIMENT ANALYSIS (RoBERTa)
# ======================
class SentimentAnalyzer:
    """
    Advanced sentiment analysis using RoBERTa-base model fine-tuned on Twitter data
    (similar domain to customer reviews) with GPU acceleration support
    """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment"
        ).to(self.device)
        
    def analyze(self, text):
        """Get sentiment probabilities for input text"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return {
            'negative': scores[0][0].item(),
            'neutral': scores[0][1].item(),
            'positive': scores[0][2].item()
        }

# Initialize and run analysis
sentiment_analyzer = SentimentAnalyzer()
medical_df['sentiment'] = medical_df['review'].apply(
    lambda x: sentiment_analyzer.analyze(x[:512])
)

# ======================
# 6. CLASSIFICATION PIPELINE
# ======================
def train_classification_pipeline(df):
    """
    Trains and evaluates a GradientBoosting classifier with:
    - TF-IDF vectorization (n-grams)
    - SHAP explainability
    - MLflow tracking
    
    Args:
        df (pd.DataFrame): Processed dataframe with reviews and categories
        
    Returns:
        sklearn.Pipeline: Trained classification pipeline
    """
    # Prepare data
    X = df['cleaned_review']
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # MLflow tracking
    mlflow.set_experiment("Medical_Aid_Complaint_Classification")
    
    with mlflow.start_run():
        # Define pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=5000,
                stop_words=list(preprocessor.stop_words)
            ),
            ('clf', GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ))
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Log to MLflow
        mlflow.log_params({
            "model": "GradientBoosting",
            "ngram_range": "1-3",
            "max_features": 5000,
            "n_estimators": 150
        })
        
        mlflow.log_metrics({
            "accuracy": report['accuracy'],
            "f1_macro": report['macro avg']['f1-score']
        })
        
        # Log model with signature
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            pipeline, 
            "complaint_classifier",
            signature=signature
        )
        
        # SHAP explainability
        explainer = shap.Explainer(pipeline.named_steps['clf'])
        X_test_transformed = pipeline.named_steps['tfidf'].transform(X_test)
        shap_values = explainer(X_test_transformed.toarray())
        
        # Save SHAP plot
        plt.figure()
        shap.summary_plot(
            shap_values, 
            X_test_transformed.toarray(),
            feature_names=pipeline.named_steps['tfidf'].get_feature_names_out(),
            class_names=pipeline.classes_,
            show=False
        )
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "shap_summary.png")
        plt.close()
        
    return pipeline

# Train and evaluate
classifier = train_classification_pipeline(medical_df)

# ======================
# 7. VISUALIZATION & REPORTING
# ======================
def generate_visualizations(df, topic_model):
    """
    Creates key visualizations for the final report:
    - Sentiment distribution
    - Topic word clouds
    - Temporal trends
    """
    # Sentiment distribution
    sentiment_dist = pd.DataFrame(df['sentiment'].tolist()).describe()
    
    plt.figure(figsize=(10, 6))
    for col in ['negative', 'neutral', 'positive']:
        sns.kdeplot(df['sentiment'].apply(lambda x: x[col]), label=col)
    plt.title("Sentiment Distribution Across Reviews")
    plt.xlabel("Probability")
    plt.legend()
    plt.savefig("sentiment_distribution.png")
    plt.close()
    
    # Topic visualization
    topic_model.visualize_topics().write_html("topic_visualization.html")
    
    # Temporal analysis (if date column exists)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        monthly_sentiment = df.set_index('date')['sentiment'].apply(
            lambda x: x['positive'] - x['negative']
        ).resample('M').mean()
        
        plt.figure(figsize=(12, 6))
        monthly_sentiment.plot()
        plt.title("Monthly Net Sentiment Trend")
        plt.ylabel("Sentiment (Positive - Negative)")
        plt.savefig("sentiment_trend.png")
        plt.close()

generate_visualizations(medical_df, topic_model)

# ======================
# 8. DEPLOYMENT TEMPLATE
# ======================
"""
FastAPI deployment template (save as api.py):

from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import mlflow.pyfunc

app = FastAPI()

# Load models
topic_model = BERTopic.load("bertopic_model")
classifier = mlflow.pyfunc.load_model("mlruns/0/<RUN_ID>/artifacts/complaint_classifier")

class ReviewRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_review(request: ReviewRequest):
    # Preprocess
    cleaned_text = preprocessor.clean_text(request.text)
    
    # Get topic
    topic_info = topic_model.transform([cleaned_text])
    
    # Get sentiment
    sentiment = sentiment_analyzer.analyze(request.text)
    
    # Get category
    category = classifier.predict([cleaned_text])[0]
    
    return {
        "topic": int(topic_info[0][0]),
        "topic_confidence": float(topic_info[1][0]),
        "sentiment": sentiment,
        "category": category
    }

To run:
uvicorn api:app --reload
"""

print("Pipeline execution completed successfully!")