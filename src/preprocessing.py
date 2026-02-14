import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
import torch

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self._download_nltk_resources()
        
    def _download_nltk_resources(self):
        """Download necessary NLTK data sparingly."""
        resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'punkt_tab']
        for res in resources:
            try:
                nltk.data.find(f'tokenizers/{res}')
            except LookupError:
                nltk.download(res, quiet=True)
            except ValueError:
                # Handle cases where resource path is different (e.g. corpora vs tokenizers)
                try:
                    nltk.data.find(f'corpora/{res}')
                except LookupError:
                    nltk.download(res, quiet=True)

    def clean_text(self, text):
        """Clean and preprocess text."""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess(self, text):
        """Complete preprocessing pipeline."""
        text = self.clean_text(text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

class SentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipeline = pipeline("sentiment-analysis", model=model_name, device=self.device)

    def get_score(self, text):
        """Get sentiment score from text (-1 to 1)."""
        try:
            text = text[:512]  # Truncate to BERT max length
            result = self.pipeline(text)[0]
            score = result['score']
            return score if result['label'] == 'POSITIVE' else -score
        except Exception:
            return 0.0

    def analyze_dataframe(self, df):
        """Apply sentiment analysis to dataframe in place."""
        print("Running sentiment analysis on dataset...")
        df['sentiment_score'] = df['content'].apply(self.get_score)
        df['title_sentiment'] = df['title'].apply(self.get_score)
        df['combined_sentiment'] = 0.7 * df['sentiment_score'] + 0.3 * df['title_sentiment']
        return df
