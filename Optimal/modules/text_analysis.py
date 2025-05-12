# modules/text_analysis.py
import os
import json
from typing import Dict, Any, List, Tuple
import numpy as np
from transformers import pipeline

class TextAnalyzer:
    """Analyze text for sentiment and extract features."""
    
    def __init__(self):
        """Initialize analyzers."""
        self.sentiment_analyzer = pipeline("sentiment-analysis")
    
    def analyze_utterance(self, text: str) -> Dict[str, Any]:
        """Analyze an utterance for sentiment and features."""
        # Run sentiment analysis
        sentiment_result = self.sentiment_analyzer(text)
        sentiment = sentiment_result[0]
        
        # Extract basic features
        word_count = len(text.split())
        avg_word_length = sum(len(word) for word in text.split()) / max(1, word_count)
        contains_question = "?" in text
        
        return {
            "sentiment_label": sentiment["label"],
            "sentiment_score": sentiment["score"],
            "word_count": word_count,
            "avg_word_length": avg_word_length,
            "contains_question": contains_question
        }
    
    def analyze_dialogue(self, dialogue: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze a full dialogue."""
        # Filter for patient utterances only
        patient_utterances = [u for u in dialogue if u["speaker"] == "patient"]
        
        # Analyze each patient utterance
        utterance_analyses = [self.analyze_utterance(u["text"]) for u in patient_utterances]
        
        # Calculate overall sentiment
        sentiment_scores = [a["sentiment_score"] for a in utterance_analyses]
        avg_sentiment = sum(sentiment_scores) / max(1, len(sentiment_scores))
        
        # Calculate responsiveness (based on word count)
        word_counts = [a["word_count"] for a in utterance_analyses]
        avg_word_count = sum(word_counts) / max(1, len(word_counts))
        
        # Determine overall sentiment label
        if avg_sentiment > 0.6:
            overall_sentiment = "Positive"
        elif avg_sentiment < 0.4:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"
        
        return {
            "utterance_analyses": utterance_analyses,
            "avg_sentiment_score": avg_sentiment,
            "overall_sentiment": overall_sentiment,
            "avg_word_count": avg_word_count,
            "num_patient_utterances": len(patient_utterances)
        }