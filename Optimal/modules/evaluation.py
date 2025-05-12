# modules/evaluation.py
import os
import json
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ApproachabilityEvaluator:
    """Evaluate the performance of the approachability prediction."""
    
    def evaluate_dataset(self, dataset_dir: str = "output") -> Dict[str, Any]:
        """Evaluate all interactions in the dataset."""
        # Get all metadata files
        metadata_files = []
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith("_metadata.json"):
                    metadata_files.append(os.path.join(root, file))
        
        # Extract true and predicted labels
        y_true = []
        y_pred = []
        y_scores = []
        
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
                # Get ground truth and prediction
                ground_truth = metadata.get("robot_should_approach", False)
                score = metadata.get("approachability", 0.5)
                prediction = score >= 0.5
                
                y_true.append(int(ground_truth))
                y_pred.append(int(prediction))
                y_scores.append(score)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "num_samples": len(y_true),
            "positive_rate": sum(y_true) / len(y_true)
        }
    
    def calculate_correlation(self, dataset_dir: str = "output") -> Dict[str, float]:
        """Calculate correlation between features and approachability."""
        # Get all metadata files
        metadata_files = []
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith("_metadata.json"):
                    metadata_files.append(os.path.join(root, file))
        
        # Extract features and target
        features = {
            "emotion_positive": [],
            "emotion_negative": [],
            "tom_receptive": [],
            "tom_not_receptive": [],
            "sentiment_positive": [],
            "sentiment_negative": []
        }
        approachability = []
        
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
                # Get approachability score
                score = metadata.get("approachability", 0.5)
                approachability.append(score)
                
                # Extract features
                features["emotion_positive"].append(1 if metadata.get("emotion") == "Positive" else 0)
                features["emotion_negative"].append(1 if metadata.get("emotion") == "Negative" else 0)
                features["tom_receptive"].append(1 if metadata.get("ToM") == "Receptive" else 0)
                features["tom_not_receptive"].append(1 if metadata.get("ToM") == "Not Receptive" else 0)
                features["sentiment_positive"].append(1 if metadata.get("speech_sentiment") == "Positive" else 0)
                features["sentiment_negative"].append(1 if metadata.get("speech_sentiment") == "Negative" else 0)
        
        # Calculate correlations
        correlations = {}
        for feature_name, feature_values in features.items():
            if len(feature_values) > 0 and len(approachability) > 0:
                # Convert to numpy arrays for correlation calculation
                x = np.array(feature_values)
                y = np.array(approachability)
                
                # Calculate correlation
                correlation = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 else 0
                correlations[feature_name] = correlation
        
        return correlations