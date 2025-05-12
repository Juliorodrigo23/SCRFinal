# modules/approach_timing_optimizer.py
import numpy as np
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import os

class ApproachTimingOptimizer:
    """
    Determines the optimal timing for robot approach based on multimodal analysis.
    This module implements the core functionality described in the paper
    'Perfecting the Moment: Optimizing Human-Robot Interaction Timing'.
    """
    
    def __init__(self, confidence_threshold: float = 0.65, patience: int = 3):
        """
        Initialize the approach timing optimizer.
        
        Args:
            confidence_threshold: Minimum confidence level required for approach decision
            patience: Number of consecutive positive readings required before approach
        """
        self.confidence_threshold = confidence_threshold
        self.patience = patience
        self.history = []
        self.approach_decisions = []
    
    def analyze_segment(self, 
                        profile_data: Dict[str, Any],
                        audio_analysis: Dict[str, Any],
                        visual_analysis: Dict[str, Any],
                        tom_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a segment to determine if it's an optimal time for robot approach.
        
        Args:
            profile_data: Patient profile information
            audio_analysis: Audio analysis results
            visual_analysis: Visual analysis results  
            tom_analysis: Theory of Mind analysis results
            
        Returns:
            Decision data including approach recommendation and confidence
        """
        # Combine multimodal factors with different weights
        weights = {
            'audio_sentiment': 0.3,
            'visual_engagement': 0.25,
            'tom_receptiveness': 0.3,
            'context_appropriateness': 0.15
        }
        
        # Extract key features from each modality
        sentiment_score = self._normalize_sentiment(audio_analysis.get('speech_sentiment', 'Neutral'))
        
        visual_engagement = visual_analysis.get('engagement_level', 0.5)
        visual_attention = visual_analysis.get('attention_score', 0.5)
        combined_visual = (visual_engagement + visual_attention) / 2
        
        tom_receptiveness = self._convert_tom_to_score(tom_analysis.get('ToM', 'Unsure'))
        
        # Calculate context appropriateness based on profile and cognitive level
        context_score = self._calculate_context_score(profile_data, audio_analysis)
        
        # Calculate weighted approach score
        approach_score = (
            weights['audio_sentiment'] * sentiment_score +
            weights['visual_engagement'] * combined_visual +
            weights['tom_receptiveness'] * tom_receptiveness +
            weights['context_appropriateness'] * context_score
        )
        
        # Calculate confidence based on consistency and strength of signals
        confidence = self._calculate_confidence(
            sentiment_score, combined_visual, tom_receptiveness, context_score
        )
        
        # Determine if this moment is appropriate for approach
        decision = {
            'timestamp': len(self.history),
            'approach_score': approach_score,
            'confidence': confidence,
            'recommend_approach': approach_score >= 0.6 and confidence >= self.confidence_threshold,
            'audio_factors': {
                'sentiment': audio_analysis.get('speech_sentiment', 'Neutral'),
                'sentiment_score': sentiment_score
            },
            'visual_factors': {
                'engagement': visual_engagement,
                'attention': visual_attention
            },
            'tom_factors': {
                'state': tom_analysis.get('ToM', 'Unsure'),
                'receptiveness': tom_receptiveness
            },
            'context_factors': {
                'context_score': context_score,
                'cognitive_level': profile_data.get('cognitive_level', 'Unknown')
            }
        }
        
        # Add to history
        self.history.append(decision)
        
        # Apply patience logic - only recommend approach after consistent positive signals
        if len(self.history) >= self.patience:
            recent_decisions = [d['recommend_approach'] for d in self.history[-self.patience:]]
            sustained_approach = all(recent_decisions)
            decision['sustained_approach_recommended'] = sustained_approach
        else:
            decision['sustained_approach_recommended'] = False
        
        self.approach_decisions.append(decision['sustained_approach_recommended'])
        
        return decision
    
    def _normalize_sentiment(self, sentiment: str) -> float:
        """Convert sentiment string to normalized score."""
        sentiment_map = {
            'Positive': 0.9,
            'Happy': 0.9,
            'Neutral': 0.5,
            'Negative': 0.2,
            'Sad': 0.3,
            'Angry': 0.1,
            'Unknown': 0.4
        }
        return sentiment_map.get(sentiment, 0.5)
    
    def _convert_tom_to_score(self, tom_state: str) -> float:
        """Convert Theory of Mind state to normalized score."""
        tom_map = {
            'Receptive': 0.9,
            'Unsure': 0.5,
            'Not Receptive': 0.1
        }
        return tom_map.get(tom_state, 0.5)
    
    def _calculate_context_score(self, profile_data: Dict[str, Any], 
                                audio_analysis: Dict[str, Any]) -> float:
        """
        Calculate contextual appropriateness score based on profile.
        Considers cognitive level, time of day, and activity.
        """
        # Base score
        score = 0.5
        
        # Adjust for cognitive level - mild is more approachable
        cognitive_level = profile_data.get('cognitive_level', 'Moderate')
        if cognitive_level == 'Mild':
            score += 0.2
        elif cognitive_level == 'Severe':
            score -= 0.2
        
        # Adjust for speaking time - if already speaking a lot, less approachable
        speaking_time = audio_analysis.get('total_speaking_time', 0)
        if speaking_time > 20:  # seconds
            score -= 0.1
        
        # Ensure score is in [0,1] range
        return max(0.0, min(1.0, score))
    
    def _calculate_confidence(self, *factors) -> float:
        """
        Calculate confidence based on consistency of factors.
        Higher variance means lower confidence.
        """
        if not factors:
            return 0.5
            
        # Calculate variance between factors
        variance = np.var(factors)
        
        # High variance means inconsistent signals - lower confidence
        confidence = 1.0 - min(1.0, variance * 5)
        
        # If we have history, increase confidence if signals are stable
        if len(self.history) > 1:
            last_scores = [h['approach_score'] for h in self.history[-3:]] if len(self.history) >= 3 else [h['approach_score'] for h in self.history]
            historical_variance = np.var(last_scores)
            temporal_stability = 1.0 - min(1.0, historical_variance * 5)
            confidence = 0.7 * confidence + 0.3 * temporal_stability
            
        return confidence
    
    def visualize_approach_timeline(self, output_dir: str) -> str:
        """
        Generate visualization of approach decisions over time.
        
        Args:
            output_dir: Directory to save visualization
            
        Returns:
            Path to saved visualization
        """
        if not self.history:
            return ""
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data for plotting
        timestamps = [h['timestamp'] for h in self.history]
        approach_scores = [h['approach_score'] for h in self.history]
        confidence_scores = [h['confidence'] for h in self.history]
        approach_decisions = self.approach_decisions
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot approach scores and confidence
        plt.subplot(2, 1, 1)
        plt.plot(timestamps, approach_scores, 'b-', label='Approach Score')
        plt.plot(timestamps, confidence_scores, 'g--', label='Confidence')
        plt.axhline(y=0.6, color='r', linestyle='-', alpha=0.3, label='Approach Threshold')
        plt.axhline(y=self.confidence_threshold, color='orange', linestyle='-', alpha=0.3, label='Confidence Threshold')
        plt.xlabel('Time Segment')
        plt.ylabel('Score')
        plt.title('Approach Score and Confidence Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot approach decisions
        plt.subplot(2, 1, 2)
        decision_y = [1 if d else 0 for d in approach_decisions]
        plt.step(timestamps, decision_y, 'r-', where='post', label='Approach Decision')
        plt.fill_between(timestamps, 0, decision_y, step='post', alpha=0.3, color='red')
        plt.xlabel('Time Segment')
        plt.yticks([0, 1], ['No', 'Yes'])
        plt.ylabel('Approach?')
        plt.title('Robot Approach Decisions Over Time')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, 'approach_timeline.png')
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def get_optimal_approach_windows(self) -> List[Dict[str, Any]]:
        """
        Identify optimal windows of time for robot approach.
        
        Returns:
            List of time windows optimal for approach
        """
        if not self.approach_decisions:
            return []
            
        approach_windows = []
        in_window = False
        start_idx = 0
        
        # Find contiguous blocks of approach decisions
        for i, decision in enumerate(self.approach_decisions):
            if decision and not in_window:
                # Start of a new window
                in_window = True
                start_idx = i
            elif not decision and in_window:
                # End of a window
                in_window = False
                approach_windows.append({
                    'start_segment': start_idx,
                    'end_segment': i - 1,
                    'duration': i - start_idx,
                    'avg_score': np.mean([self.history[j]['approach_score'] for j in range(start_idx, i)]),
                    'avg_confidence': np.mean([self.history[j]['confidence'] for j in range(start_idx, i)])
                })
        
        # Handle case where we end in an approach window
        if in_window:
            i = len(self.approach_decisions)
            approach_windows.append({
                'start_segment': start_idx,
                'end_segment': i - 1,
                'duration': i - start_idx,
                'avg_score': np.mean([self.history[j]['approach_score'] for j in range(start_idx, i)]),
                'avg_confidence': np.mean([self.history[j]['confidence'] for j in range(start_idx, i)])
            })
        
        # Sort windows by average score * confidence to prioritize
        approach_windows.sort(key=lambda x: x['avg_score'] * x['avg_confidence'], reverse=True)
        
        return approach_windows