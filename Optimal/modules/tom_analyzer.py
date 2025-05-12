# modules/tom_analyzer.py
from typing import Dict, Any, List, Tuple
from modules.profile_module import PatientProfile
from modules.text_analysis import TextAnalyzer

class ToMAnalyzer:
    """Theory of Mind analyzer for patient interactions."""
    
    def __init__(self, text_analyzer: TextAnalyzer):
        """Initialize with text analyzer."""
        self.text_analyzer = text_analyzer
    
    def analyze_interaction(self, 
                         profile: PatientProfile, 
                         dialogue: List[Dict[str, str]], 
                         dialogue_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze interaction for Theory of Mind state."""
        # Extract key features from profile and dialogue analysis
        personality = profile.personality.lower()
        cognitive_level = profile.cognitive_level
        sentiment = dialogue_analysis["overall_sentiment"]
        avg_word_count = dialogue_analysis["avg_word_count"]
        
        # Initialize variables for ToM estimation
        receptiveness = 0.5  # Default middle value
        emotion = "Neutral"  # Default emotion
        
        # Adjust for personality traits
        if "grumpy" in personality:
            receptiveness -= 0.1
        if "warm" in personality:
            receptiveness += 0.1
        if "quiet" in personality:
            receptiveness -= 0.05
        if "jovial" in personality or "social" in personality:
            receptiveness += 0.1
        
        # Adjust for cognitive level
        if cognitive_level == "Severe":
            receptiveness -= 0.2
        elif cognitive_level == "Mild":
            receptiveness += 0.1
        
        # Adjust for sentiment
        if sentiment == "Positive":
            receptiveness += 0.2
            emotion = "Positive"
        elif sentiment == "Negative":
            receptiveness -= 0.2
            emotion = "Negative"
        
        # Adjust for verbosity (engagement)
        if avg_word_count > 15:
            receptiveness += 0.1
        elif avg_word_count < 5:
            receptiveness -= 0.1
        
        # Determine ToM state
        tom_state = "Receptive"
        if receptiveness < 0.4:
            tom_state = "Not Receptive"
        elif receptiveness < 0.6:
            tom_state = "Unsure"
        
        # Ensure approachability stays in valid range
        approachability = max(0.0, min(1.0, receptiveness))
        
        # Determine if robot should approach
        robot_should_approach = approachability >= 0.5
        
        return {
            "approachability": round(approachability, 2),
            "emotion": emotion,
            "ToM": tom_state,
            "speech_sentiment": sentiment,
            "robot_should_approach": robot_should_approach
        }