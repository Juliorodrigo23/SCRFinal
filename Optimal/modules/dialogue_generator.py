# modules/dialogue_generator.py
import openai
import json
from typing import Dict, List, Tuple

from config.config import Config
from modules.profile_module import PatientProfile

class DialogueGenerator:
    """Generate dialogue using OpenAI's GPT API."""
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        openai.api_key = config.get("api_keys.openai")
    
    def generate_dialogue(self, profile: PatientProfile, routine_event: str) -> List[Dict[str, str]]:
        """Generate a dialogue between robot and patient."""
        context = profile.get_prompt_context()
        
        prompt = f"""
Generate a realistic dialogue between a robot assistant (CompanionCare) and a dementia patient.

PATIENT CONTEXT:
{context}

CURRENT SITUATION:
The patient is currently engaged in: {routine_event}
Time of day: {"Morning" if "Morning" in routine_event else "Afternoon"}

INSTRUCTIONS:
- Create a short, natural conversation (2-5 utterances per speaker)
- The robot should start with: "Hello, {profile.name}, how can I assist?"
- Include emotional reactions that reflect the patient's personality and cognitive level
- Format the output as a JSON array with each utterance having 'speaker' (either 'robot' or 'patient') and 'text' fields

Example format:
[
  {{"speaker": "robot", "text": "Hello, {profile.name}, how can I assist?"}},
  {{"speaker": "patient", "text": "Who are you? I don't remember asking for help."}},
  {{"speaker": "robot", "text": "I'm your CompanionCare assistant. Would you like help with {routine_event.lower()}?"}},
  {{"speaker": "patient", "text": "Oh, yes please. I'm feeling a bit confused today."}}
]
"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an expert in creating realistic dialogues for dementia patient simulations."},
                       {"role": "user", "content": prompt}]
        )
        
        try:
            dialogue = json.loads(response.choices[0].message.content)
            return dialogue
        except json.JSONDecodeError:
            # Fallback in case of improper JSON
            content = response.choices[0].message.content
            # Extract content between code blocks if present
            if "```json" in content and "```" in content:
                json_content = content.split("```json")[1].split("```")[0].strip()
                return json.loads(json_content)
            
            # Simple fallback
            return [
                {"speaker": "robot", "text": f"Hello, {profile.name}, how can I assist?"},
                {"speaker": "patient", "text": "I'm not sure what I'm supposed to be doing."},
                {"speaker": "robot", "text": f"We're working on {routine_event.lower()}. Would you like some help?"},
                {"speaker": "patient", "text": "Yes, please. That would be good."}
            ]
    
    def save_dialogue(self, dialogue: List[Dict[str, str]], profile_name: str, routine_event: str) -> str:
        """Save dialogue to file and return the file path."""
        # Create a clean filename from the routine event
        clean_event = routine_event.replace(" - ", "_").replace(" ", "_").lower()
        
        # Create the output directory path
        output_dir = f"data/{profile_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output file path
        output_file = f"{output_dir}/{clean_event}_dialogue.json"
        
        # Save dialogue to file
        with open(output_file, 'w') as f:
            json.dump(dialogue, f, indent=2)
        
        return output_file