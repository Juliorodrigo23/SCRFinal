# modules/image_generator.py
import os
import openai
from PIL import Image
import requests
from io import BytesIO
from typing import Dict

from config.config import Config
from modules.profile_module import PatientProfile

class ImageGenerator:
    """Generate images using OpenAI's DALL-E."""
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        openai.api_key = config.get("api_keys.openai")
    
    def generate_image(self, profile: PatientProfile, routine_event: str, emotion: str) -> str:
        """Generate an image for a profile and interaction."""
        # Create detailed prompt for DALL-E
        prompt = self._create_image_prompt(profile, routine_event, emotion)
        
        try:
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            
            image_url = response['data'][0]['url']
            
            # Download and save the image
            return self._download_and_save_image(image_url, profile.name, routine_event)
            
        except Exception as e:
            print(f"Error generating image: {e}")
            return ""
    
    def _create_image_prompt(self, profile: PatientProfile, routine_event: str, emotion: str) -> str:
        """Create a detailed prompt for DALL-E."""
        # Base description of the person
        base_desc = f"Elderly {profile.ethnicity} {profile.gender.lower()} with {profile.hair.lower()} hair"
        
        # Add emotional state
        emotion_map = {
            "Positive": "smiling, looking content",
            "Neutral": "with a neutral expression",
            "Negative": "looking confused or upset"
        }
        emotion_desc = emotion_map.get(emotion, "with a neutral expression")
        
        # Add setting based on routine event
        setting = ""
        if "dress" in routine_event.lower():
            setting = "in a bedroom near a closet"
        elif "breakfast" in routine_event.lower():
            setting = "at a dining table with breakfast items"
        elif "medication" in routine_event.lower():
            setting = "with medication containers nearby"
        elif "prayer" in routine_event.lower():
            setting = "with prayer items nearby"
        elif "tai chi" in routine_event.lower():
            setting = "in comfortable clothes in a calm space"
        elif "photo" in routine_event.lower():
            setting = "looking at a photo album"
        elif "read" in routine_event.lower():
            setting = "with a newspaper or book"
        elif "nap" in routine_event.lower():
            setting = "sitting in a comfortable chair"
        elif "activity" in routine_event.lower():
            setting = "in a common room with other seniors"
        else:
            setting = "in a care facility living space"
        
        # Add specific notes from profile
        specific_notes = ""
        if "military" in profile.notes.lower():
            specific_notes = ", military posture, possibly with a military photo or medal visible"
        elif "religious" in profile.notes.lower():
            specific_notes = ", possibly with religious symbols or items nearby"
        elif "teacher" in profile.notes.lower():
            specific_notes = ", scholarly appearance, possibly with reading glasses"
        
        # Combine all elements into a detailed prompt
        prompt = f"{base_desc}, {emotion_desc}, {setting}{specific_notes}. Realistic style, soft lighting, respectful depiction of an elderly person with dementia."
        
        return prompt
    
    def _download_and_save_image(self, image_url: str, profile_name: str, routine_event: str) -> str:
        """Download and save an image from URL."""
        # Create a clean event name for file paths
        clean_event = routine_event.replace(" - ", "_").replace(" ", "_").lower()
        
        # Create output directory
        output_dir = f"output/{profile_name}/{routine_event.split(' - ')[0]}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output file path
        output_path = f"{output_dir}/{clean_event}_image.png"
        
        # Download and save the image
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img.save(output_path)
        
        return output_path