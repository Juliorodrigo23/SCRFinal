# modules/dataset_packager.py
import os
import json
import shutil
from typing import Dict, Any, List

class DatasetPackager:
    """Package all components of the interaction into a dataset."""
    
    def package_interaction(self,
                          profile_name: str,
                          routine_event: str,
                          dialogue: List[Dict[str, str]],
                          audio_files: Dict[str, str],
                          tom_analysis: Dict[str, Any],
                          image_path: str) -> Dict[str, Any]:
        """Package an interaction into a dataset."""
        # Create a clean event name for file paths
        clean_event = routine_event.replace(" - ", "_").replace(" ", "_").lower()
        time_period = routine_event.split(" - ")[0]
        
        # Create output directory
        output_dir = f"output/{profile_name}/{time_period}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create metadata
        metadata = {
            "profile_id": profile_name,
            "interaction": routine_event,
            "approachability": tom_analysis["approachability"],
            "emotion": tom_analysis["emotion"],
            "ToM": tom_analysis["ToM"],
            "speech_sentiment": tom_analysis["speech_sentiment"],
            "robot_should_approach": tom_analysis["robot_should_approach"]
        }
        
        # Save metadata
        metadata_path = f"{output_dir}/{clean_event}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save transcript
        transcript_path = f"{output_dir}/{clean_event}_transcript.txt"
        with open(transcript_path, 'w') as f:
            for utterance in dialogue:
                speaker = "Robot" if utterance["speaker"] == "robot" else profile_name
                f.write(f"{speaker}: {utterance['text']}\n")
        
        # Copy image file if not already in the right location
        if image_path and image_path != f"{output_dir}/{clean_event}_image.png" and os.path.exists(image_path):
            shutil.copy(image_path, f"{output_dir}/{clean_event}_image.png")
        
        # Return paths to all components
        return {
            "metadata": metadata_path,
            "transcript": transcript_path,
            "audio_files": audio_files,
            "image": f"{output_dir}/{clean_event}_image.png" if image_path else None
        }