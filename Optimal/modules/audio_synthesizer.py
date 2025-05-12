# modules/audio_synthesizer.py
import os
import requests
import json
from typing import Dict, List, Tuple
from pydub import AudioSegment
from config.config import Config

class AudioSynthesizer:
    """Generate audio using ElevenLabs API."""
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.api_key = config.get("api_keys.elevenlabs")
        self.voice_mapping = config.get("voice_mapping")
        self.base_url = "https://api.elevenlabs.io/v1"
    
    def _get_voice_id(self, voice_name: str) -> str:
        """Get voice ID from voice name using ElevenLabs API."""
        headers = {
            "Accept": "application/json",
            "xi-api-key": self.api_key
        }
        
        response = requests.get(f"{self.base_url}/voices", headers=headers)
        voices = response.json().get("voices", [])
        
        # Try to find a voice with a name that includes the requested voice name
        for voice in voices:
            if voice_name.lower() in voice.get("name", "").lower():
                return voice.get("voice_id")
        
        # If not found, return the first voice as a fallback
        if voices:
            return voices[0].get("voice_id")
        
        raise ValueError("No voices found in ElevenLabs account.")
    
    def synthesize_speech(self, text: str, speaker_type: str, speaker_name: str) -> bytes:
        """Synthesize speech using ElevenLabs API."""
        # Determine the voice to use
        voice_name = self.voice_mapping.get(speaker_name if speaker_type == "patient" else "robot")
        voice_id = self._get_voice_id(voice_name)
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.7,
                "similarity_boost": 0.5
            }
        }
        
        response = requests.post(
            f"{self.base_url}/text-to-speech/{voice_id}",
            json=data,
            headers=headers
        )
        
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Error synthesizing speech: {response.text}")
    
    def process_dialogue(self, dialogue: List[Dict[str, str]], profile_name: str, routine_event: str) -> Dict[str, str]:
        """Process a dialogue and generate audio files for each utterance."""
        # Create a clean event name for file paths
        clean_event = routine_event.replace(" - ", "_").replace(" ", "_").lower()
        
        # Create output directory
        output_dir = f"output/{profile_name}/{routine_event.split(' - ')[0]}"
        os.makedirs(output_dir, exist_ok=True)
        
        audio_files = {}
        full_audio_path = f"{output_dir}/{clean_event}_full.wav"
        audio_segments = []
        # Process each utterance and save audio
        for i, utterance in enumerate(dialogue):
            speaker_type = utterance["speaker"]
            text = utterance["text"]
            
            # Generate audio content
            audio_content = self.synthesize_speech(
                text=text,
                speaker_type=speaker_type,
                speaker_name=profile_name
            )
            
            # Save individual utterance
            file_path = f"{output_dir}/{clean_event}_{speaker_type}_{i+1}.wav"
            with open(file_path, "wb") as f:
                f.write(audio_content)
            
            audio_files[f"{speaker_type}_{i+1}"] = file_path
        
        # Add to list for combining
            try:
                # Load as AudioSegment for combining
                segment = AudioSegment.from_file(file_path)
                
                # Add a short pause between utterances (500ms)
                if i > 0:
                    pause = AudioSegment.silent(duration=500)  # 500ms pause
                    audio_segments.append(pause)
                
                audio_segments.append(segment)
            except Exception as e:
                print(f"Warning: Could not load audio file for combining: {e}")
        
        # Combine all audio segments into a single file
        if audio_segments:
            try:
                print(f"Combining {len(audio_segments)} audio segments into a full conversation audio...")
                combined_audio = audio_segments[0]
                for segment in audio_segments[1:]:
                    combined_audio += segment
                
                # Export the combined audio
                combined_audio.export(full_audio_path, format="wav")
                audio_files["full_conversation"] = full_audio_path
                print(f"Saved full conversation audio to {full_audio_path}")
            except Exception as e:
                print(f"Warning: Failed to combine audio segments: {e}")
        
        return audio_files