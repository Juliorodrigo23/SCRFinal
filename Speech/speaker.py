#!/usr/bin/env python3
"""
Voice-Activated Reminder System
Speaker Module: Handles text-to-speech functionality using pyttsx3
"""

import pyttsx3

class Speaker:
    """Class to handle text-to-speech functionality."""
    
    def __init__(self, rate=150, volume=1.0, voice=None):
        """
        Initialize the text-to-speech engine.
        
        Args:
            rate (int): Speech rate (words per minute)
            volume (float): Volume level (0.0 to 1.0)
            voice (str): Voice ID to use (None for default)
        """
        self.engine = pyttsx3.init()
        self.set_properties(rate, volume, voice)
    
    def set_properties(self, rate=None, volume=None, voice=None):
        """
        Set speech properties.
        
        Args:
            rate (int): Speech rate (words per minute)
            volume (float): Volume level (0.0 to 1.0)
            voice (str): Voice ID to use
        """
        if rate is not None:
            self.engine.setProperty('rate', rate)
        
        if volume is not None:
            self.engine.setProperty('volume', volume)
        
        if voice is not None:
            voices = self.engine.getProperty('voices')
            for v in voices:
                if voice in v.id:
                    self.engine.setProperty('voice', v.id)
                    break
    
    def get_available_voices(self):
        """
        Get list of available voices.
        
        Returns:
            list: List of available voice IDs
        """
        voices = self.engine.getProperty('voices')
        return [v.id for v in voices]
    
    def speak(self, text):
        """
        Convert text to speech.
        
        Args:
            text (str): Text to be spoken
        """
        try:
            print(f"Speaking: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
    
    def speak_async(self, text):
        """
        Speak text asynchronously without blocking.
        
        Args:
            text (str): Text to be spoken
        """
        try:
            import threading
            thread = threading.Thread(target=self.speak, args=(text,))
            thread.daemon = True
            thread.start()
        except Exception as e:
            print(f"Error in async speech: {e}")