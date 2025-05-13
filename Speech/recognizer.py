#!/usr/bin/env python3
"""
Voice-Activated Reminder System
Speech Recognition Module: Captures audio and returns recognized string
"""

import speech_recognition as sr
import time

class Recognizer:
    """Class to handle speech recognition functionality."""
    
    def __init__(self):
        """Initialize the speech recognizer."""
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise
        with self.microphone as source:
            print("Calibrating for ambient noise. Please remain silent...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Calibration complete.")
    
    def listen_for_command(self, timeout=5, phrase_time_limit=5):
        """
        Listen for a voice command and convert it to text.
        
        Args:
            timeout (int): How long to wait before timing out
            phrase_time_limit (int): Max length of the phrase to listen for
            
        Returns:
            str: The recognized text, or None if recognition failed
        """
        with self.microphone as source:
            print("Listening...")
            try:
                # Listen for audio input
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                print("Processing speech...")
                
                # Convert speech to text using Google's speech recognition
                text = self.recognizer.recognize_google(audio)
                return text
                
            except sr.WaitTimeoutError:
                print("Listening timed out. No speech detected.")
                return None
                
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
                
            except sr.RequestError as e:
                print(f"Speech recognition service error: {e}")
                return None
            
            except Exception as e:
                print(f"Unexpected error in speech recognition: {e}")
                return None
    
    def continuous_listen(self, callback, stop_event):
        """
        Continuously listen for commands until stopped.
        
        Args:
            callback (function): Function to call with recognized text
            stop_event (threading.Event): Event to signal when to stop listening
        """
        while not stop_event.is_set():
            text = self.listen_for_command()
            if text:
                callback(text)
            time.sleep(0.1)  # Small delay to prevent CPU hogging