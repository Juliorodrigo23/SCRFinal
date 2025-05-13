#!/usr/bin/env python3
"""
Voice-Activated Reminder System
Main Entry Point: This module orchestrates the entire flow:
waits for wake word → calls recognizer → parses → stores → runs scheduler
"""

import time
import threading
from recognizer import Recognizer
from speaker import Speaker
from parser import CommandParser
from reminders import ReminderStore
from scheduler import ReminderScheduler

class VoiceReminderSystem:
    """Main class that orchestrates the voice reminder system flow."""
    
    def __init__(self, wake_word="computer"):
        """Initialize components of the voice reminder system."""
        self.wake_word = wake_word.lower()
        self.recognizer = Recognizer()
        self.speaker = Speaker()
        self.parser = CommandParser()
        self.reminder_store = ReminderStore()
        self.scheduler = ReminderScheduler(self.reminder_store, self.speaker)
        self.is_running = False
        
    def start(self):
        """Start the voice reminder system."""
        self.is_running = True
        
        # Start the scheduler in a separate thread
        scheduler_thread = threading.Thread(target=self.scheduler.start_monitoring)
        scheduler_thread.daemon = True  # Thread will exit when main program exits
        scheduler_thread.start()
        
        self.speaker.speak("Voice reminder system activated. I'm listening for the wake word.")
        print("Voice reminder system active. Say '{}' to start.".format(self.wake_word))
        
        try:
            while self.is_running:
                # Listen for wake word
                print("Listening for wake word...")
                audio_text = self.recognizer.listen_for_command()
                
                if not audio_text:
                    continue
                
                print(f"Heard: {audio_text}")
                
                # Check if wake word was spoken
                if self.wake_word in audio_text.lower():
                    self.speaker.speak("Yes, how can I help you?")
                    print("Wake word detected! Listening for command...")
                    
                    # Listen for the actual command
                    command_audio = self.recognizer.listen_for_command()
                    if command_audio:
                        print(f"Command heard: {command_audio}")
                        
                        # Parse the command
                        try:
                            reminder = self.parser.parse_command(command_audio)
                            if reminder:
                                # Store the reminder
                                self.reminder_store.add_reminder(reminder)
                                
                                # Confirm to the user
                                confirmation = f"Reminder set for {reminder['task']} at {reminder['time_str']}"
                                print(confirmation)
                                self.speaker.speak(confirmation)
                            else:
                                self.speaker.speak("Sorry, I couldn't understand that reminder command.")
                        except Exception as e:
                            print(f"Error parsing command: {e}")
                            self.speaker.speak("Sorry, I had trouble understanding that command.")
                
                # Slight pause to prevent CPU hogging
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nShutting down voice reminder system...")
            self.stop()
    
    def stop(self):
        """Stop the voice reminder system."""
        self.is_running = False
        self.scheduler.stop_monitoring()
        self.speaker.speak("Voice reminder system deactivated.")
        print("Voice reminder system stopped.")


if __name__ == "__main__":
    print("Initializing Voice Reminder System...")
    system = VoiceReminderSystem()
    system.start()