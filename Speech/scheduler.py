#!/usr/bin/env python3
"""
Voice-Activated Reminder System
Scheduler Module: Time monitoring and reminder triggering
"""

import time
import threading
from datetime import datetime, timedelta

class ReminderScheduler:
    """Class to monitor time and trigger reminders."""
    
    def __init__(self, reminder_store, speaker, check_interval=10):
        """
        Initialize the reminder scheduler.
        
        Args:
            reminder_store (ReminderStore): The store containing reminders
            speaker (Speaker): The text-to-speech component
            check_interval (int): How often to check for due reminders (in seconds)
        """
        self.reminder_store = reminder_store
        self.speaker = speaker
        self.check_interval = check_interval
        self.stop_event = threading.Event()
        self.last_check_time = None
    
    def start_monitoring(self):
        """Start monitoring for due reminders."""
        print("Starting reminder scheduler...")
        self.stop_event.clear()
        self.last_check_time = datetime.now()
        
        try:
            while not self.stop_event.is_set():
                current_time = datetime.now()
                
                # Get reminders that are due
                due_reminders = self.reminder_store.get_due_reminders(current_time)
                
                # Process due reminders
                for reminder in due_reminders:
                    self._announce_reminder(reminder)
                    self.reminder_store.mark_reminded(reminder['id'])
                
                # Update last check time
                self.last_check_time = current_time
                
                # Sleep for the check interval
                time.sleep(self.check_interval)
                
        except Exception as e:
            print(f"Error in reminder scheduler: {e}")
        
        print("Reminder scheduler stopped.")
    
    def stop_monitoring(self):
        """Stop monitoring for due reminders."""
        self.stop_event.set()
    
    def _announce_reminder(self, reminder):
        """
        Announce a reminder.
        
        Args:
            reminder (dict): The reminder to announce
        """
        task = reminder.get('task', 'Unknown task')
        time_str = reminder.get('time_str', 'now')
        
        # Create a natural-sounding announcement
        announcement = f"Reminder: It's time to {task}."
        
        # Log the announcement
        print(f"REMINDER TRIGGERED: {announcement}")
        
        # Speak the announcement
        self.speaker.speak(announcement)
        
        # Optionally, repeat the announcement after a short delay
        # threading.Timer(5, self.speaker.speak, args=(announcement,)).start()
    
    def get_next_reminder_time(self):
        """
        Get the time of the next scheduled reminder.
        
        Returns:
            datetime: Time of next reminder, or None if no reminders are scheduled
        """
        reminders = self.reminder_store.get_all_reminders(active_only=True)
        if not reminders:
            return None
        
        # Find the earliest reminder time
        now = datetime.now()
        future_reminders = [r for r in reminders if r.get('time') > now]
        
        if not future_reminders:
            return None
        
        return min(r.get('time') for r in future_reminders)
    
    def get_time_until_next_reminder(self):
        """
        Get the time until the next reminder.
        
        Returns:
            timedelta: Time until next reminder, or None if no reminders are scheduled
        """
        next_time = self.get_next_reminder_time()
        if next_time is None:
            return None
        
        return next_time - datetime.now()
    
    def get_upcoming_reminders(self, hours=24):
        """
        Get upcoming reminders for the next specified hours.
        
        Args:
            hours (int): Number of hours to look ahead
            
        Returns:
            list: List of upcoming reminders
        """
        now = datetime.now()
        end_time = now + timedelta(hours=hours)
        
        reminders = self.reminder_store.get_all_reminders(active_only=True)
        return [r for r in reminders if now < r.get('time') <= end_time]