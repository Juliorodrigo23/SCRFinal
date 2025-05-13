#!/usr/bin/env python3
"""
Voice-Activated Reminder System
Reminders Module: Data structure to store and manage reminders
"""

import json
import threading
import datetime
from datetime import datetime

class ReminderStore:
    """Class to store and manage reminders."""
    
    def __init__(self, storage_file=None):
        """
        Initialize the reminder store.
        
        Args:
            storage_file (str): Path to file for persistent storage (None for in-memory only)
        """
        self.reminders = []
        self.storage_file = storage_file
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Load existing reminders if storage file exists
        if storage_file:
            try:
                self.load_from_file()
            except Exception as e:
                print(f"Error loading reminders from file: {e}")
    
    def add_reminder(self, reminder):
        """
        Add a new reminder.
        
        Args:
            reminder (dict): A reminder with 'task', 'time', and 'time_str' keys
            
        Returns:
            int: The ID of the newly added reminder
        """
        with self.lock:
            # Generate a new ID
            reminder_id = len(self.reminders)
            
            # Add status and ID fields
            reminder['id'] = reminder_id
            reminder['active'] = True
            reminder['created_at'] = datetime.now()
            
            # Add to list
            self.reminders.append(reminder)
            
            # Save to file if storage is enabled
            if self.storage_file:
                self.save_to_file()
            
            return reminder_id
    
    def get_reminder(self, reminder_id):
        """
        Get a reminder by ID.
        
        Args:
            reminder_id (int): The ID of the reminder to get
            
        Returns:
            dict: The reminder with the specified ID, or None if not found
        """
        with self.lock:
            try:
                return self.reminders[reminder_id]
            except IndexError:
                return None
    
    def update_reminder(self, reminder_id, updates):
        """
        Update a reminder.
        
        Args:
            reminder_id (int): The ID of the reminder to update
            updates (dict): The fields to update
            
        Returns:
            bool: True if the update was successful, False otherwise
        """
        with self.lock:
            reminder = self.get_reminder(reminder_id)
            if not reminder:
                return False
            
            # Update fields
            for key, value in updates.items():
                reminder[key] = value
            
            # Save to file if storage is enabled
            if self.storage_file:
                self.save_to_file()
            
            return True
    
    def delete_reminder(self, reminder_id):
        """
        Delete a reminder.
        
        Args:
            reminder_id (int): The ID of the reminder to delete
            
        Returns:
            bool: True if the deletion was successful, False otherwise
        """
        with self.lock:
            reminder = self.get_reminder(reminder_id)
            if not reminder:
                return False
            
            # Mark as inactive instead of removing
            reminder['active'] = False
            
            # Save to file if storage is enabled
            if self.storage_file:
                self.save_to_file()
            
            return True
    
    def get_all_reminders(self, active_only=True):
        """
        Get all reminders.
        
        Args:
            active_only (bool): If True, only return active reminders
            
        Returns:
            list: A list of reminders
        """
        with self.lock:
            if active_only:
                return [r for r in self.reminders if r.get('active', True)]
            return self.reminders
    
    def get_due_reminders(self, current_time=None):
        """
        Get reminders that are due at or before the specified time.
        
        Args:
            current_time (datetime): The time to check against (default: now)
            
        Returns:
            list: A list of due reminders
        """
        if current_time is None:
            current_time = datetime.now()
        
        with self.lock:
            due_reminders = []
            for reminder in self.get_all_reminders(active_only=True):
                # Convert string time representation to datetime if needed
                reminder_time = reminder.get('time')
                if isinstance(reminder_time, str):
                    try:
                        reminder_time = datetime.fromisoformat(reminder_time)
                        reminder['time'] = reminder_time
                    except ValueError:
                        # Skip reminders with invalid time format
                        continue
                
                # Check if the reminder is due
                if reminder_time <= current_time:
                    due_reminders.append(reminder)
            
            return due_reminders
    
    def mark_reminded(self, reminder_id):
        """
        Mark a reminder as reminded.
        
        Args:
            reminder_id (int): The ID of the reminder
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.update_reminder(reminder_id, {
            'active': False,
            'reminded_at': datetime.now()
        })
    
    def save_to_file(self):
        """Save reminders to the storage file."""
        if not self.storage_file:
            return
        
        try:
            # Convert datetime objects to ISO format strings for JSON serialization
            reminders_copy = []
            for reminder in self.reminders:
                reminder_copy = reminder.copy()
                for key, value in reminder_copy.items():
                    if isinstance(value, datetime):
                        reminder_copy[key] = value.isoformat()
                reminders_copy.append(reminder_copy)
            
            with open(self.storage_file, 'w') as f:
                json.dump(reminders_copy, f, indent=2)
        except Exception as e:
            print(f"Error saving reminders to file: {e}")
    
    def load_from_file(self):
        """Load reminders from the storage file."""
        if not self.storage_file:
            return
        
        try:
            with open(self.storage_file, 'r') as f:
                reminders_data = json.load(f)
            
            # Convert ISO format strings back to datetime objects
            for reminder in reminders_data:
                for key in ['time', 'created_at', 'reminded_at']:
                    if key in reminder and reminder[key]:
                        reminder[key] = datetime.fromisoformat(reminder[key])
            
            self.reminders = reminders_data
        except FileNotFoundError:
            # It's okay if the file doesn't exist yet
            self.reminders = []
        except Exception as e:
            print(f"Error loading reminders from file: {e}")
            self.reminders = []