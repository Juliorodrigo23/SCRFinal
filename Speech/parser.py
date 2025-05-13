#!/usr/bin/env python3
"""
Voice-Activated Reminder System
Parser Module: Handles command parsing (intent + entity extraction)
"""

import re
import datetime
from datetime import datetime, time, timedelta

class CommandParser:
    """Class to parse voice commands into structured reminders."""
    
    def __init__(self):
        """Initialize the command parser."""
        # Common time patterns
        self.time_patterns = [
            # Simple hour patterns
            r'at (\d{1,2}) (am|pm|AM|PM)',  # "at 3 pm"
            r'at (\d{1,2})(am|pm|AM|PM)',   # "at 3pm"
            r'at (\d{1,2}) o\'?clock',      # "at 3 o'clock"
            
            # Hour and minute patterns
            r'at (\d{1,2}):(\d{2}) (am|pm|AM|PM)',  # "at 3:30 pm"
            r'at (\d{1,2}):(\d{2})',                # "at 15:30" (24-hour)
            
            # Relative time patterns
            r'in (\d+) (minute|minutes|min|mins)',  # "in 30 minutes"
            r'in (\d+) (hour|hours|hr|hrs)',        # "in 2 hours"
            r'in an hour',                          # "in an hour"
            r'in half an hour',                     # "in half an hour"
            
            # Named time patterns
            r'(this morning|this afternoon|this evening|tonight|tomorrow morning|tomorrow afternoon|tomorrow evening)'
        ]
    
    def parse_command(self, command_text):
        """
        Parse the command text into a structured reminder.
        
        Args:
            command_text (str): The command to parse
            
        Returns:
            dict: A structured reminder with 'task', 'time', and 'time_str' keys,
                  or None if parsing fails
        """
        if not command_text:
            return None
        
        # Extract time information
        time_info = self._extract_time_info(command_text)
        if not time_info:
            print("No time information found in command.")
            return None
        
        # Extract task description (everything before the time pattern)
        task_description = self._extract_task(command_text, time_info['time_str'])
        if not task_description:
            print("No task description found in command.")
            return None
        
        # Create and return the structured reminder
        return {
            'task': task_description.strip(),
            'time': time_info['time'],
            'time_str': time_info['time_str']
        }
    
    def _extract_time_info(self, text):
        """
        Extract time information from the command text.
        
        Args:
            text (str): The command text
            
        Returns:
            dict: Time information with 'time' (datetime object) and 'time_str' (original string) keys,
                  or None if no time information is found
        """
        now = datetime.now()
        
        # Check for each time pattern
        for pattern in self.time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                matched_time_str = match.group(0)
                
                # Handle "at X am/pm" patterns
                if pattern == r'at (\d{1,2}) (am|pm|AM|PM)':
                    hour = int(match.group(1))
                    am_pm = match.group(2).lower()
                    
                    if am_pm == 'pm' and hour < 12:
                        hour += 12
                    elif am_pm == 'am' and hour == 12:
                        hour = 0
                    
                    reminder_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                    
                    # If the time has already passed today, set it for tomorrow
                    if reminder_time < now:
                        reminder_time += timedelta(days=1)
                    
                    return {
                        'time': reminder_time,
                        'time_str': matched_time_str
                    }
                
                # Handle "at Xam/pm" patterns
                elif pattern == r'at (\d{1,2})(am|pm|AM|PM)':
                    hour = int(match.group(1))
                    am_pm = match.group(2).lower()
                    
                    if am_pm == 'pm' and hour < 12:
                        hour += 12
                    elif am_pm == 'am' and hour == 12:
                        hour = 0
                    
                    reminder_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                    
                    if reminder_time < now:
                        reminder_time += timedelta(days=1)
                    
                    return {
                        'time': reminder_time,
                        'time_str': matched_time_str
                    }
                
                # Handle "at X o'clock" patterns
                elif pattern == r'at (\d{1,2}) o\'?clock':
                    hour = int(match.group(1))
                    
                    # Assume afternoon hours if it's currently morning and the hour is small
                    if now.hour < 12 and hour < 7:
                        hour += 12
                    
                    reminder_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                    
                    if reminder_time < now:
                        reminder_time += timedelta(days=1)
                    
                    return {
                        'time': reminder_time,
                        'time_str': matched_time_str
                    }
                
                # Handle "at X:XX am/pm" patterns
                elif pattern == r'at (\d{1,2}):(\d{2}) (am|pm|AM|PM)':
                    hour = int(match.group(1))
                    minute = int(match.group(2))
                    am_pm = match.group(3).lower()
                    
                    if am_pm == 'pm' and hour < 12:
                        hour += 12
                    elif am_pm == 'am' and hour == 12:
                        hour = 0
                    
                    reminder_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    
                    if reminder_time < now:
                        reminder_time += timedelta(days=1)
                    
                    return {
                        'time': reminder_time,
                        'time_str': matched_time_str
                    }
                
                # Handle "at XX:XX" (24-hour) patterns
                elif pattern == r'at (\d{1,2}):(\d{2})':
                    hour = int(match.group(1))
                    minute = int(match.group(2))
                    
                    reminder_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    
                    if reminder_time < now:
                        reminder_time += timedelta(days=1)
                    
                    return {
                        'time': reminder_time,
                        'time_str': matched_time_str
                    }
                
                # Handle "in X minutes" patterns
                elif pattern == r'in (\d+) (minute|minutes|min|mins)':
                    minutes = int(match.group(1))
                    reminder_time = now + timedelta(minutes=minutes)
                    
                    return {
                        'time': reminder_time,
                        'time_str': matched_time_str
                    }
                
                # Handle "in X hours" patterns
                elif pattern == r'in (\d+) (hour|hours|hr|hrs)':
                    hours = int(match.group(1))
                    reminder_time = now + timedelta(hours=hours)
                    
                    return {
                        'time': reminder_time,
                        'time_str': matched_time_str
                    }
                
                # Handle "in an hour" pattern
                elif pattern == r'in an hour':
                    reminder_time = now + timedelta(hours=1)
                    
                    return {
                        'time': reminder_time,
                        'time_str': matched_time_str
                    }
                
                # Handle "in half an hour" pattern
                elif pattern == r'in half an hour':
                    reminder_time = now + timedelta(minutes=30)
                    
                    return {
                        'time': reminder_time,
                        'time_str': matched_time_str
                    }
                
                # Handle named time patterns
                elif pattern == r'(this morning|this afternoon|this evening|tonight|tomorrow morning|tomorrow afternoon|tomorrow evening)':
                    time_of_day = match.group(1).lower()
                    
                    # Set hours based on time of day
                    if 'morning' in time_of_day:
                        hour = 9
                    elif 'afternoon' in time_of_day:
                        hour = 14
                    elif 'evening' in time_of_day or 'tonight' in time_of_day:
                        hour = 19
                    
                    reminder_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                    
                    # If it's tomorrow or if the time has already passed today
                    if 'tomorrow' in time_of_day or reminder_time < now:
                        reminder_time += timedelta(days=1)
                    
                    return {
                        'time': reminder_time,
                        'time_str': matched_time_str
                    }
        
        return None
    
    def _extract_task(self, text, time_str):
        """
        Extract task description from the command text.
        
        Args:
            text (str): The command text
            time_str (str): The time string to remove from the text
            
        Returns:
            str: The task description, or None if no task could be extracted
        """
        # Find common reminder phrases
        remind_matches = re.search(r'remind me to (.*)', text, re.IGNORECASE)
        if remind_matches:
            task = remind_matches.group(1)
            # Remove the time part from the task
            return task.replace(time_str, '').strip()
        
        # If no specific reminder phrase, just get everything before the time
        time_index = text.lower().find(time_str.lower())
        if time_index > 0:
            # Get everything before the time string
            task = text[:time_index].strip()
            # Remove common command prefixes
            task = re.sub(r'^(set|create|add|make) (a |an |)(reminder|alarm) (to |for |about |)', '', task, flags=re.IGNORECASE)
            return task
        
        # If time is at the beginning, get everything after
        elif time_index == 0:
            task = text[len(time_str):].strip()
            task = re.sub(r'^(remind me to |remind me |to |)', '', task, flags=re.IGNORECASE)
            return task
        
        return None