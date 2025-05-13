#!/usr/bin/env python3
"""
Voice-Activated Reminder System
Utils Module: Helper functions for the voice reminder system
"""

import re
import datetime
from datetime import datetime, timedelta

def format_time(dt):
    """
    Format a datetime object in a human-readable format.
    
    Args:
        dt (datetime): The datetime to format
        
    Returns:
        str: The formatted datetime string
    """
    now = datetime.now()
    today = now.date()
    tomorrow = today + timedelta(days=1)
    
    # Format for today/tomorrow
    if dt.date() == today:
        return f"today at {dt.strftime('%I:%M %p').lstrip('0')}"
    elif dt.date() == tomorrow:
        return f"tomorrow at {dt.strftime('%I:%M %p').lstrip('0')}"
    else:
        return dt.strftime('%A, %B %d at %I:%M %p').lstrip('0')

def format_timedelta(td):
    """
    Format a timedelta in a human-readable format.
    
    Args:
        td (timedelta): The timedelta to format
        
    Returns:
        str: The formatted timedelta string
    """
    seconds = int(td.total_seconds())
    
    if seconds < 0:
        return "overdue"
    
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    
    if not parts:
        return "less than a minute"
    
    if len(parts) == 1:
        return parts[0]
    
    return ", ".join(parts[:-1]) + " and " + parts[-1]

def clean_text(text):
    """
    Clean and normalize text for better processing.
    
    Args:
        text (str): The text to clean
        
    Returns:
        str: The cleaned text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Normalize common words and abbreviations
    replacements = {
        "p.m.": "pm",
        "a.m.": "am",
        "p. m.": "pm",
        "a. m.": "am",
        "oclock": "o'clock",
        "o clock": "o'clock",
        "tomorrow": "tomorrow",
        "mins": "minutes",
        "min": "minutes"
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

def extract_numbers(text):
    """
    Extract numbers from text.
    
    Args:
        text (str): The text to extract numbers from
        
    Returns:
        list: The extracted numbers
    """
    # Extract numbers (both digits and words)
    digit_numbers = re.findall(r'\b\d+\b', text)
    
    # Convert word numbers to digits
    word_to_number = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
        'seventy': 70, 'eighty': 80, 'ninety': 90
    }
    
    word_numbers = []
    for word, number in word_to_number.items():
        if re.search(r'\b' + word + r'\b', text):
            word_numbers.append(number)
    
    # Combine and convert to integers
    all_numbers = digit_numbers + [str(n) for n in word_numbers]
    return [int(n) for n in all_numbers]

def is_time_reference(text):
    """
    Check if text contains a time reference.
    
    Args:
        text (str): The text to check
        
    Returns:
        bool: True if text contains a time reference, False otherwise
    """
    time_keywords = [
        'now', 'today', 'tomorrow', 'morning', 'afternoon', 'evening', 'night',
        'minute', 'hour', 'oclock', "o'clock", 'am', 'pm'
    ]
    
    for keyword in time_keywords:
        if re.search(r'\b' + keyword + r'\b', text, re.IGNORECASE):
            return True
    
    # Check for time patterns (XX:XX)
    if re.search(r'\b\d{1,2}:\d{2}\b', text):
        return True
    
    # Check for hour references
    if re.search(r'\b\d{1,2}\s*(am|pm|a\.m\.|p\.m\.)\b', text, re.IGNORECASE):
        return True
    
    return False

def get_next_occurrence(hour, minute=0, must_be_future=True):
    """
    Get the next occurrence of a specific time.
    
    Args:
        hour (int): The hour (0-23)
        minute (int): The minute (0-59)
        must_be_future (bool): If True, return a future time even if it means advancing to tomorrow
        
    Returns:
        datetime: The next occurrence of the specified time
    """
    now = datetime.now()
    next_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    if must_be_future and next_time <= now:
        next_time += timedelta(days=1)
    
    return next_time