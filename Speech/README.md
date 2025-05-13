# Voice-Activated Reminder System

A simple voice-activated reminder system that listens for voice commands, parses them into structured reminders, and announces them at the specified time.

## Features

- Wake word detection ("computer")
- Speech-to-text conversion using Google Web Speech API
- Natural language command parsing
- Text-to-speech reminder announcements
- Background scheduler that monitors for due reminders

## Requirements

- Python 3.6 or higher
- Required libraries (see requirements.txt)

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/voice-reminder.git
cd voice-reminder
```

2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Run the system:
```
python main.py
```

2. Wait for the system to initialize and calibrate.

3. Say the wake word "computer" followed by your reminder command, such as:
   - "Computer, remind me to take medication at 3 PM"
   - "Computer, set a reminder for the meeting at 2:30 PM"
   - "Computer, remind me to call mom in 30 minutes"

4. The system will confirm your reminder and announce it at the specified time.

## Example Commands

- "Computer, remind me to take medication at 3 PM"
- "Computer, set a reminder for the meeting at 2:30 PM"
- "Computer, remind me to call mom in 30 minutes"
- "Computer, remind me to check the oven in half an hour"
- "Computer, remind me to water the plants tomorrow morning"

## Project Structure

- `main.py` - Entry point and main loop
- `recognizer.py` - Speech recognition functionality
- `speaker.py` - Text-to-speech functionality
- `parser.py` - Command parsing
- `reminders.py` - Reminder storage and management
- `scheduler.py` - Background scheduler for reminders
- `utils.py` - Helper functions

## Customization

- To change the wake word, modify the `wake_word` parameter in the `VoiceReminderSystem` initialization
- To adjust speech recognition settings, modify the parameters in the `Recognizer` class
- To adjust text-to-speech settings, modify the parameters in the `Speaker` class

## Limitations

- The system requires an internet connection for speech recognition (Google Web Speech API)
- Command parsing is based on regular expressions and may not handle complex or ambiguous commands
- The system runs in the foreground and must remain running to deliver reminders

## Future Improvements

- Add persistent storage for reminders
- Improve natural language understanding
- Add offline speech recognition
- Add a GUI
- Add reminder editing and deletion commands
- Add reminder repetition (daily, weekly, etc.)