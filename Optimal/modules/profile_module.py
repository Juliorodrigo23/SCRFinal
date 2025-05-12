# modules/profile_module.py
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class PatientProfile:
    name: str
    age: int
    gender: str
    ethnicity: str
    hair: str
    personality: str
    cognitive_level: str
    notes: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary format."""
        return {
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "ethnicity": self.ethnicity,
            "hair": self.hair,
            "personality": self.personality,
            "cognitive_level": self.cognitive_level,
            "notes": self.notes
        }
    
    def get_prompt_context(self) -> str:
        """Create a contextual prompt for dialogue generation."""
        cognitive_map = {
            "Mild": "occasionally forgets recent events but can maintain conversations",
            "Moderate": "significant memory issues and may become confused during conversations",
            "Severe": "severe memory loss and difficulty communicating clearly"
        }
        
        context = f"{self.name} is a {self.age}-year-old {self.ethnicity} {self.gender} with {self.hair} hair. "
        context += f"They have a {self.personality.lower()} personality. "
        context += f"They have {self.cognitive_level.lower()} dementia, which means they {cognitive_map.get(self.cognitive_level, '')}. "
        context += f"Additional notes: {self.notes}."
        
        return context


def get_all_profiles() -> List[PatientProfile]:
    """Return all predefined patient profiles."""
    profiles = [
        PatientProfile(
            name="Jack",
            age=84,
            gender="Male",
            ethnicity="White",
            hair="Buzzcut",
            personality="Grumpy, detail-oriented",
            cognitive_level="Moderate",
            notes="Ex-military, values punctuality and order"
        ),
        PatientProfile(
            name="Maria",
            age=78,
            gender="Female",
            ethnicity="Hispanic",
            hair="Grey bun",
            personality="Warm, spiritual",
            cognitive_level="Mild",
            notes="Religious routine, prays every morning"
        ),
        PatientProfile(
            name="Chen",
            age=82,
            gender="Male",
            ethnicity="Chinese",
            hair="Short white",
            personality="Quiet, observant",
            cognitive_level="Severe",
            notes="Likes Tai Chi, communicates more with gestures than words"
        ),
        PatientProfile(
            name="Fatima",
            age=79,
            gender="Female",
            ethnicity="Middle Eastern",
            hair="Covered",
            personality="Curious, family-oriented",
            cognitive_level="Moderate",
            notes="Tech-anxious, prefers traditional approaches"
        ),
        PatientProfile(
            name="Robert",
            age=81,
            gender="Male",
            ethnicity="Black",
            hair="Balding",
            personality="Jovial, social",
            cognitive_level="Mild",
            notes="Ex-teacher, enjoys telling stories about his teaching days"
        )
    ]
    return profiles

def get_profile_by_name(name: str) -> PatientProfile:
    """Get a specific profile by name."""
    for profile in get_all_profiles():
        if profile.name.lower() == name.lower():
            return profile
    raise ValueError(f"Profile with name '{name}' not found.")


def get_routine_events() -> Dict[str, List[str]]:
    """Define routine events for each profile."""
    routine_events = {
        "Jack": ["Morning - Getting dressed", "Afternoon - Medication time"],
        "Maria": ["Morning - Prayer routine", "Afternoon - Family visit"],
        "Chen": ["Morning - Tai Chi session", "Afternoon - Nap time"],
        "Fatima": ["Morning - Breakfast", "Afternoon - Looking at old photos"],
        "Robert": ["Morning - Reading newspaper", "Afternoon - Group activity"]
    }
    return routine_events