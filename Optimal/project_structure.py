# project_structure.py
import os

def create_project_structure():
    """Create the directory structure for the project."""
    base_dirs = [
        "data",
        "config",
        "modules",
        "output"
    ]
    
    # Create base directories
    for directory in base_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Create profile-specific directories
    profiles = ["Jack", "Maria", "Chen", "Fatima", "Robert"]
    interactions = ["Morning", "Afternoon"]
    
    for profile in profiles:
        for interaction in interactions:
            os.makedirs(f"output/{profile}/{interaction}", exist_ok=True)
    
    print("Project structure created successfully.")

if __name__ == "__main__":
    create_project_structure()