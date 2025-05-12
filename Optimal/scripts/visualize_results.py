# scripts/visualize_results.py
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def load_metadata_files(output_dir="output"):
    """Load all metadata files from the output directory."""
    metadata_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith("_metadata.json"):
                metadata_files.append(os.path.join(root, file))
    
    # Load metadata from each file
    metadata_list = []
    for file_path in metadata_files:
        with open(file_path, 'r') as f:
            metadata = json.load(f)
            
            # Extract profile and interaction type
            parts = file_path.split(os.sep)
            if len(parts) >= 3:
                metadata["profile"] = parts[-3]  # Assuming path structure: output/profile/time_period/metadata.json
                metadata["time_period"] = parts[-2]  # Morning or Afternoon
            
            metadata_list.append(metadata)
    
    return metadata_list

def visualize_approachability_distribution(metadata_list):
    """Visualize the distribution of approachability scores."""
    approachability_scores = [m["approachability"] for m in metadata_list]
    
    plt.figure(figsize=(10, 6))
    plt.hist(approachability_scores, bins=10, alpha=0.7)
    plt.xlabel("Approachability Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Approachability Scores")
    plt.grid(True, alpha=0.3)
    plt.savefig("output/approachability_distribution.png")
    plt.close()

def visualize_approachability_by_profile(metadata_list):
    """Visualize approachability scores by profile."""
    df = pd.DataFrame(metadata_list)
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="profile", y="approachability", data=df)
    plt.xlabel("Profile")
    plt.ylabel("Approachability Score")
    plt.title("Approachability Scores by Profile")
    plt.grid(True, alpha=0.3)
    plt.savefig("output/approachability_by_profile.png")
    plt.close()

def visualize_emotion_distribution(metadata_list):
    """Visualize the distribution of emotions."""
    emotions = [m["emotion"] for m in metadata_list]
    emotion_counts = pd.Series(emotions).value_counts()
    
    plt.figure(figsize=(10, 6))
    emotion_counts.plot(kind="bar", alpha=0.7)
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.title("Distribution of Emotions")
    plt.grid(True, alpha=0.3)
    plt.savefig("output/emotion_distribution.png")
    plt.close()

def visualize_tom_by_emotion(metadata_list):
    """Visualize relationship between ToM state and emotion."""
    df = pd.DataFrame(metadata_list)
    
    # Create a cross-tabulation of ToM and Emotion
    cross_tab = pd.crosstab(df["ToM"], df["emotion"])
    
    plt.figure(figsize=(12, 8))
    cross_tab.plot(kind="bar", stacked=True)
    plt.xlabel("Theory of Mind State")
    plt.ylabel("Count")
    plt.title("Theory of Mind State by Emotion")
    plt.legend(title="Emotion")
    plt.grid(True, alpha=0.3)
    plt.savefig("output/tom_by_emotion.png")
    plt.close()

def main():
    """Main function to generate visualizations."""
    # Check if output directory exists
    if not os.path.exists("output"):
        print("Error: Output directory not found")
        return
    
    # Load metadata
    metadata_list = load_metadata_files()
    
    if not metadata_list:
        print("Error: No metadata files found")
        return
    
    print(f"Loaded {len(metadata_list)} metadata files")
    
    # Generate visualizations
    visualize_approachability_distribution(metadata_list)
    visualize_approachability_by_profile(metadata_list)
    visualize_emotion_distribution(metadata_list)
    visualize_tom_by_emotion(metadata_list)
    
    print("Generated visualizations in output directory")

if __name__ == "__main__":
    main()