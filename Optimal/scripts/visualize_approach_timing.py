# scripts/visualize_approach_timing.py
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict, Any

def load_timing_data(output_dir="output"):
    """Load all approach timing data from the output directory."""
    timing_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file == "approach_timing_data.json":
                timing_files.append(os.path.join(root, file))
    
    # Load timing data from each file
    all_timing_data = []
    profile_event_mapping = {}
    
    for file_path in timing_files:
        with open(file_path, 'r') as f:
            timing_data = json.load(f)
            
            # Extract profile and interaction type from path
            path_parts = file_path.split(os.sep)
            if len(path_parts) >= 3:
                profile = path_parts[-3]
                time_period = path_parts[-2]
                
                # Add to the mapping
                key = f"{profile}_{time_period}"
                profile_event_mapping[key] = {
                    "profile": profile,
                    "time_period": time_period
                }
                
                # Annotate timing data with profile and event
                for result in timing_data.get("timing_results", []):
                    result["profile"] = profile
                    result["time_period"] = time_period
                    all_timing_data.append(result)
    
    return all_timing_data, profile_event_mapping

def visualize_approach_patterns(timing_data):
    """Visualize patterns in approach timing decisions."""
    df = pd.DataFrame(timing_data)
    
    plt.figure(figsize=(15, 10))
    
    # 1. Analyze approach score distribution by profile
    plt.subplot(2, 2, 1)
    sns.boxplot(x="profile", y="approach_score", data=df)
    plt.xlabel("Patient Profile")
    plt.ylabel("Approach Score")
    plt.title("Approach Score Distribution by Patient Profile")
    plt.grid(True, alpha=0.3)
    
    # 2. Analyze confidence distribution by time period
    plt.subplot(2, 2, 2)
    sns.boxplot(x="time_period", y="confidence", data=df)
    plt.xlabel("Time Period")
    plt.ylabel("Confidence")
    plt.title("Confidence Distribution by Time Period")
    plt.grid(True, alpha=0.3)
    
    # 3. Analyze factors that contribute to approach decisions
    plt.subplot(2, 2, 3)
    
    # Extract the factors
    audio_factors = [d.get("audio_factors", {}).get("sentiment_score", 0) for d in timing_data]
    visual_factors = [(d.get("visual_factors", {}).get("engagement", 0) + 
                      d.get("visual_factors", {}).get("attention", 0))/2 for d in timing_data]
    tom_factors = [d.get("tom_factors", {}).get("receptiveness", 0) for d in timing_data]
    context_factors = [d.get("context_factors", {}).get("context_score", 0) for d in timing_data]
    
    factor_labels = ["Audio", "Visual", "ToM", "Context"]
    factor_data = [audio_factors, visual_factors, tom_factors, context_factors]
    
    plt.boxplot(factor_data, labels=factor_labels)
    plt.ylabel("Factor Score")
    plt.title("Distribution of Factors Influencing Approach Decisions")
    plt.grid(True, alpha=0.3)
    
    # 4. Analyze approach decisions over time segments
    plt.subplot(2, 2, 4)
    
    # Ensure timestamps are ordered
    timestamps = sorted(df["timestamp"].unique())
    # Calculate percentage of 'Yes' decisions at each timestamp
    yes_percentages = []
    
    for ts in timestamps:
        segment_data = df[df["timestamp"] == ts]
        yes_count = segment_data["recommend_approach"].sum()
        total_count = len(segment_data)
        yes_percentages.append(yes_count / total_count * 100 if total_count > 0 else 0)
    
    plt.plot(timestamps, yes_percentages, 'o-', color='blue')
    plt.xlabel("Time Segment")
    plt.ylabel("% of 'Yes' Approach Decisions")
    plt.title("Approach Decision Trends Over Time")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("output/approach_patterns_analysis.png")
    plt.close()
    
    return "output/approach_patterns_analysis.png"

def visualize_optimal_windows(output_dir="output"):
    """Visualize the optimal approach windows."""
    # Find all optimal window data
    window_data = []
    
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file == "approach_timing_data.json":
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    if "optimal_windows" in data and data["optimal_windows"]:
                        # Extract profile and event
                        path_parts = file_path.split(os.sep)
                        if len(path_parts) >= 3:
                            profile = path_parts[-3]
                            event = path_parts[-2]
                            
                            for window in data["optimal_windows"]:
                                window["profile"] = profile
                                window["event"] = event
                                window_data.append(window)
    
    if not window_data:
        print("No optimal window data found")
        return ""
    
    # Create DataFrame for analysis
    df = pd.DataFrame(window_data)
    
    plt.figure(figsize=(12, 8))
    
    # Plot window durations by profile
    plt.subplot(2, 1, 1)
    sns.barplot(x="profile", y="duration", data=df)
    plt.xlabel("Patient Profile")
    plt.ylabel("Window Duration (segments)")
    plt.title("Optimal Approach Window Duration by Patient Profile")
    plt.grid(True, alpha=0.3)
    
    # Plot window quality (score * confidence) by event
    plt.subplot(2, 1, 2)
    df["window_quality"] = df["avg_score"] * df["avg_confidence"]
    sns.barplot(x="event", y="window_quality", data=df)
    plt.xlabel("Time Period")
    plt.ylabel("Window Quality (Score × Confidence)")
    plt.title("Optimal Approach Window Quality by Time Period")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig("output/optimal_approach_windows.png")
    plt.close()
    
    return "output/optimal_approach_windows.png"

def main():
    """Main function to generate visualizations."""
    # Check if output directory exists
    if not os.path.exists("output"):
        print("Error: Output directory not found")
        return
    
    # Load and visualize timing data
    timing_data, profile_event_mapping = load_timing_data()
    
    if not timing_data:
        print("No timing data found")
        return
    
    print(f"Loaded timing data for {len(timing_data)} segments")
    
    # Generate visualizations
    patterns_viz = visualize_approach_patterns(timing_data)
    windows_viz = visualize_optimal_windows()
    
    print(f"Generated approach timing visualizations:")
    print(f"- Approach patterns: {patterns_viz}")
    print(f"- Optimal windows: {windows_viz}")

if __name__ == "__main__":
    main()