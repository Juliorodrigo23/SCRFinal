# main.py
import os
import argparse
import json
from typing import List, Dict, Any

from config.config import Config
from modules.profile_module import PatientProfile, get_all_profiles, get_routine_events
from modules.dialogue_generator import DialogueGenerator
from modules.audio_synthesizer import AudioSynthesizer
from modules.text_analysis import TextAnalyzer
from modules.tom_analyzer import ToMAnalyzer
from modules.image_generator import ImageGenerator
from modules.dataset_packager import DatasetPackager
from modules.evaluation import ApproachabilityEvaluator
from project_structure import create_project_structure

def generate_interaction(
    profile: PatientProfile,
    routine_event: str,
    config: Config,
    dialogue_generator: DialogueGenerator,
    audio_synthesizer: AudioSynthesizer,
    text_analyzer: TextAnalyzer,
    tom_analyzer: ToMAnalyzer,
    image_generator: ImageGenerator,
    dataset_packager: DatasetPackager
) -> Dict[str, Any]:
    """Generate a complete interaction for a profile and routine event."""
    print(f"Generating interaction for {profile.name} - {routine_event}...")
    
    # 1. Generate dialogue
    dialogue = dialogue_generator.generate_dialogue(profile, routine_event)
    print(f"Generated dialogue with {len(dialogue)} utterances")
    
    # 2. Analyze text
    dialogue_analysis = text_analyzer.analyze_dialogue(dialogue)
    print(f"Analyzed dialogue: Sentiment = {dialogue_analysis['overall_sentiment']}")
    
    # 3. Analyze Theory of Mind
    tom_analysis = tom_analyzer.analyze_interaction(profile, dialogue, dialogue_analysis)
    print(f"ToM analysis: Approachability = {tom_analysis['approachability']}, ToM = {tom_analysis['ToM']}")
    
    # 4. Synthesize audio
    audio_files = audio_synthesizer.process_dialogue(dialogue, profile.name, routine_event)
    print(f"Generated {len(audio_files)} audio files")
    
    # 5. Generate image
    image_path = image_generator.generate_image(profile, routine_event, tom_analysis["emotion"])
    print(f"Generated image at {image_path}")
    
    # 6. Package dataset
    package_result = dataset_packager.package_interaction(
        profile_name=profile.name,
        routine_event=routine_event,
        dialogue=dialogue,
        audio_files=audio_files,
        tom_analysis=tom_analysis,
        image_path=image_path
    )
    print(f"Packaged interaction data at {package_result['metadata']}")
    
    return {
        "profile": profile.name,
        "routine_event": routine_event,
        "metadata": tom_analysis,
        "package_paths": package_result
    }

def main():
    """Main function to orchestrate the entire process."""
    parser = argparse.ArgumentParser(description="Generate multimodal interaction dataset")
    parser.add_argument("--profiles", nargs="+", help="Specific profiles to generate for")
    parser.add_argument("--evaluate-only", action="store_true", help="Only run evaluation on existing data")
    args = parser.parse_args()
    
    # Create project structure
    create_project_structure()
    
    # Initialize configuration
    config = Config()
    
    # Check for API keys
    if not config.get("api_keys.openai") or not config.get("api_keys.elevenlabs"):
        print("Error: API keys for OpenAI and/or ElevenLabs not set")
        print("Please set them in config/config.json or as environment variables")
        return
    
    # Initialize components
    dialogue_generator = DialogueGenerator(config)
    audio_synthesizer = AudioSynthesizer(config)
    text_analyzer = TextAnalyzer()
    tom_analyzer = ToMAnalyzer(text_analyzer)
    image_generator = ImageGenerator(config)
    dataset_packager = DatasetPackager()
    approachability_evaluator = ApproachabilityEvaluator()
    
    # Skip generation if evaluation only
    if not args.evaluate_only:
        # Get profiles and routine events
        profiles = get_all_profiles()
        routine_events = get_routine_events()
        
        # Filter profiles if specified
        if args.profiles:
            profiles = [p for p in profiles if p.name in args.profiles]
        
        # Generate interactions for each profile and routine event
        results = []
        for profile in profiles:
            for routine_event in routine_events[profile.name]:
                result = generate_interaction(
                    profile=profile,
                    routine_event=routine_event,
                    config=config,
                    dialogue_generator=dialogue_generator,
                    audio_synthesizer=audio_synthesizer,
                    text_analyzer=text_analyzer,
                    tom_analyzer=tom_analyzer,
                    image_generator=image_generator,
                    dataset_packager=dataset_packager
                )
                results.append(result)
        
        # Save results summary
        with open("output/generation_results.json", "w") as f:
            json.dump({
                "num_interactions": len(results),
                "profiles": [r["profile"] for r in results],
                "routine_events": [r["routine_event"] for r in results]
            }, f, indent=2)
    
    # Evaluate dataset
    print("\nEvaluating dataset...")
    evaluation_metrics = approachability_evaluator.evaluate_dataset()
    feature_correlations = approachability_evaluator.calculate_correlation()
    
    # Save evaluation results
    with open("output/evaluation_results.json", "w") as f:
        json.dump({
            "metrics": evaluation_metrics,
            "correlations": feature_correlations
        }, f, indent=2)
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {evaluation_metrics['accuracy']:.4f}")
    print(f"Precision: {evaluation_metrics['precision']:.4f}")
    print(f"Recall: {evaluation_metrics['recall']:.4f}")
    print(f"F1 Score: {evaluation_metrics['f1_score']:.4f}")
    print(f"Number of samples: {evaluation_metrics['num_samples']}")
    
    print("\nFeature Correlations with Approachability:")
    for feature, correlation in feature_correlations.items():
        print(f"{feature}: {correlation:.4f}")

if __name__ == "__main__":
    main()