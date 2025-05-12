"""
Main integration script for Habitat-Sim with YOLO for CompanionCare prototype.
This module combines environment, detection, mapping, and navigation for assisted living tasks.
"""

import os
import cv2
import numpy as np
import time
import argparse
import habitat_sim
import matplotlib.pyplot as plt
import logging

from environment import HabitatEnvironment, get_available_scenes
from yolo_detector import YOLODetector
from semantic_mapper import SemanticMapper
from navigation import ObjectGoalNavigator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CompanionCare")


class CompanionCareSystem:
    def __init__(
        self,
        scene_path,
        yolo_weights_path,
        yolo_config_path=None,
        output_dir="./outputs",
        target_objects=None,
        device="cuda" if habitat_sim.cuda_enabled else "cpu"
    ):
        """
        Initialize the CompanionCare system
        
        Args:
            scene_path (str): Path to Habitat scene file
            yolo_weights_path (str): Path to YOLO weights
            yolo_config_path (str, optional): Path to YOLO config
            output_dir (str): Directory to save outputs
            target_objects (list, optional): List of objects to detect
            device (str): Device to run YOLO on
        """
        self.output_dir = output_dir
        self.target_objects = target_objects or ["chair", "couch", "bed", "toilet", 
                                                "tv", "bottle", "cup", "bowl", "book"]
        self.device = device
        self.scene_path = scene_path
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Initialize components
        logger.info("Initializing Habitat environment...")
        self.env = HabitatEnvironment(scene_path, enable_physics=True)
        
        logger.info(f"Initializing YOLO detector with device: {device}...")
        self.detector = YOLODetector(
            weights_path=yolo_weights_path,
            config_path=yolo_config_path,
            device=device,
            target_classes=self.target_objects
        )
        
        logger.info("Initializing semantic mapper...")
        self.mapper = SemanticMapper(map_resolution=0.1, map_size=20.0)
        
        logger.info("Initializing navigator...")
        self.navigator = ObjectGoalNavigator(
            env=self.env,
            detector=self.detector,
            mapper=self.mapper,
            target_objects=self.target_objects
        )
        
        # System state
        self.is_running = False
        self.current_task = None
        self.task_history = []
        
        logger.info("CompanionCare system initialized")
    
    def run_demo(self, duration=120, explore_time=30, visualize=True):
        """
        Run a complete CompanionCare demo
        
        Args:
            duration (int): Total duration in seconds
            explore_time (int): Initial exploration time in seconds
            visualize (bool): Whether to visualize the demo
            
        Returns:
            dict: Demo results
        """
        logger.info(f"Starting CompanionCare demo (duration: {duration}s)")
        
        self.is_running = True
        start_time = time.time()
        results = {
            'exploration': None,
            'navigation_tasks': []
        }
        
        try:
            # 1. First explore the environment to build a map
            logger.info(f"Starting exploration phase ({explore_time}s)")
            results['exploration'] = self.navigator.explore_environment(
                duration=explore_time,
                visualize=visualize
            )
            
            # Save the exploration map
            map_path = os.path.join(self.output_dir, "exploration_map.png")
            self.mapper.save_map(map_path)
            
            # 2. Identify high-priority objects for ADL support
            detected_classes = list(self.mapper.semantic_map.keys())
            logger.info(f"Detected objects: {detected_classes}")
            
            # Define priority tasks based on detected objects
            adl_tasks = []
            
            # Priority 1: Medication assistance
            if "bottle" in detected_classes:
                adl_tasks.append({
                    'name': 'medication_reminder',
                    'type': 'object',
                    'target': 'bottle',
                    'priority': 1,
                    'description': 'Navigate to medication bottle for reminder'
                })
            
            # Priority 2: Hydration support
            if "cup" in detected_classes:
                adl_tasks.append({
                    'name': 'hydration_support',
                    'type': 'object',
                    'target': 'cup',
                    'priority': 2,
                    'description': 'Navigate to cup to remind about hydration'
                })
            
            # Priority 3: Food assistance
            if "bowl" in detected_classes:
                adl_tasks.append({
                    'name': 'meal_assistance',
                    'type': 'object',
                    'target': 'bowl',
                    'priority': 3,
                    'description': 'Navigate to food bowl for meal assistance'
                })
            
            # Priority 4: Mobility support
            if "chair" in detected_classes:
                adl_tasks.append({
                    'name': 'mobility_support',
                    'type': 'object',
                    'target': 'chair',
                    'priority': 4,
                    'description': 'Navigate to chair for mobility assistance'
                })
            
            # If no specific tasks, add generic tasks
            if not adl_tasks:
                # Find any detected object
                if detected_classes:
                    target = detected_classes[0]
                    adl_tasks.append({
                        'name': 'general_assistance',
                        'type': 'object',
                        'target': target,
                        'priority': 5,
                        'description': f'Navigate to {target} for general assistance'
                    })
                else:
                    # Explore a random position
                    adl_tasks.append({
                        'name': 'area_exploration',
                        'type': 'position',
                        'target': np.array([5.0, 0.0, 5.0]),  # Random position
                        'priority': 5,
                        'description': 'Explore new area'
                    })
            
            # Sort tasks by priority
            adl_tasks.sort(key=lambda x: x['priority'])
            
            # 3. Execute each task in priority order
            remaining_time = duration - (time.time() - start_time)
            
            for task in adl_tasks:
                # Skip if we're out of time
                if time.time() - start_time >= duration:
                    logger.info("Demo time limit reached")
                    break
                
                # Allocate time for this task (or remaining time)
                task_time = min(30, remaining_time)
                if task_time <= 0:
                    break
                
                logger.info(f"Starting task: {task['name']} ({task['description']})")
                task_result = self._execute_task(task, task_time, visualize)
                results['navigation_tasks'].append(task_result)
                
                # Update remaining time
                remaining_time = duration - (time.time() - start_time)
            
            # 4. Return to a "home" position if time remains
            if time.time() - start_time < duration:
                logger.info("Returning to home position")
                
                # Use the first explored position as "home"
                if self.navigator.pose_history:
                    home_position = self.navigator.pose_history[0]['position']
                    
                    home_task = {
                        'name': 'return_home',
                        'type': 'position',
                        'target': home_position,
                        'priority': 9,
                        'description': 'Return to home position'
                    }
                    
                    task_result = self._execute_task(home_task, remaining_time, visualize)
                    results['navigation_tasks'].append(task_result)
            
            # 5. Save final semantic map
            final_map_path = os.path.join(self.output_dir, "final_semantic_map.png")
            self.mapper.save_map(final_map_path)
            
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        except Exception as e:
            logger.error(f"Error during demo: {str(e)}", exc_info=True)
        finally:
            self.is_running = False
            
            # Calculate overall statistics
            total_time = time.time() - start_time
            successful_tasks = sum(1 for t in results['navigation_tasks'] if t['success'])
            
            results['summary'] = {
                'total_duration': total_time,
                'total_tasks': len(results['navigation_tasks']),
                'successful_tasks': successful_tasks,
                'success_rate': successful_tasks / max(1, len(results['navigation_tasks'])),
                'detected_objects': detected_classes
            }
            
            logger.info(f"Demo complete. Duration: {total_time:.1f}s")
            logger.info(f"Tasks: {len(results['navigation_tasks'])}, Successful: {successful_tasks}")
            
            # Close visualizations
            if visualize:
                cv2.destroyAllWindows()
            
            return results
    
    def _execute_task(self, task, max_duration, visualize):
        """Execute a specific task"""
        start_time = time.time()
        task_type = task['type']
        target = task['target']
        
        result = {
            'task_name': task['name'],
            'description': task['description'],
            'start_time': start_time,
            'type': task_type,
            'target': target,
            'success': False
        }
        
        try:
            # Start navigation based on task type
            if task_type == 'object':
                success = self.navigator.navigate_to_object(
                    object_class=target,
                    min_confidence=0.4,
                    max_age=300
                )
            elif task_type == 'position':
                success = self.navigator.navigate_to_position(target)
            else:
                logger.error(f"Unknown task type: {task_type}")
                return result
            
            # If navigation started successfully, run the loop
            if success:
                # Maximum steps based on time (assume 5 steps per second)
                max_steps = int(max_duration * 5)
                
                navigation_result = self.navigator.run_navigation_loop(
                    max_steps=max_steps,
                    visualize=visualize
                )
                
                # Update result with navigation details
                result.update(navigation_result)
            else:
                logger.warning(f"Failed to start navigation for task: {task['name']}")
                result['failure_reason'] = "Navigation start failed"
        
        except Exception as e:
            logger.error(f"Error executing task {task['name']}: {str(e)}", exc_info=True)
            result['failure_reason'] = str(e)
        
        # Record end time and duration
        end_time = time.time()
        result['end_time'] = end_time
        result['duration'] = end_time - start_time
        
        # Generate a screenshot if visualizing
        if visualize:
            try:
                # Get current RGB frame
                rgb_frame = self.env.get_rgb_frame()
                
                # Get detections
                detection_results = self.detector.detect(rgb_frame)
                vis_frame = self.detector.visualize_detections(rgb_frame, detection_results)
                
                # Add task info
                cv2.putText(vis_frame, f"Task: {task['name']}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
                
                cv2.putText(vis_frame, f"Status: {'Success' if result.get('success', False) else 'Incomplete'}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0, 200, 0) if result.get('success', False) else (0, 0, 200), 2)
                
                # Save screenshot
                screenshot_path = os.path.join(
                    self.output_dir, 
                    f"task_{task['name']}_{int(start_time)}.png"
                )
                cv2.imwrite(screenshot_path, vis_frame)
                result['screenshot'] = screenshot_path
            except Exception as e:
                logger.error(f"Error saving task screenshot: {str(e)}")
        
        logger.info(f"Task complete: {task['name']}, Success: {result.get('success', False)}")
        return result
    
    def close(self):
        """Clean up resources"""
        logger.info("Closing CompanionCare system")
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
        
        # Close any open windows
        cv2.destroyAllWindows()


def run_demo(args):
    """Run the CompanionCare demo with command-line arguments"""
    # Check if there are available scenes
    available_scenes = get_available_scenes()
    
    scene_path = args.scene
    
    # If no scene specified, use the first available one
    if scene_path is None or not os.path.exists(scene_path):
        if available_scenes:
            scene_name = list(available_scenes.keys())[0]
            scene_path = available_scenes[scene_name]
            print(f"Using available scene: {scene_name} at {scene_path}")
        else:
            print("No scenes available. Please download Habitat test scenes or specify a valid scene path.")
            print("See: https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md")
            return
    
    # Initialize the system
    system = CompanionCareSystem(
        scene_path=scene_path,
        yolo_weights_path=args.yolo_weights,
        yolo_config_path=args.yolo_config,
        output_dir=args.output_dir,
        target_objects=args.target_objects.split(',') if args.target_objects else None,
        device=args.device
    )
    
    try:
        # Run the demo
        results = system.run_demo(
            duration=args.duration,
            explore_time=args.explore_time,
            visualize=not args.no_visualization
        )
        
        # Save results to file
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            else:
                return obj
        
        results_json = convert_numpy(results)
        
        results_path = os.path.join(args.output_dir, "demo_results.json")
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"Results saved to {results_path}")
        
        # Generate summary plots
        try:
            # Plot detected objects
            if 'exploration' in results and results['exploration']:
                detected_objects = results['exploration'].get('unique_objects', [])
                if detected_objects:
                    plt.figure(figsize=(10, 6))
                    plt.bar(detected_objects, 
                            [results['exploration']['detections'].get(obj, 0) for obj in detected_objects])
                    plt.title("Detected Objects During Exploration")
                    plt.xlabel("Object Class")
                    plt.ylabel("Count")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.output_dir, "detected_objects.png"))
            
            # Plot task success/failure
            if 'navigation_tasks' in results and results['navigation_tasks']:
                task_names = [task['task_name'] for task in results['navigation_tasks']]
                task_success = [1 if task.get('success', False) else 0 for task in results['navigation_tasks']]
                
                plt.figure(figsize=(10, 6))
                plt.bar(task_names, task_success, color=['green' if s else 'red' for s in task_success])
                plt.title("Task Completion Status")
                plt.xlabel("Task")
                plt.ylabel("Success (1) / Failure (0)")
                plt.ylim(0, 1.2)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, "task_completion.png"))
        
        except Exception as e:
            print(f"Error generating summary plots: {str(e)}")
        
    finally:
        # Clean up
        system.close()


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="CompanionCare: Habitat-Sim with YOLO for assisted living")
    
    parser.add_argument('--scene', type=str, default=None,
                        help='Path to Habitat scene file (.glb)')
    
    parser.add_argument('--yolo_weights', type=str, default='yolov5s.pt',
                        help='Path to YOLO weights file')
    
    parser.add_argument('--yolo_config', type=str, default=None,
                        help='Path to YOLO config file (for Darknet models)')
    
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save outputs')
    
    parser.add_argument('--target_objects', type=str, default=None,
                        help='Comma-separated list of target object classes')
    
    parser.add_argument('--device', type=str, 
                        default="cuda" if habitat_sim.cuda_enabled else "cpu",
                        help='Device for YOLO inference (cuda/cpu)')
    
    parser.add_argument('--duration', type=int, default=120,
                        help='Total demo duration in seconds')
    
    parser.add_argument('--explore_time', type=int, default=30,
                        help='Initial exploration time in seconds')
    
    parser.add_argument('--no_visualization', action='store_true',
                        help='Disable visualization')
    
    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()