"""
Navigation module for Habitat-Sim with YOLO integration.
This module handles path planning and navigation based on semantic map data.
"""

import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from collections import deque


class ObjectGoalNavigator:
    def __init__(self, env, detector, mapper, target_objects=None):
        """
        Initialize object goal navigator
        
        Args:
            env (HabitatEnvironment): Habitat environment
            detector (YOLODetector): YOLO detector
            mapper (SemanticMapper): Semantic mapper
            target_objects (list, optional): List of target object classes
        """
        self.env = env
        self.detector = detector
        self.mapper = mapper
        self.target_objects = target_objects or []
        
        # Navigation state
        self.current_goal = None
        self.current_path = []
        self.navigation_active = False
        self.arrival_distance = 0.5  # Distance considered as "arrived"
        
        # Navigation stats
        self.nav_stats = {
            'goals_reached': 0,
            'goals_failed': 0,
            'total_distance': 0.0,
            'navigation_time': 0.0,
            'detections': {}  # Counts by class
        }
        
        print(f"Navigator initialized with targets: {self.target_objects}")
    
    def set_target_objects(self, target_objects):
        """Set or update target object classes"""
        self.target_objects = target_objects
        print(f"Target objects updated: {self.target_objects}")
    
    def process_current_view(self):
        """
        Process current camera view with detector and update semantic map
        
        Returns:
            dict: Detection results
        """
        # Get current RGB and depth observations
        obs = self.env.get_observation()
        rgb_frame = obs["rgb"]
        depth_frame = obs["depth"]
        
        # Get agent state
        agent_state = self.env.agent.get_state()
        position = agent_state.position
        rotation = agent_state.rotation
        
        # Update mapper with agent pose
        self.mapper.update_pose(position, rotation)
        
        # Detect objects in RGB frame
        detection_results = self.detector.detect(rgb_frame)
        detections = detection_results['detections']
        
        # Update detection stats
        for det in detections:
            class_name = det['class_name']
            if class_name not in self.nav_stats['detections']:
                self.nav_stats['detections'][class_name] = 0
            self.nav_stats['detections'][class_name] += 1
        
        # Camera parameters (FOV, etc.)
        camera_params = {
            'hfov': 90.0,  # Horizontal FOV in degrees
            'width': rgb_frame.shape[1],
            'height': rgb_frame.shape[0]
        }
        
        # Update semantic map with detections
        object_positions = self.mapper.update_from_detection(
            detections, depth_frame, camera_params, position, rotation
        )
        
        # Update any active navigation goal
        if self.navigation_active:
            self._update_navigation()
        
        return {
            'detections': detections,
            'fps': detection_results['fps'],
            'object_positions': object_positions
        }
    
    def _update_navigation(self):
        """Update navigation state and check for goal completion"""
        if not self.current_goal or not self.navigation_active:
            return
        
        # Get current position
        agent_state = self.env.agent.get_state()
        position = agent_state.position
        
        # Calculate distance to goal
        goal_position = self.current_goal['position']
        distance = np.linalg.norm(position - goal_position)
        
        # Check if we've reached the goal
        if distance <= self.arrival_distance:
            print(f"Goal reached: {self.current_goal['type']} (distance: {distance:.2f}m)")
            self.navigation_active = False
            self.nav_stats['goals_reached'] += 1
            return True
        
        # Check if path needs updating
        if len(self.current_path) == 0:
            # Try to replan path
            self._plan_path_to_goal()
            
            # If still no path, consider it a failure
            if len(self.current_path) == 0:
                print(f"Failed to find path to goal: {self.current_goal['type']}")
                self.navigation_active = False
                self.nav_stats['goals_failed'] += 1
                return False
        
        # Move toward next waypoint
        next_waypoint = self.current_path[0]
        waypoint_distance = np.linalg.norm(position - next_waypoint)
        
        # If we've reached this waypoint, remove it
        if waypoint_distance <= self.arrival_distance:
            self.current_path.pop(0)
            
            # If that was the last waypoint, we're done
            if len(self.current_path) == 0:
                print(f"Final waypoint reached, but goal not detected as reached. Distance: {distance:.2f}m")
                # This can happen if the goal position was updated since path planning
                
        # Return in-progress status
        return False
    
    def _plan_path_to_goal(self):
        """Plan a path to the current goal"""
        if not self.current_goal:
            return []
        
        # Get current position
        agent_state = self.env.agent.get_state()
        position = agent_state.position
        
        # Different path planning based on goal type
        if self.current_goal['type'] == 'object':
            # Get path to object from semantic map
            object_class = self.current_goal['object_class']
            path = self.mapper.find_path_to_object(
                object_class, position, 
                min_confidence=0.5, 
                max_age=300
            )
            
            if path:
                self.current_path = path
                return True
            else:
                print(f"No path found to {object_class}")
                return False
                
        elif self.current_goal['type'] == 'position':
            # Direct path to position
            self.current_path = [self.current_goal['position']]
            return True
            
        return False
    
    def navigate_to_object(self, object_class, min_confidence=0.5, max_age=300):
        """
        Start navigation to find an object of specified class
        
        Args:
            object_class (str): Object class to navigate to
            min_confidence (float): Minimum detection confidence
            max_age (float): Maximum age of detection in seconds
            
        Returns:
            bool: True if navigation started, False otherwise
        """
        # Get current position
        agent_state = self.env.agent.get_state()
        position = agent_state.position
        
        # Find nearest object of this class
        obj = self.mapper.get_nearest_object(
            object_class, position, 
            min_confidence=min_confidence, 
            max_age=max_age
        )
        
        if obj is None:
            print(f"No {object_class} found in semantic map")
            return False
            
        # Set as current goal
        self.current_goal = {
            'type': 'object',
            'object_class': object_class,
            'position': obj['world_position'],
            'start_time': time.time(),
            'start_position': position.copy(),
            'confidence': obj['confidence']
        }
        
        # Plan path
        success = self._plan_path_to_goal()
        
        if success:
            print(f"Starting navigation to {object_class} at {obj['world_position']}")
            self.navigation_active = True
            return True
        else:
            print(f"Failed to plan path to {object_class}")
            self.current_goal = None
            return False
    
    def navigate_to_position(self, position):
        """
        Start navigation to a specific position
        
        Args:
            position (numpy.ndarray): Target position [x, y, z]
            
        Returns:
            bool: True if navigation started, False otherwise
        """
        # Get current position
        agent_state = self.env.agent.get_state()
        current_position = agent_state.position
        
        # Set as current goal
        self.current_goal = {
            'type': 'position',
            'position': position,
            'start_time': time.time(),
            'start_position': current_position.copy()
        }
        
        # Plan path
        success = self._plan_path_to_goal()
        
        if success:
            print(f"Starting navigation to position {position}")
            self.navigation_active = True
            return True
        else:
            print(f"Failed to plan path to position {position}")
            self.current_goal = None
            return False
    
    def stop_navigation(self):
        """Stop active navigation"""
        if self.navigation_active:
            print("Navigation stopped")
            self.navigation_active = False
            self.current_path = []
            # Don't reset current_goal so we can resume if needed
    
    def resume_navigation(self):
        """Resume previous navigation if stopped"""
        if not self.navigation_active and self.current_goal is not None:
            print(f"Resuming navigation to {self.current_goal['type']}")
            self._plan_path_to_goal()
            self.navigation_active = True
            return True
        return False
    
    def step_navigation(self):
        """
        Take one navigation step toward goal
        
        Returns:
            bool: True if navigation is complete, False if still in progress
        """
        if not self.navigation_active:
            return False
            
        # If no path or empty path, try to plan
        if len(self.current_path) == 0:
            success = self._plan_path_to_goal()
            if not success or len(self.current_path) == 0:
                print("No valid path found")
                self.navigation_active = False
                self.nav_stats['goals_failed'] += 1
                return True  # Navigation is complete (failed)
        
        # Get next waypoint
        next_waypoint = self.current_path[0]
        
        # Use the Habitat navigate_to method to move toward waypoint
        success = self.env.navigate_to(next_waypoint, tolerance=0.5)
        
        # Update navigation state
        return self._update_navigation()
    
    def run_navigation_loop(self, max_steps=100, visualize=True):
        """
        Run navigation loop until goal is reached or max steps
        
        Args:
            max_steps (int): Maximum number of steps to take
            visualize (bool): Whether to visualize the navigation
            
        Returns:
            dict: Navigation results
        """
        if not self.navigation_active:
            print("No active navigation goal")
            return False
            
        start_time = time.time()
        steps_taken = 0
        agent_path = []
        
        # Record starting position
        agent_state = self.env.agent.get_state()
        start_position = agent_state.position.copy()
        agent_path.append(start_position)
        
        # Navigation loop
        while self.navigation_active and steps_taken < max_steps:
            # Process current view (updates detections and map)
            self.process_current_view()
            
            # Take a navigation step
            completed = self.step_navigation()
            
            # Record current position
            agent_state = self.env.agent.get_state()
            current_position = agent_state.position.copy()
            agent_path.append(current_position)
            
            # Update distance traveled
            if len(agent_path) >= 2:
                segment_distance = np.linalg.norm(agent_path[-1] - agent_path[-2])
                self.nav_stats['total_distance'] += segment_distance
            
            # Visualize if requested
            if visualize:
                # Get RGB observation
                rgb_frame = self.env.get_rgb_frame()
                
                # Create visualization with detections
                detection_results = self.detector.detect(rgb_frame)
                vis_frame = self.detector.visualize_detections(rgb_frame, detection_results)
                
                # Show navigation info on frame
                goal_info = f"Goal: {self.current_goal['type']}"
                if self.current_goal['type'] == 'object':
                    goal_info += f" ({self.current_goal['object_class']})"
                
                cv2.putText(vis_frame, goal_info, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show distance to goal
                distance = np.linalg.norm(current_position - self.current_goal['position'])
                cv2.putText(vis_frame, f"Distance: {distance:.2f}m", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display visualization
                cv2.imshow("Navigation View", vis_frame)
                
                # Get semantic map visualization
                map_vis = self.mapper.visualize_map(agent_position=current_position)
                cv2.imshow("Semantic Map", map_vis)
                
                # Update display
                cv2.waitKey(1)
            
            steps_taken += 1
            
            # If navigation completed, break
            if completed:
                break
        
        # Record navigation time
        elapsed_time = time.time() - start_time
        self.nav_stats['navigation_time'] += elapsed_time
        
        # Calculate straight-line distance
        direct_distance = np.linalg.norm(current_position - start_position)
        
        # Final distance to goal
        final_distance = np.linalg.norm(current_position - self.current_goal['position'])
        
        # Prepare results
        results = {
            'success': not self.navigation_active,  # If not active, we either succeeded or hit max steps
            'steps_taken': steps_taken,
            'distance_traveled': self.nav_stats['total_distance'],
            'direct_distance': direct_distance,
            'final_distance': final_distance,
            'path_efficiency': direct_distance / max(0.1, self.nav_stats['total_distance']),
            'time_elapsed': elapsed_time,
            'goal_type': self.current_goal['type'],
            'path': agent_path
        }
        
        print(f"Navigation complete:")
        print(f"  Success: {results['success']}")
        print(f"  Steps: {steps_taken}, Distance: {results['distance_traveled']:.2f}m")
        print(f"  Final distance to goal: {final_distance:.2f}m")
        print(f"  Path efficiency: {results['path_efficiency']:.2f}")
        
        return results

    def explore_environment(self, duration=60, visualize=True):
        """
        Explore the environment to build semantic map
        
        Args:
            duration (float): Duration to explore in seconds
            visualize (bool): Whether to visualize exploration
            
        Returns:
            dict: Exploration statistics
        """
        print(f"Starting exploration for {duration} seconds")
        
        start_time = time.time()
        steps_taken = 0
        detections_count = 0
        explored_cells = set()
        
        # Track current exploration target
        exploration_target = None
        
        # Simple state machine for exploration
        exploration_state = "scan"  # "scan", "move", "rotate"
        scan_duration = 5.0
        scan_start_time = time.time()
        
        while time.time() - start_time < duration:
            # Get current position
            agent_state = self.env.agent.get_state()
            position = agent_state.position
            
            # Process current view (updates detections and map)
            results = self.process_current_view()
            
            # Track unique explored cells
            map_pos = self.mapper._world_to_map(position)
            explored_cells.add(map_pos)
            
            # Count detections
            detections_count += len(results['detections'])
            
            # Update state machine
            current_time = time.time()
            
            if exploration_state == "scan":
                # Rotate in place to scan surroundings
                self.env.perform_action("turn_left")
                
                # After scan duration, switch to move state
                if current_time - scan_start_time > scan_duration:
                    exploration_state = "move"
                    
                    # Choose a random unexplored direction
                    angles = [0, 45, 90, 135, 180, 225, 270, 315]  
                    angle = np.random.choice(angles)
                    
                    # Convert to radians
                    angle_rad = angle * np.pi / 180
                    
                    # Calculate target position (5m away)
                    dx = 5.0 * np.cos(angle_rad)
                    dz = 5.0 * np.sin(angle_rad)
                    
                    exploration_target = np.array([
                        position[0] + dx,
                        position[1],
                        position[2] + dz
                    ])
                    
                    print(f"Moving to new location: {exploration_target}")
            
            elif exploration_state == "move":
                # Move to exploration target
                if exploration_target is not None:
                    # Check if we've reached target or are blocked
                    distance = np.linalg.norm(position - exploration_target)
                    
                    if distance < 1.0:
                        print("Reached exploration target")
                        exploration_state = "scan"
                        scan_start_time = current_time
                    else:
                        # Try to navigate to target
                        success = self.env.navigate_to(exploration_target, tolerance=1.0)
                        
                        # If stuck, switch back to scanning
                        if not success:
                            print("Navigation failed, switching to scan")
                            exploration_state = "scan"
                            scan_start_time = current_time
                else:
                    exploration_state = "scan"
                    scan_start_time = current_time
            
            # Visualize if requested
            if visualize and steps_taken % 5 == 0:  # Update visualization every 5 steps
                # Get RGB observation
                rgb_frame = self.env.get_rgb_frame()
                
                # Create visualization with detections
                detection_results = self.detector.detect(rgb_frame)
                vis_frame = self.detector.visualize_detections(rgb_frame, detection_results)
                
                # Show exploration info
                cv2.putText(vis_frame, f"Exploring: {exploration_state}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                elapsed = current_time - start_time
                cv2.putText(vis_frame, f"Time: {elapsed:.1f}s / {duration:.1f}s", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.putText(vis_frame, f"Detections: {detections_count}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display visualization
                cv2.imshow("Exploration View", vis_frame)
                
                # Get semantic map visualization
                map_vis = self.mapper.visualize_map(agent_position=position)
                cv2.imshow("Semantic Map", map_vis)
                
                # Update display
                cv2.waitKey(1)
            
            steps_taken += 1
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        
        # Get unique object classes found
        unique_objects = list(self.mapper.semantic_map.keys())
        
        results = {
            'duration': elapsed_time,
            'steps_taken': steps_taken,
            'total_detections': detections_count,
            'unique_objects': unique_objects,
            'num_unique_objects': len(unique_objects),
            'cells_explored': len(explored_cells)
        }
        
        print(f"Exploration complete:")
        print(f"  Duration: {elapsed_time:.1f}s, Steps: {steps_taken}")
        print(f"  Total detections: {detections_count}")
        print(f"  Unique objects found: {', '.join(unique_objects)}")
        print(f"  Area explored: {len(explored_cells)} cells")
        
        return results


if __name__ == "__main__":
    print("This module provides navigation capabilities for the Habitat-YOLO integration.")
    print("It should be imported by main.py rather than run directly.")
