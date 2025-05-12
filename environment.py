"""
Environment setup for Habitat-Sim with YOLO integration.
This module handles environment initialization, sensor setup, and frame extraction.
"""

import os
import cv2
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb


class HabitatEnvironment:
    def __init__(self, scene_path, enable_physics=True):
        """
        Initialize the Habitat environment with RGB and depth sensors
        
        Args:
            scene_path (str): Path to the scene file (.glb or .ply)
            enable_physics (bool): Whether to enable physics simulation
        """
        self.scene_path = scene_path
        self.sim = None
        self.agent = None
        self.enable_physics = enable_physics
        
        # Initialize the simulator
        self._initialize_simulator()
        
    def _initialize_simulator(self):
        """Set up the Habitat simulator and agent with sensors"""
        # Simulator configuration
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.scene_path
        backend_cfg.enable_physics = self.enable_physics
        
        # Agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        
        # RGB sensor
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "rgb"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [640, 480]
        rgb_sensor_spec.position = [0.0, 1.5, 0.0]  # Camera position relative to agent
        rgb_sensor_spec.orientation = [0.0, 0.0, 0.0]  # Camera orientation
        rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        
        # Depth sensor
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [640, 480]
        depth_sensor_spec.position = [0.0, 1.5, 0.0]
        depth_sensor_spec.orientation = [0.0, 0.0, 0.0]
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        
        # Add sensors to agent
        agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]
        
        # Configure agent movement
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
        }
        
        # Create configuration
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        
        # Initialize simulator
        self.sim = habitat_sim.Simulator(cfg)
        self.agent = self.sim.initialize_agent(0)
        
        print(f"Environment initialized with scene: {self.scene_path}")
        print(f"Agent created with sensors: {self.agent.sensor_specifications}")
    
    def get_observation(self):
        """Get current RGB and depth observations"""
        observations = self.sim.get_sensor_observations()
        rgb = observations["rgb"]
        depth = observations["depth"]
        
        # Convert depth to more usable format
        depth_in_meters = np.clip(depth, 0, 10)  # Clip to 10m max distance
        
        return {
            "rgb": rgb,
            "depth": depth_in_meters
        }
    
    def get_rgb_frame(self):
        """Get only the RGB frame (for YOLO processing)"""
        return self.get_observation()["rgb"]
    
    def perform_action(self, action):
        """Perform an action in the environment
        
        Args:
            action (str): One of "move_forward", "turn_left", "turn_right"
            
        Returns:
            bool: Success of the action
        """
        if action not in self.agent.action_space:
            print(f"Invalid action: {action}")
            return False
        
        observations = self.sim.step(action)
        return True
    
    def navigate_to(self, target_position, tolerance=0.5):
        """
        Simple navigation toward a target position
        
        Args:
            target_position (numpy.array): [x, y, z] position to navigate to
            tolerance (float): How close to get to the target
            
        Returns:
            bool: Whether target was reached
        """
        MAX_STEPS = 100  # Prevent infinite loops
        steps_taken = 0
        
        while steps_taken < MAX_STEPS:
            agent_state = self.agent.get_state()
            current_position = agent_state.position
            
            # Calculate distance to target
            distance = np.linalg.norm(current_position - target_position)
            
            # If close enough to target, we're done
            if distance < tolerance:
                print(f"Target reached within tolerance: {distance:.2f}m")
                return True
            
            # Calculate direction to target
            direction = target_position - current_position
            direction[1] = 0  # Ignore height difference
            direction = direction / np.linalg.norm(direction)
            
            # Calculate angle between agent's forward direction and target
            forward = np.array([0, 0, -1])  # Default forward direction in Habitat
            forward = habitat_sim.utils.common.quat_rotate_vector(
                agent_state.rotation, forward
            )
            
            # Calculate angle (dot product)
            dot_product = np.dot(direction, forward)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
            
            # Decide action based on angle
            if angle < 0.1:  # Nearly facing target
                self.perform_action("move_forward")
            elif np.cross(forward, direction)[1] > 0:  # Target is to the left
                self.perform_action("turn_left")
            else:  # Target is to the right
                self.perform_action("turn_right")
            
            steps_taken += 1
            
        print(f"Failed to reach target after {MAX_STEPS} steps")
        return False
    
    def close(self):
        """Clean up the simulator"""
        if self.sim is not None:
            self.sim.close()
            
    def display_observation(self, window_name="Habitat Observation"):
        """
        Display the current RGB observation (for debugging)
        """
        rgb = self.get_rgb_frame()
        
        # Convert RGB to BGR (OpenCV format)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Display
        cv2.imshow(window_name, bgr)
        cv2.waitKey(1)  # Update display


def get_available_scenes():
    """
    List available test scenes from Habitat
    Returns a list of scene paths
    """
    test_scenes = {
        "apartment_1": "data/scene_datasets/habitat-test-scenes/apartment_1.glb",
        "skokloster": "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
        "van_gogh_room": "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
    }
    
    # Check which scenes actually exist
    available_scenes = {}
    for name, path in test_scenes.items():
        if os.path.exists(path):
            available_scenes[name] = path
    
    if not available_scenes:
        print("No test scenes found. You may need to download them.")
        print("See: https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md")
    
    return available_scenes


if __name__ == "__main__":
    # Simple test to check environment setup
    available_scenes = get_available_scenes()
    
    if available_scenes:
        scene_name = list(available_scenes.keys())[0]
        scene_path = available_scenes[scene_name]
        print(f"Testing with scene: {scene_name} at {scene_path}")
        
        env = HabitatEnvironment(scene_path)
        
        # Take some actions
        for _ in range(10):
            env.perform_action("move_forward")
            obs = env.get_observation()
            print(f"RGB shape: {obs['rgb'].shape}, Depth shape: {obs['depth'].shape}")
            
            # Display (uncomment for visualization)
            # env.display_observation()
            
        env.close()
    else:
        print("Please download Habitat test scenes to run this example.")