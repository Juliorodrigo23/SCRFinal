"""
Semantic mapper for combining Habitat-Sim with YOLO detections.
This module builds and maintains a semantic map of detected objects.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import time


class SemanticMapper:
    def __init__(self, map_resolution=0.05, map_size=20.0):
        """
        Initialize a semantic map for tracking object detections
        
        Args:
            map_resolution (float): Resolution of the map in meters per pixel
            map_size (float): Size of the map in meters
        """
        self.map_resolution = map_resolution
        self.map_size = map_size
        
        # Calculate map dimensions in pixels
        self.map_dims = int(map_size / map_resolution)
        
        # Initialize map
        self.semantic_map = {}
        self.object_positions = {}  # Store latest positions of objects
        self.confidence_map = {}    # Store confidence of detections
        self.timestamp_map = {}     # When objects were last seen
        
        # Keep track of robot pose history
        self.pose_history = []
        
        print(f"Semantic mapper initialized with {self.map_dims}x{self.map_dims} pixels")
        print(f"Map resolution: {map_resolution}m/pixel, size: {map_size}x{map_size}m")
        
    def _world_to_map(self, world_pos):
        """Convert world coordinates to map coordinates"""
        # Shift origin to center of map
        map_pos_x = int((world_pos[0] + self.map_size / 2) / self.map_resolution)
        map_pos_z = int((world_pos[2] + self.map_size / 2) / self.map_resolution)
        
        # Ensure within map bounds
        map_pos_x = np.clip(map_pos_x, 0, self.map_dims - 1)
        map_pos_z = np.clip(map_pos_z, 0, self.map_dims - 1)
        
        return map_pos_x, map_pos_z
    
    def _map_to_world(self, map_pos):
        """Convert map coordinates to world coordinates"""
        world_x = map_pos[0] * self.map_resolution - self.map_size / 2
        world_z = map_pos[1] * self.map_resolution - self.map_size / 2
        
        return np.array([world_x, 0.0, world_z])  # Assume y=0 (ground level)
    
    def update_pose(self, agent_position, agent_rotation):
        """
        Update the agent's pose history
        
        Args:
            agent_position (numpy.ndarray): [x, y, z] position
            agent_rotation (quaternion): Agent rotation as quaternion
        """
        self.pose_history.append({
            'position': agent_position.copy(),
            'rotation': agent_rotation.copy(),
            'timestamp': time.time()
        })
        
        # Keep history to a reasonable size
        if len(self.pose_history) > 1000:
            self.pose_history = self.pose_history[-1000:]
            
    def _calculate_object_world_position(self, detection, depth_frame, camera_params, agent_position, agent_rotation):
        """
        Calculate the 3D world position of a detected object
        
        Args:
            detection (dict): Object detection from YOLODetector
            depth_frame (numpy.ndarray): Depth frame from Habitat
            camera_params (dict): Camera parameters (FOV, etc)
            agent_position (numpy.ndarray): Agent position
            agent_rotation (quaternion): Agent rotation
            
        Returns:
            numpy.ndarray: [x, y, z] world position of object
            float: Depth uncertainty estimate
        """
        import habitat_sim
        from habitat_sim.utils.common import quat_rotate_vector
        
        # Get bounding box
        x1, y1, x2, y2 = detection['bbox']
        
        # Get center point of the bottom of the bounding box
        # (better approximation of object's position on the ground)
        center_x = (x1 + x2) // 2
        bottom_y = y2
        
        # Get depth at this point (with some averaging for stability)
        height, width = depth_frame.shape
        
        # Ensure points are within frame bounds
        center_x = np.clip(center_x, 0, width - 1)
        bottom_y = np.clip(bottom_y, 0, height - 1)
        
        # Sample multiple points around bottom center for robustness
        sample_radius = 3
        sample_points = []
        for dy in range(-sample_radius, sample_radius + 1):
            for dx in range(-sample_radius, sample_radius + 1):
                px = np.clip(center_x + dx, 0, width - 1)
                py = np.clip(bottom_y + dy, 0, height - 1)
                if 0 <= py < height and 0 <= px < width:
                    sample_points.append((px, py))
        
        # Get depth values at sample points
        depth_values = [depth_frame[py, px] for px, py in sample_points]
        
        # Filter out invalid depth values (e.g., 0 or very large values)
        valid_depths = [d for d in depth_values if 0.01 < d < 10.0]
        
        if not valid_depths:
            # If no valid depths, try the center point
            if 0.01 < depth_frame[bottom_y, center_x] < 10.0:
                depth = depth_frame[bottom_y, center_x]
            else:
                # Fall back to middle of bounding box
                mid_y = (y1 + y2) // 2
                mid_y = np.clip(mid_y, 0, height - 1)
                if 0.01 < depth_frame[mid_y, center_x] < 10.0:
                    depth = depth_frame[mid_y, center_x]
                else:
                    # No valid depth found
                    return None, float('inf')
            
            uncertainty = 1.0  # High uncertainty when using fallback
        else:
            # Use median depth for robustness against outliers
            depth = np.median(valid_depths)
            
            # Calculate uncertainty based on depth variance
            if len(valid_depths) > 1:
                uncertainty = np.std(valid_depths) / depth
            else:
                uncertainty = 0.2  # Default uncertainty
        
        # Calculate FOV in radians
        hfov = camera_params.get('hfov', 90) * np.pi / 180
        
        # Calculate ray direction in camera frame
        cx = width / 2
        cy = height / 2
        fx = cx / np.tan(hfov / 2)  # Focal length in pixels
        
        # Convert pixel to normalized device coordinates
        x_ndc = (center_x - cx) / fx
        y_ndc = (bottom_y - cy) / fx
        
        # Create ray direction vector
        ray_dir = np.array([x_ndc, y_ndc, 1.0])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        # Rotate ray to world frame
        ray_dir_world = quat_rotate_vector(agent_rotation, ray_dir)
        
        # Calculate world position
        world_pos = agent_position + ray_dir_world * depth
        
        # Set y coordinate to ground level (assuming objects are on the floor)
        # This is an approximation - for more accurate placement, could use floor detection
        world_pos[1] = 0.0
        
        return world_pos, uncertainty
    
    def update_from_detection(self, detections, depth_frame, camera_params, agent_position, agent_rotation):
        """
        Update semantic map with new detections
        
        Args:
            detections (list): List of detections from YOLODetector
            depth_frame (numpy.ndarray): Depth frame from Habitat
            camera_params (dict): Camera parameters (FOV, etc)
            agent_position (numpy.ndarray): Agent position
            agent_rotation (quaternion): Agent rotation
            
        Returns:
            dict: Updated object positions
        """
        current_time = time.time()
        
        # Process each detection
        for detection in detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Calculate world position
            world_pos, uncertainty = self._calculate_object_world_position(
                detection, depth_frame, camera_params,
                agent_position, agent_rotation
            )
            
            # Skip if position couldn't be determined
            if world_pos is None:
                continue
                
            # Convert to map coordinates
            map_pos = self._world_to_map(world_pos)
            
            # Initialize maps for this class if not exists
            if class_name not in self.semantic_map:
                self.semantic_map[class_name] = np.zeros((self.map_dims, self.map_dims), dtype=np.float32)
                self.confidence_map[class_name] = np.zeros((self.map_dims, self.map_dims), dtype=np.float32)
                self.timestamp_map[class_name] = np.zeros((self.map_dims, self.map_dims), dtype=np.float32)
            
            # Update maps at this position
            x, z = map_pos
            
            # Check if this is a more confident detection than previous ones
            prev_conf = self.confidence_map[class_name][x, z]
            
            # Update if:
            # 1. No previous detection at this location, or
            # 2. This detection is more confident, or
            # 3. Previous detection is old (>60s)
            if (prev_conf == 0 or 
                confidence > prev_conf or 
                current_time - self.timestamp_map[class_name][x, z] > 60):
                
                # Update semantic map (use confidence-weighted averaging)
                if prev_conf > 0:
                    # Blend with previous detection
                    alpha = confidence / (prev_conf + confidence)
                    self.semantic_map[class_name][x, z] = alpha * 1.0 + (1 - alpha) * self.semantic_map[class_name][x, z]
                else:
                    # New detection
                    self.semantic_map[class_name][x, z] = 1.0
                
                # Update confidence and timestamp
                self.confidence_map[class_name][x, z] = confidence
                self.timestamp_map[class_name][x, z] = current_time
                
                # Update latest position
                self.object_positions[class_name] = {
                    'world_position': world_pos,
                    'map_position': map_pos,
                    'confidence': confidence,
                    'timestamp': current_time,
                    'uncertainty': uncertainty
                }
                
        return self.object_positions
    
    def get_object_position(self, class_name, min_confidence=0.0, max_age=float('inf')):
        """
        Get the latest known position of an object
        
        Args:
            class_name (str): Class name to look for
            min_confidence (float): Minimum confidence level
            max_age (float): Maximum age in seconds
            
        Returns:
            dict or None: Object position data or None if not found
        """
        if class_name not in self.object_positions:
            return None
        
        pos_data = self.object_positions[class_name]
        current_time = time.time()
        
        # Check if data meets criteria
        if (pos_data['confidence'] >= min_confidence and
            current_time - pos_data['timestamp'] <= max_age):
            return pos_data
        
        return None
    
    def get_nearest_object(self, class_name, agent_position, min_confidence=0.0, max_age=float('inf'), max_distance=float('inf')):
        """
        Find the nearest instance of an object to the agent
        
        Args:
            class_name (str): Class name to look for
            agent_position (numpy.ndarray): Agent position
            min_confidence (float): Minimum confidence level
            max_age (float): Maximum age in seconds
            max_distance (float): Maximum distance to search
            
        Returns:
            dict or None: Object position data or None if not found
        """
        if class_name not in self.semantic_map:
            return None
        
        # Get current timestamp
        current_time = time.time()
        
        # Find all instances of this class
        instances = []
        confidence_map = self.confidence_map[class_name]
        timestamp_map = self.timestamp_map[class_name]
        
        # Search through the map (this could be optimized further)
        for x in range(self.map_dims):
            for z in range(self.map_dims):
                confidence = confidence_map[x, z]
                timestamp = timestamp_map[x, z]
                
                # Skip if below confidence threshold or too old
                if confidence < min_confidence or current_time - timestamp > max_age:
                    continue
                
                # Convert map coordinates to world
                world_pos = self._map_to_world((x, z))
                
                # Calculate distance to agent
                distance = np.linalg.norm(world_pos - agent_position)
                
                # Skip if too far
                if distance > max_distance:
                    continue
                
                instances.append({
                    'world_position': world_pos,
                    'map_position': (x, z),
                    'confidence': confidence,
                    'timestamp': timestamp,
                    'distance': distance
                })
        
        # Sort by distance
        instances.sort(key=lambda x: x['distance'])
        
        return instances[0] if instances else None
        
    def visualize_map(self, agent_position=None, class_filter=None, include_trajectory=True, min_confidence=0.2):
        """
        Visualize the semantic map
        
        Args:
            agent_position (numpy.ndarray, optional): Agent position to show on map
            class_filter (list, optional): List of classes to show (None for all)
            include_trajectory (bool): Whether to show agent trajectory
            min_confidence (float): Minimum confidence to show
            
        Returns:
            numpy.ndarray: Visualization image
        """
        # Create RGB map
        vis_map = np.zeros((self.map_dims, self.map_dims, 3), dtype=np.uint8)
        
        # Add all object classes with different colors
        for i, class_name in enumerate(self.semantic_map.keys()):
            if class_filter and class_name not in class_filter:
                continue
                
            # Skip if no detections above threshold
            if np.max(self.confidence_map[class_name]) < min_confidence:
                continue
                
            # Generate color for this class (using HSV for better control)
            hue = i / max(1, len(self.semantic_map) - 1)  # Distribute hues
            hsv_color = np.array([hue, 0.8, 0.8])  # Saturation and value fixed
            rgb_color = hsv_to_rgb(hsv_color.reshape(1, 1, 3)).flatten() * 255
            
            # Get binary mask of detections
            mask = self.confidence_map[class_name] >= min_confidence
            
            # Apply color to vis_map where mask is True
            vis_map[mask] = rgb_color
        
        # Draw trajectory if requested
        if include_trajectory and self.pose_history:
            for pose in self.pose_history:
                pos = pose['position']
                map_x, map_z = self._world_to_map(pos)
                
                # Ensure within bounds
                if 0 <= map_x < self.map_dims and 0 <= map_z < self.map_dims:
                    cv2.circle(vis_map, (map_x, map_z), 1, (100, 100, 100), -1)
        
        # Draw agent position if provided
        if agent_position is not None:
            map_x, map_z = self._world_to_map(agent_position)
            
            # Ensure within bounds
            if 0 <= map_x < self.map_dims and 0 <= map_z < self.map_dims:
                # Draw agent as a circle with triangle for orientation
                cv2.circle(vis_map, (map_x, map_z), 5, (0, 255, 0), -1)
                
                # If we have agent rotation, draw orientation indicator
                if len(self.pose_history) > 0:
                    import habitat_sim
                    from habitat_sim.utils.common import quat_rotate_vector
                    
                    # Get latest rotation
                    rot = self.pose_history[-1]['rotation']
                    
                    # Calculate forward vector
                    forward = np.array([0, 0, -1])  # Default forward in Habitat
                    forward = quat_rotate_vector(rot, forward)
                    forward = forward / np.linalg.norm(forward) * 10  # Scale length
                    
                    # Calculate endpoint
                    end_x = int(map_x + forward[0])
                    end_z = int(map_z + forward[2])
                    
                    # Draw orientation line
                    cv2.line(vis_map, (map_x, map_z), (end_x, end_z), (0, 255, 255), 2)
        
        # Add a legend to the visualization
        if class_filter is None:
            class_list = list(self.semantic_map.keys())
        else:
            class_list = [c for c in class_filter if c in self.semantic_map]
            
        # Add borders to the map
        bordered_map = cv2.copyMakeBorder(vis_map, 30, 0, 0, 100, 
                                          cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        # Add legend
        for i, class_name in enumerate(class_list):
            if class_name not in self.semantic_map:
                continue
                
            # Skip if no detections above threshold
            if np.max(self.confidence_map[class_name]) < min_confidence:
                continue
                
            # Generate color as before
            hue = i / max(1, len(self.semantic_map) - 1)
            hsv_color = np.array([hue, 0.8, 0.8])
            rgb_color = hsv_to_rgb(hsv_color.reshape(1, 1, 3)).flatten() * 255
            
            # Add color box and text
            legend_y = 10 + i * 20
            cv2.rectangle(bordered_map, (self.map_dims + 10, legend_y), 
                          (self.map_dims + 25, legend_y + 15), rgb_color.astype(int), -1)
            cv2.putText(bordered_map, class_name, (self.map_dims + 30, legend_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return bordered_map
    
    def save_map(self, filename):
        """
        Save semantic map visualization to a file
        
        Args:
            filename (str): Output filename
            
        Returns:
            bool: Success
        """
        try:
            # Create visualization
            vis_map = self.visualize_map(include_trajectory=True)
            
            # Save to file
            cv2.imwrite(filename, vis_map)
            print(f"Saved semantic map to {filename}")
            return True
        except Exception as e:
            print(f"Error saving map: {e}")
            return False
            
    def find_path_to_object(self, class_name, agent_position, min_confidence=0.5, max_age=300):
        """
        Find a simple path to the nearest object of a given class
        
        Args:
            class_name (str): Object class to find
            agent_position (numpy.ndarray): Current agent position
            min_confidence (float): Minimum detection confidence
            max_age (float): Maximum age of detection in seconds
            
        Returns:
            list or None: List of waypoints to object or None if not found
        """
        # Find nearest object
        obj = self.get_nearest_object(
            class_name, agent_position, 
            min_confidence=min_confidence, 
            max_age=max_age
        )
        
        if obj is None:
            return None
            
        # For now, return a direct path (just the endpoint)
        # In a more sophisticated implementation, this could use A* pathfinding
        return [obj['world_position']]


if __name__ == "__main__":
    # Simple test to check SemanticMapper functionality
    mapper = SemanticMapper(map_resolution=0.1, map_size=10.0)
    
    # Simulate some detections
    test_data = [
        {'class_name': 'chair', 'position': np.array([1.0, 0.0, 2.0]), 'confidence': 0.8},
        {'class_name': 'table', 'position': np.array([-2.0, 0.0, -1.5]), 'confidence': 0.9},
        {'class_name': 'bed', 'position': np.array([3.0, 0.0, -3.0]), 'confidence': 0.7},
        {'class_name': 'chair', 'position': np.array([1.2, 0.0, 2.1]), 'confidence': 0.6}  # Near the first chair
    ]
    
    # Manually add to map for testing
    for item in test_data:
        pos = item['position']
        x, z = mapper._world_to_map(pos)
        
        class_name = item['class_name']
        confidence = item['confidence']
        
        # Initialize maps for this class if not exists
        if class_name not in mapper.semantic_map:
            mapper.semantic_map[class_name] = np.zeros((mapper.map_dims, mapper.map_dims), dtype=np.float32)
            mapper.confidence_map[class_name] = np.zeros((mapper.map_dims, mapper.map_dims), dtype=np.float32)
            mapper.timestamp_map[class_name] = np.zeros((mapper.map_dims, mapper.map_dims), dtype=np.float32)
        
        # Update maps
        mapper.semantic_map[class_name][x, z] = 1.0
        mapper.confidence_map[class_name][x, z] = confidence
        mapper.timestamp_map[class_name][x, z] = time.time()
        
        # Update latest position
        mapper.object_positions[class_name] = {
            'world_position': pos,
            'map_position': (x, z),
            'confidence': confidence,
            'timestamp': time.time(),
            'uncertainty': 0.1
        }
    
    # Simulate agent trajectory
    for i in range(20):
        pos = np.array([i * 0.2 - 2, 0.0, i * 0.1 - 1])
        rot = np.array([0, 0, 0, 1])  # Identity quaternion
        mapper.update_pose(pos, rot)
    
    # Visualize map
    vis_map = mapper.visualize_map(agent_position=np.array([0.0, 0.0, 0.0]))
    
    # Show visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(vis_map, cv2.COLOR_BGR2RGB))
    plt.title("Semantic Map Test")
    plt.show()
    
    # Test find path
    agent_pos = np.array([0.0, 0.0, 0.0])
    path = mapper.find_path_to_object('chair', agent_pos)
    if path:
        print(f"Path to nearest chair: {path}")
    else:
        print("No path found")
