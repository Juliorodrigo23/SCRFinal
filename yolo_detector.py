"""
YOLO object detection module for Habitat-Sim.
Handles loading the YOLO model and detecting objects in RGB frames.
"""

import cv2
import numpy as np
import torch
import time


class YOLODetector:
    def __init__(
        self, 
        weights_path, 
        config_path=None, 
        device="cuda" if torch.cuda.is_available() else "cpu",
        conf_threshold=0.5,
        nms_threshold=0.4,
        target_classes=None
    ):
        """
        Initialize YOLO detector
        
        Args:
            weights_path (str): Path to YOLO weights file (.pt for PyTorch or .weights for Darknet)
            config_path (str, optional): Path to YOLO config file for Darknet models
            device (str): Device to run inference on ('cuda' or 'cpu')
            conf_threshold (float): Confidence threshold for detections
            nms_threshold (float): Non-maximum suppression threshold
            target_classes (list, optional): List of class names to detect (None for all classes)
        """
        self.weights_path = weights_path
        self.config_path = config_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.target_classes = target_classes
        self.model = None
        self.classes = []
        
        # Track model type (PyTorch vs Darknet)
        self.model_type = "pytorch" if weights_path.endswith(".pt") else "darknet"
        
        # Load model
        self._load_model()
        
        print(f"YOLO detector initialized on {self.device}")
        if self.target_classes:
            print(f"Filtering for classes: {self.target_classes}")
            
    def _load_model(self):
        """Load the YOLO model based on model type"""
        if self.model_type == "pytorch":
            # Load PyTorch YOLO model (YOLOv5/v7)
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                           path=self.weights_path, device=self.device)
                
                # Set confidence threshold
                self.model.conf = self.conf_threshold
                self.model.iou = self.nms_threshold
                
                # Get class names
                self.classes = self.model.names
                print(f"Loaded PyTorch YOLO model with {len(self.classes)} classes")
                
            except Exception as e:
                print(f"Error loading PyTorch model: {e}")
                print("Falling back to OpenCV DNN model")
                self.model_type = "darknet"
                
        if self.model_type == "darknet":
            # Ensure config path is provided for Darknet models
            if self.config_path is None:
                raise ValueError("Config path must be provided for Darknet YOLO models")
                
            # Load Darknet YOLO model via OpenCV DNN
            self.model = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
            
            # Set device
            if self.device == "cuda":
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Load class names
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            print(f"Loaded Darknet YOLO model with {len(self.classes)} classes")
    
    def _preprocess_frame(self, frame):
        """
        Preprocess frame for YOLO detection
        
        Args:
            frame (numpy.ndarray): RGB image (HxWxC)
            
        Returns:
            Processed image ready for YOLO inference
        """
        if self.model_type == "pytorch":
            # PyTorch models handle preprocessing internally
            return frame
        else:
            # For Darknet models, we need to preprocess manually
            blob = cv2.dnn.blobFromImage(
                frame, 1/255.0, (416, 416), swapRB=True, crop=False
            )
            return blob
    
    def detect(self, frame):
        """
        Detect objects in RGB frame
        
        Args:
            frame (numpy.ndarray): RGB image from Habitat
            
        Returns:
            list: List of detections [
                {
                    'class_id': int,
                    'class_name': str,
                    'confidence': float,
                    'bbox': [x1, y1, x2, y2]  # Format: [top-left, bottom-right]
                }, ...
            ]
        """
        start_time = time.time()
        detections = []
        
        if self.model_type == "pytorch":
            # PyTorch model inference
            processed_frame = self._preprocess_frame(frame)
            results = self.model(processed_frame)
            
            # Extract detections
            predictions = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
            
            for *box, conf, cls_id in predictions:
                cls_id = int(cls_id)
                class_name = self.classes[cls_id]
                
                # Skip if not in target classes
                if self.target_classes and class_name not in self.target_classes:
                    continue
                
                detections.append({
                    'class_id': cls_id,
                    'class_name': class_name,
                    'confidence': float(conf),
                    'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                })
                
        else:
            # Darknet model inference using OpenCV DNN
            height, width = frame.shape[:2]
            blob = self._preprocess_frame(frame)
            
            # Set input to network
            self.model.setInput(blob)
            
            # Get output layer names
            layer_names = self.model.getLayerNames()
            output_layers = [layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]
            
            # Forward pass
            outputs = self.model.forward(output_layers)
            
            # Process outputs
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > self.conf_threshold:
                        class_name = self.classes[class_id]
                        
                        # Skip if not in target classes
                        if self.target_classes and class_name not in self.target_classes:
                            continue
                        
                        # Convert to pixel coordinates
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Calculate bounding box
                        x1 = max(0, int(center_x - w / 2))
                        y1 = max(0, int(center_y - h / 2))
                        x2 = min(width, int(center_x + w / 2))
                        y2 = min(height, int(center_y + h / 2))
                        
                        detections.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': float(confidence),
                            'bbox': [x1, y1, x2, y2]
                        })
            
            # Apply Non-Maximum Suppression
            indices = cv2.dnn.NMSBoxes(
                [d['bbox'] for d in detections],
                [d['confidence'] for d in detections],
                self.conf_threshold,
                self.nms_threshold
            )
            
            if len(indices) > 0:
                # Keep only the detections that survived NMS
                if isinstance(indices, tuple):  # OpenCV 4.5.4+
                    indices = indices[0]
                detections = [detections[i] for i in indices]
        
        elapsed = time.time() - start_time
        fps = 1 / elapsed
        
        return {
            'detections': detections,
            'fps': fps,
            'elapsed': elapsed
        }
    
    def visualize_detections(self, frame, detection_results):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame (numpy.ndarray): Original RGB frame
            detection_results (dict): Results from detect() method
            
        Returns:
            numpy.ndarray: Frame with visualized detections
        """
        # Make a copy to avoid modifying the original
        vis_frame = frame.copy()
        
        # Convert RGB to BGR for OpenCV
        if vis_frame.shape[2] == 3:
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            
        # Draw each detection
        for det in detection_results['detections']:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Generate color based on class_id
            color = (int(hash(class_name) % 255), 
                    int(hash(class_name + "1") % 255), 
                    int(hash(class_name + "2") % 255))
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            text = f"{class_name}: {confidence:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_frame, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_frame, text, (x1, y1 - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        # Draw FPS
        fps_text = f"FPS: {detection_results['fps']:.1f}"
        cv2.putText(vis_frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return vis_frame
            

if __name__ == "__main__":
    # Simple test to check YOLODetector functionality
    try:
        # Test with PyTorch model (YOLOv5s)
        detector = YOLODetector(
            weights_path="yolov5s.pt",  # Will auto-download from PyTorch Hub
            target_classes=["chair", "bottle", "bed", "cup", "keyboard"]
        )
        
        # Load test image
        test_img = cv2.imread("test_image.jpg")
        if test_img is None:
            # Create dummy image if no test image is available
            test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
        # Convert BGR to RGB
        test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            
        # Run detection
        results = detector.detect(test_img_rgb)
        
        print(f"Detection complete. Found {len(results['detections'])} objects.")
        for det in results['detections']:
            print(f"  {det['class_name']}: {det['confidence']:.2f}")
            
        # Visualize
        vis_img = detector.visualize_detections(test_img_rgb, results)
        
        # Display
        cv2.imshow("YOLO Test", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error testing YOLO detector: {e}")
        print("You may need to install PyTorch and YOLOv5 dependencies.")