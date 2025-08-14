import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
from datetime import datetime
import json
import argparse
from pathlib import Path
import logging

class AdvancedHelmetDetector:
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """
        Advanced Helmet Detection System
        
        Args:
            model_path: Path to custom trained model (optional)
            confidence_threshold: Minimum confidence for detection
        """
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            self.logger.info(f"Loaded custom model from {model_path}")
        else:
            # Use pre-trained YOLOv8 model and fine-tune for helmet detection
            self.model = YOLO('yolov8n.pt')  # You can use yolov8s.pt, yolov8m.pt for better accuracy
            self.logger.info("Loaded YOLOv8 base model")
        
        # Class names for helmet detection
        self.class_names = {
            0: 'person',
            1: 'helmet',
            2: 'no_helmet'
        }
        
        # Colors for bounding boxes
        self.colors = {
            'helmet': (0, 255, 0),      # Green
            'no_helmet': (0, 0, 255),   # Red
            'person': (255, 0, 0)       # Blue
        }
        
        # Statistics
        self.detection_stats = {
            'total_detections': 0,
            'helmet_count': 0,
            'no_helmet_count': 0,
            'violations': []
        }

    def detect_in_image(self, image_path, save_result=True):
        """
        Detect helmets in a single image
        
        Args:
            image_path: Path to input image
            save_result: Whether to save annotated result
        
        Returns:
            dict: Detection results
        """
        if not os.path.exists(image_path):
            self.logger.error(f"Image not found: {image_path}")
            return None
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Could not load image: {image_path}")
            return None
        
        # Run detection
        results = self.model(image, conf=self.confidence_threshold)
        
        # Process results
        detections = self._process_detections(results[0], image)
        
        if save_result:
            output_path = f"results/annotated_{os.path.basename(image_path)}"
            os.makedirs("results", exist_ok=True)
            cv2.imwrite(output_path, detections['annotated_image'])
            self.logger.info(f"Saved result to {output_path}")
        
        return detections

    def detect_live_video(self, source=0, save_video=False):
        """
        Real-time helmet detection from webcam or video file
        
        Args:
            source: Video source (0 for webcam, or path to video file)
            save_video: Whether to save output video
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            self.logger.error(f"Could not open video source: {source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if saving
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/helmet_detection_{timestamp}.mp4"
            os.makedirs("results", exist_ok=True)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.logger.info("Starting live detection. Press 'q' to quit, 's' to save screenshot")
        
        frame_count = 0
        violation_alert_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection every few frames for performance
            if frame_count % 3 == 0:  # Process every 3rd frame
                results = self.model(frame, conf=self.confidence_threshold)
                detections = self._process_detections(results[0], frame)
                frame = detections['annotated_image']
                
                # Check for violations
                if detections['no_helmet_count'] > 0:
                    violation_alert_frames += 1
                    if violation_alert_frames > 10:  # Alert after consistent violations
                        self._trigger_violation_alert(detections)
                        violation_alert_frames = 0
                else:
                    violation_alert_frames = max(0, violation_alert_frames - 1)
            
            # Add real-time info overlay
            self._add_info_overlay(frame, detections if 'detections' in locals() else None)
            
            # Display frame
            cv2.imshow('Advanced Helmet Detection', frame)
            
            # Save video frame
            if save_video:
                out.write(frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"results/screenshot_{timestamp}.jpg"
                cv2.imwrite(screenshot_path, frame)
                self.logger.info(f"Screenshot saved: {screenshot_path}")
        
        # Cleanup
        cap.release()
        if save_video:
            out.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        self._print_statistics()

    def _process_detections(self, results, image):
        """Process YOLO detection results"""
        detections = {
            'helmet_count': 0,
            'no_helmet_count': 0,
            'person_count': 0,
            'boxes': [],
            'annotated_image': image.copy()
        }
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = map(int, box)
                class_id = int(cls)
                
                # For helmet detection, we need to identify persons and check for helmets
                if class_id == 0:  # Person detected
                    detections['person_count'] += 1
                    
                    # Check if person has helmet (this would need custom training)
                    # For demo, we'll simulate helmet detection based on head region
                    has_helmet = self._check_helmet_on_person(image, (x1, y1, x2, y2))
                    
                    if has_helmet:
                        detections['helmet_count'] += 1
                        label = f"Helmet: {conf:.2f}"
                        color = self.colors['helmet']
                    else:
                        detections['no_helmet_count'] += 1
                        label = f"No Helmet: {conf:.2f}"
                        color = self.colors['no_helmet']
                        
                        # Log violation
                        self.detection_stats['violations'].append({
                            'timestamp': datetime.now().isoformat(),
                            'confidence': float(conf),
                            'bbox': [x1, y1, x2, y2]
                        })
                    
                    # Draw bounding box and label
                    cv2.rectangle(detections['annotated_image'], (x1, y1), (x2, y2), color, 2)
                    cv2.putText(detections['annotated_image'], label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    detections['boxes'].append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class': 'helmet' if has_helmet else 'no_helmet'
                    })
        
        # Update global statistics
        self.detection_stats['total_detections'] += detections['person_count']
        self.detection_stats['helmet_count'] += detections['helmet_count']
        self.detection_stats['no_helmet_count'] += detections['no_helmet_count']
        
        return detections

    def _check_helmet_on_person(self, image, bbox):
        """
        Advanced helmet detection on person's head region
        This is a simplified version - in production, you'd use a specialized helmet detection model
        """
        x1, y1, x2, y2 = bbox
        
        # Extract head region (top 30% of person bbox)
        head_height = int((y2 - y1) * 0.3)
        head_region = image[y1:y1+head_height, x1:x2]
        
        if head_region.size == 0:
            return False
        
        # Simple color-based helmet detection (this is basic - replace with ML model)
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        
        # Define helmet color ranges (you can expand this)
        helmet_colors = [
            # White helmet
            (np.array([0, 0, 200]), np.array([180, 30, 255])),
            # Yellow helmet
            (np.array([20, 100, 100]), np.array([30, 255, 255])),
            # Red helmet
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
        ]
        
        total_helmet_pixels = 0
        total_pixels = head_region.shape[0] * head_region.shape[1]
        
        for lower, upper in helmet_colors:
            mask = cv2.inRange(hsv, lower, upper)
            helmet_pixels = cv2.countNonZero(mask)
            total_helmet_pixels += helmet_pixels
        
        # If more than 15% of head region matches helmet colors
        helmet_ratio = total_helmet_pixels / total_pixels
        return helmet_ratio > 0.15

    def _add_info_overlay(self, frame, detections):
        """Add information overlay to frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text information
        if detections:
            info_text = [
                f"Persons Detected: {detections['person_count']}",
                f"With Helmet: {detections['helmet_count']}",
                f"Without Helmet: {detections['no_helmet_count']}",
                f"Compliance: {(detections['helmet_count']/(detections['person_count'] or 1)*100):.1f}%"
            ]
        else:
            info_text = ["Initializing detection..."]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (20, 30 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _trigger_violation_alert(self, detections):
        """Trigger alert for helmet violations"""
        self.logger.warning(f"HELMET VIOLATION DETECTED! {detections['no_helmet_count']} person(s) without helmet")
        
        # You can add more alert mechanisms here:
        # - Send email/SMS alerts
        # - Save violation images
        # - Log to database
        # - Sound alarm

    def _print_statistics(self):
        """Print detection statistics"""
        print("\n" + "="*50)
        print("HELMET DETECTION STATISTICS")
        print("="*50)
        print(f"Total Detections: {self.detection_stats['total_detections']}")
        print(f"With Helmet: {self.detection_stats['helmet_count']}")
        print(f"Without Helmet: {self.detection_stats['no_helmet_count']}")
        if self.detection_stats['total_detections'] > 0:
            compliance = (self.detection_stats['helmet_count'] / self.detection_stats['total_detections']) * 100
            print(f"Compliance Rate: {compliance:.2f}%")
        print(f"Total Violations: {len(self.detection_stats['violations'])}")
        print("="*50)

    def train_custom_model(self, dataset_path, epochs=100):
        """
        Train a custom helmet detection model
        
        Args:
            dataset_path: Path to YOLO format dataset
            epochs: Number of training epochs
        """
        self.logger.info(f"Starting custom model training with dataset: {dataset_path}")
        
        # Train the model
        results = self.model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=640,
            device=self.device,
            project='helmet_detection',
            name='custom_model'
        )
        
        self.logger.info("Training completed!")
        return results

    def save_statistics(self, filepath="helmet_detection_stats.json"):
        """Save detection statistics to file"""
        with open(filepath, 'w') as f:
            json.dump(self.detection_stats, f, indent=2)
        self.logger.info(f"Statistics saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Advanced Helmet Detection System')
    parser.add_argument('--mode', choices=['image', 'video', 'live'], default='live',
                       help='Detection mode')
    parser.add_argument('--source', default=0,
                       help='Source path (image/video file) or camera index')
    parser.add_argument('--model', default=None,
                       help='Path to custom trained model')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--save', action='store_true',
                       help='Save output results')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = AdvancedHelmetDetector(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    try:
        if args.mode == 'image':
            result = detector.detect_in_image(args.source, save_result=args.save)
            if result:
                print(f"Detection completed: {result['helmet_count']} with helmet, {result['no_helmet_count']} without helmet")
        
        elif args.mode == 'video':
            detector.detect_live_video(source=args.source, save_video=args.save)
        
        elif args.mode == 'live':
            detector.detect_live_video(source=int(args.source) if args.source.isdigit() else args.source, 
                                     save_video=args.save)
    
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Save statistics
        detector.save_statistics()

if __name__ == "__main__":
    main()
