import cv2
import numpy as np
import torch
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import threading
import time

class AdvancedHelmetDetector:
    def __init__(self):
        self.model = None
        self.confidence_threshold = 0.4  # lowered threshold for better detection
        self.iou_threshold = 0.45
        self.detection_history = []
        self.setup_directories()
        self.load_model()
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = ['models', 'uploads', 'results', 'training_data', 'logs']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def load_model(self):
        """Load YOLOv8 model"""
        try:
            # Try to load custom trained model first
            if os.path.exists('models/custom_helmet_model.pt'):
                self.model = YOLO('models/custom_helmet_model.pt')
                print("✅ Custom helmet model loaded successfully!")
            else:
                # Load pre-trained YOLOv8 model
                self.model = YOLO('yolov8n.pt')
                print("✅ Pre-trained YOLOv8 model loaded!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def detect_helmet_in_image(self, image_path, save_result=True):
        """Detect helmets in a single image with high accuracy from all angles"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Preprocess image for better detection
            processed_images = self.preprocess_image_for_angles(image)
            
            all_detections = []
            best_annotated_image = image.copy()
            
            # Process each angle/preprocessing variant
            for processed_img in processed_images:
                # Run detection
                results = self.model(processed_img, conf=self.confidence_threshold, iou=self.iou_threshold)
                
                # Process results
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get coordinates and confidence
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            # Enhanced helmet classification
                            helmet_score = self.enhanced_helmet_classification(processed_img, x1, y1, x2, y2)
                            
                            # Only add if helmet score is high enough
                            if helmet_score > 0.3:
                                detection = {
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': float(confidence),
                                    'helmet_score': float(helmet_score),
                                    'class': 'helmet' if helmet_score > 0.6 else 'no-helmet',
                                    'timestamp': datetime.now().isoformat()
                                }
                                all_detections.append(detection)
            
            # Remove duplicate detections using NMS
            final_detections = self.apply_custom_nms(all_detections)
            
            # Annotate the original image with final detections
            annotated_image = self.annotate_image(image, final_detections)
            
            # Save result if requested
            if save_result:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_path = f"results/detection_{timestamp}.jpg"
                cv2.imwrite(result_path, annotated_image)
                
            # Update detection history
            self.detection_history.append({
                'image_path': image_path,
                'detections': final_detections,
                'timestamp': datetime.now().isoformat()
            })
            
            return annotated_image, final_detections
            
        except Exception as e:
            print(f"❌ Error in detection: {e}")
            return None, []
    
    def preprocess_image_for_angles(self, image):
        """Preprocess image to handle detection from different angles"""
        processed_images = [image]  # Original image
        
        # Add rotated versions for better angle detection
        angles = [-15, -10, -5, 5, 10, 15]
        for angle in angles:
            rotated = self.rotate_image(image, angle)
            processed_images.append(rotated)
        
        # Add brightness/contrast variations
        bright_img = cv2.convertScaleAbs(image, alpha=1.2, beta=20)
        dark_img = cv2.convertScaleAbs(image, alpha=0.8, beta=-20)
        processed_images.extend([bright_img, dark_img])
        
        # Add histogram equalization for better contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
        enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        processed_images.append(enhanced_img)
        
        return processed_images
    
    def rotate_image(self, image, angle):
        """Rotate image by given angle"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        return rotated
    
    def enhanced_helmet_classification(self, image, x1, y1, x2, y2):
        """Enhanced helmet classification using multiple advanced features"""
        # Extract region of interest
        roi = image[int(y1):int(y2), int(x1):int(x2)]
        
        if roi.size == 0:
            return 0.0
            
        # Convert to different color spaces for analysis
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Feature 1: Enhanced color analysis
        color_score = self.analyze_helmet_colors_enhanced(hsv_roi)
        
        # Feature 2: Advanced shape analysis
        shape_score = self.analyze_helmet_shape_enhanced(gray_roi)
        
        # Feature 3: Texture analysis with multiple methods
        texture_score = self.analyze_helmet_texture_enhanced(gray_roi)
        
        # Feature 4: Position and size analysis
        position_score = self.analyze_helmet_position_enhanced(x1, y1, x2, y2, image.shape)
        
        # Feature 5: Edge density analysis
        edge_score = self.analyze_helmet_edges(gray_roi)
        
        # Feature 6: Reflectivity analysis (helmets are often reflective)
        reflectivity_score = self.analyze_helmet_reflectivity(roi)
        
        # Combine all features with optimized weights
        final_score = (color_score * 0.25 + shape_score * 0.25 + 
                      texture_score * 0.15 + position_score * 0.15 +
                      edge_score * 0.1 + reflectivity_score * 0.1)
        
        return final_score
    
    def analyze_helmet_colors_enhanced(self, hsv_roi):
        """Enhanced color analysis for helmet detection"""
        # Extended helmet color ranges in HSV
        helmet_color_ranges = [
            # White/Light colors (most common for safety helmets)
            ([0, 0, 180], [180, 40, 255]),
            # Yellow (construction helmets)
            ([15, 80, 80], [35, 255, 255]),
            # Red (safety helmets)
            ([0, 80, 80], [15, 255, 255]),
            ([165, 80, 80], [180, 255, 255]),
            # Blue (industrial helmets)
            ([90, 80, 80], [130, 255, 255]),
            # Green (safety helmets)
            ([35, 80, 80], [85, 255, 255]),
            # Orange (construction helmets)
            ([5, 80, 80], [25, 255, 255]),
            # Gray (industrial helmets)
            ([0, 0, 50], [180, 40, 180])
        ]
        
        total_pixels = hsv_roi.shape[0] * hsv_roi.shape[1]
        helmet_pixels = 0
        
        for lower, upper in helmet_color_ranges:
            mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
            helmet_pixels += cv2.countNonZero(mask)
        
        color_ratio = min(helmet_pixels / total_pixels, 1.0)
        
        # Boost score for dominant helmet colors
        if color_ratio > 0.6:
            color_ratio *= 1.2
        
        return min(color_ratio, 1.0)
    
    def analyze_helmet_shape_enhanced(self, gray_roi):
        """Enhanced shape analysis for helmet detection"""
        # Apply multiple edge detection methods
        edges1 = cv2.Canny(gray_roi, 30, 100)
        edges2 = cv2.Canny(gray_roi, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Analyze multiple contours, not just the largest
        shape_scores = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Skip very small contours
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # Calculate multiple shape metrics
            roundness = 4 * np.pi * area / (perimeter * perimeter)
            
            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Helmets typically have aspect ratio between 0.8 and 1.5
            aspect_score = 1.0 if 0.8 <= aspect_ratio <= 1.5 else 0.5
            
            # Calculate convexity (helmets are generally convex)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / hull_area if hull_area > 0 else 0
            
            # Combine shape metrics
            shape_score = (roundness * 0.4 + aspect_score * 0.3 + convexity * 0.3)
            shape_scores.append(shape_score)
        
        return max(shape_scores) if shape_scores else 0.0
    
    def analyze_helmet_texture_enhanced(self, gray_roi):
        """Enhanced texture analysis for helmet detection"""
        if gray_roi.size == 0:
            return 0.0
        
        # Multiple texture analysis methods
        
        # 1. Gradient-based texture analysis
        grad_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        texture_variance = np.var(gradient_magnitude)
        gradient_score = max(0, 1.0 - min(texture_variance / 1000, 1.0))
        
        # 2. Local Binary Pattern analysis
        lbp_score = self.calculate_lbp_uniformity(gray_roi)
        
        # 3. Smoothness analysis using Laplacian
        laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F)
        smoothness = 1.0 - min(np.var(laplacian) / 10000, 1.0)
        
        # Combine texture scores
        final_texture_score = (gradient_score * 0.4 + lbp_score * 0.3 + smoothness * 0.3)
        
        return final_texture_score
    
    def calculate_lbp_uniformity(self, gray_roi):
        """Calculate Local Binary Pattern uniformity (helmets have uniform texture)"""
        if gray_roi.shape[0] < 3 or gray_roi.shape[1] < 3:
            return 0.0
        
        # Simple LBP calculation
        lbp = np.zeros_like(gray_roi)
        
        for i in range(1, gray_roi.shape[0] - 1):
            for j in range(1, gray_roi.shape[1] - 1):
                center = gray_roi[i, j]
                binary_string = ''
                
                # 8-neighborhood
                neighbors = [
                    gray_roi[i-1, j-1], gray_roi[i-1, j], gray_roi[i-1, j+1],
                    gray_roi[i, j+1], gray_roi[i+1, j+1], gray_roi[i+1, j],
                    gray_roi[i+1, j-1], gray_roi[i, j-1]
                ]
                
                for neighbor in neighbors:
                    binary_string += '1' if neighbor >= center else '0'
                
                lbp[i, j] = int(binary_string, 2)
        
        # Calculate uniformity (lower variance = more uniform = more helmet-like)
        lbp_variance = np.var(lbp)
        uniformity_score = max(0, 1.0 - min(lbp_variance / 10000, 1.0))
        
        return uniformity_score
    
    def analyze_helmet_position_enhanced(self, x1, y1, x2, y2, image_shape):
        """Enhanced position analysis for helmet detection"""
        height, width = image_shape[:2]
        
        # Calculate relative position and size
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        relative_x = center_x / width
        relative_y = center_y / height
        relative_size = (bbox_width * bbox_height) / (width * height)
        
        # Position scoring
        position_score = 1.0
        
        # Helmets are typically in upper portion
        if relative_y < 0.3:  # Upper 30%
            position_score *= 1.2
        elif relative_y < 0.6:  # Middle portion
            position_score *= 1.0
        else:  # Lower portion
            position_score *= 0.6
        
        # Size scoring (helmets have reasonable size)
        if 0.01 <= relative_size <= 0.3:  # Reasonable helmet size
            position_score *= 1.1
        elif relative_size > 0.5:  # Too large
            position_score *= 0.5
        elif relative_size < 0.005:  # Too small
            position_score *= 0.3
        
        return min(position_score, 1.0)
    
    def analyze_helmet_edges(self, gray_roi):
        """Analyze edge characteristics typical of helmets"""
        if gray_roi.size == 0:
            return 0.0
        
        # Apply edge detection
        edges = cv2.Canny(gray_roi, 50, 150)
        
        # Calculate edge density
        edge_pixels = cv2.countNonZero(edges)
        total_pixels = gray_roi.shape[0] * gray_roi.shape[1]
        edge_density = edge_pixels / total_pixels
        
        # Helmets typically have moderate edge density (not too smooth, not too textured)
        if 0.1 <= edge_density <= 0.4:
            return 1.0
        elif 0.05 <= edge_density <= 0.6:
            return 0.7
        else:
            return 0.3
    
    def analyze_helmet_reflectivity(self, roi):
        """Analyze reflectivity characteristics of helmets"""
        if roi.size == 0:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Find bright spots (potential reflections)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bright_pixels = cv2.countNonZero(bright_mask)
        total_pixels = gray.shape[0] * gray.shape[1]
        
        brightness_ratio = bright_pixels / total_pixels
        
        # Helmets often have some reflective areas but not too many
        if 0.05 <= brightness_ratio <= 0.3:
            return 1.0
        elif 0.01 <= brightness_ratio <= 0.5:
            return 0.7
        else:
            return 0.4
    
    def apply_custom_nms(self, detections, iou_threshold=0.5):
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if not detections:
            return []
        
        # Sort by helmet_score (higher is better)
        detections.sort(key=lambda x: x['helmet_score'], reverse=True)
        
        final_detections = []
        
        for detection in detections:
            # Check if this detection overlaps significantly with any already selected
            overlap = False
            for final_det in final_detections:
                iou = self.calculate_iou(detection['bbox'], final_det['bbox'])
                if iou > iou_threshold:
                    overlap = True
                    break
            
            if not overlap:
                final_detections.append(detection)
        
        return final_detections
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def annotate_image(self, image, detections):
        """Annotate image with enhanced detection results"""
        annotated = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            helmet_score = detection['helmet_score']
            is_helmet = detection['class'] == 'helmet'
            
            # Choose color based on detection
            if is_helmet:
                color = (0, 255, 0)  # Green for helmet
                status = "HELMET"
            else:
                color = (0, 0, 255)  # Red for no helmet
                status = "NO HELMET"
            
            # Draw bounding box with thickness based on confidence
            thickness = max(2, int(helmet_score * 5))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Add label with background
            label = f"{status}: {helmet_score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated

    def live_detection(self):
        """Real-time helmet detection from webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("🎥 Starting live detection... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection on frame
            results = self.model(frame, conf=self.confidence_threshold, iou=self.iou_threshold)
            
            # Process and annotate frame
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Enhanced helmet classification
                        helmet_score = self.enhanced_helmet_classification(frame, x1, y1, x2, y2)
                        
                        if helmet_score > 0.3:
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence),
                                'helmet_score': float(helmet_score),
                                'class': 'helmet' if helmet_score > 0.6 else 'no-helmet'
                            }
                            detections.append(detection)
            
            # Annotate frame
            annotated_frame = self.annotate_image(frame, detections)
            
            # Display frame
            cv2.imshow('Helmet Detection - Live', annotated_frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def generate_detection_report(self):
        """Generate comprehensive detection report"""
        if not self.detection_history:
            print("No detection history available")
            return
        
        # Calculate statistics
        total_detections = len(self.detection_history)
        helmet_count = 0
        no_helmet_count = 0
        
        for record in self.detection_history:
            for detection in record['detections']:
                if detection['class'] == 'helmet':
                    helmet_count += 1
                else:
                    no_helmet_count += 1
        
        # Create report
        report = {
            'total_images_processed': total_detections,
            'total_detections': helmet_count + no_helmet_count,
            'helmet_detections': helmet_count,
            'no_helmet_detections': no_helmet_count,
            'compliance_rate': helmet_count / (helmet_count + no_helmet_count) * 100 if (helmet_count + no_helmet_count) > 0 else 0,
            'generated_at': datetime.now().isoformat()
        }
        
        # Save report
        with open('logs/detection_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Detection Report Generated:")
        print(f"   Total Images: {report['total_images_processed']}")
        print(f"   Helmet Detections: {report['helmet_detections']}")
        print(f"   No-Helmet Detections: {report['no_helmet_detections']}")
        print(f"   Compliance Rate: {report['compliance_rate']:.1f}%")
        
        return report

class HelmetDetectorGUI:
    def __init__(self):
        self.detector = AdvancedHelmetDetector()
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Advanced Helmet Detector Pro")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="🛡️ Advanced Helmet Detector Pro", 
                              font=('Arial', 24, 'bold'), bg='#2c3e50', fg='white')
        title_label.pack(pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Buttons
        ttk.Button(control_frame, text="📁 Upload Image", 
                  command=self.upload_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🎥 Live Detection", 
                  command=self.start_live_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📊 Generate Report", 
                  command=self.generate_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🔧 Train Model", 
                  command=self.train_model).pack(side=tk.LEFT, padx=5)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=5)
        
        # Confidence threshold
        tk.Label(settings_frame, text="Confidence Threshold:").pack(side=tk.LEFT)
        self.confidence_var = tk.DoubleVar(value=0.4)
        confidence_scale = tk.Scale(settings_frame, from_=0.1, to=1.0, resolution=0.1,
                                  orient=tk.HORIZONTAL, variable=self.confidence_var,
                                  command=self.update_confidence)
        confidence_scale.pack(side=tk.LEFT, padx=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Detection Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Image display
        self.image_label = tk.Label(results_frame, text="Upload an image to start detection",
                                   bg='#34495e', fg='white', font=('Arial', 12))
        self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Results text
        self.results_text = tk.Text(results_frame, width=40, height=20)
        self.results_text.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(main_frame, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def update_confidence(self, value):
        """Update confidence threshold"""
        self.detector.confidence_threshold = float(value)
        self.status_var.set(f"Confidence threshold updated to {value}")
    
    def upload_image(self):
        """Upload and process image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.status_var.set("Processing image...")
            self.root.update()
            
            # Process image
            annotated_image, detections = self.detector.detect_helmet_in_image(file_path)
            
            if annotated_image is not None:
                # Display image
                self.display_image(annotated_image)
                
                # Display results
                self.display_results(detections)
                
                self.status_var.set(f"Detection complete - Found {len(detections)} objects")
            else:
                messagebox.showerror("Error", "Failed to process image")
                self.status_var.set("Error processing image")
    
    def display_image(self, cv_image):
        """Display image in GUI"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Resize image to fit display
        height, width = rgb_image.shape[:2]
        max_size = 600
        
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        
        rgb_image = cv2.resize(rgb_image, (new_width, new_height))
        
        # Convert to PIL and display
        pil_image = Image.fromarray(rgb_image)
        photo = ImageTk.PhotoImage(pil_image)
        
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo
    
    def display_results(self, detections):
        """Display detection results"""
        self.results_text.delete(1.0, tk.END)
        
        if not detections:
            self.results_text.insert(tk.END, "No detections found.\n")
            return
        
        helmet_count = sum(1 for d in detections if d['class'] == 'helmet')
        no_helmet_count = len(detections) - helmet_count
        
        self.results_text.insert(tk.END, f"🛡️ DETECTION SUMMARY\n")
        self.results_text.insert(tk.END, f"{'='*30}\n")
        self.results_text.insert(tk.END, f"Total Detections: {len(detections)}\n")
        self.results_text.insert(tk.END, f"✅ Helmets: {helmet_count}\n")
        self.results_text.insert(tk.END, f"❌ No Helmets: {no_helmet_count}\n")
        
        if len(detections) > 0:
            compliance = helmet_count / len(detections) * 100
            self.results_text.insert(tk.END, f"📊 Compliance: {compliance:.1f}%\n\n")
        
        self.results_text.insert(tk.END, "DETAILED RESULTS:\n")
        self.results_text.insert(tk.END, f"{'='*30}\n")
        
        for i, detection in enumerate(detections, 1):
            status = "✅ HELMET" if detection['class'] == 'helmet' else "❌ NO HELMET"
            helmet_score = detection['helmet_score']
            confidence = detection['confidence']
            self.results_text.insert(tk.END, f"{i}. {status}\n")
            self.results_text.insert(tk.END, f"   Helmet Score: {helmet_score:.3f}\n")
            self.results_text.insert(tk.END, f"   Detection Confidence: {confidence:.3f}\n\n")
    
    def start_live_detection(self):
        """Start live detection in separate thread"""
        self.status_var.set("Starting live detection...")
        threading.Thread(target=self.detector.live_detection, daemon=True).start()
    
    def generate_report(self):
        """Generate detection report"""
        self.status_var.set("Generating report...")
        report = self.detector.generate_detection_report()
        
        if report:
            messagebox.showinfo("Report Generated", 
                              f"Report saved to logs/detection_report.json\n\n"
                              f"Compliance Rate: {report['compliance_rate']:.1f}%")
            self.status_var.set("Report generated successfully")
        else:
            messagebox.showwarning("No Data", "No detection history available")
            self.status_var.set("No data for report")
    
    def train_model(self):
        """Launch model training"""
        messagebox.showinfo("Training", "Model training will start. This may take several hours.")
        self.status_var.set("Training model... (Check console for progress)")
        # Training would be implemented in separate script
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

if __name__ == "__main__":
    # Create GUI application
    app = HelmetDetectorGUI()
    app.run()

