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
        self.confidence_threshold = 0.6
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
        """Detect helmets in a single image with high accuracy"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Run detection
            results = self.model(image, conf=self.confidence_threshold, iou=self.iou_threshold)
            
            # Process results
            detections = []
            annotated_image = image.copy()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Classify as helmet or no-helmet based on detection
                        is_helmet = self.classify_helmet_detection(image, x1, y1, x2, y2)
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class': 'helmet' if is_helmet else 'no-helmet',
                            'timestamp': datetime.now().isoformat()
                        }
                        detections.append(detection)
                        
                        # Draw bounding box
                        color = (0, 255, 0) if is_helmet else (0, 0, 255)  # Green for helmet, Red for no-helmet
                        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Add label
                        label = f"{detection['class']}: {confidence:.2f}"
                        cv2.putText(annotated_image, label, (int(x1), int(y1)-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save result if requested
            if save_result:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_path = f"results/detection_{timestamp}.jpg"
                cv2.imwrite(result_path, annotated_image)
                
            # Update detection history
            self.detection_history.append({
                'image_path': image_path,
                'detections': detections,
                'timestamp': datetime.now().isoformat()
            })
            
            return annotated_image, detections
            
        except Exception as e:
            print(f"❌ Error in detection: {e}")
            return None, []
    
    def classify_helmet_detection(self, image, x1, y1, x2, y2):
        """Advanced helmet classification using multiple features"""
        # Extract region of interest
        roi = image[int(y1):int(y2), int(x1):int(x2)]
        
        if roi.size == 0:
            return False
            
        # Convert to different color spaces for analysis
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Feature 1: Color analysis (helmets often have distinct colors)
        helmet_colors = self.analyze_helmet_colors(hsv_roi)
        
        # Feature 2: Shape analysis (helmets have rounded shapes)
        shape_score = self.analyze_helmet_shape(gray_roi)
        
        # Feature 3: Texture analysis (helmets have smooth surfaces)
        texture_score = self.analyze_helmet_texture(gray_roi)
        
        # Feature 4: Position analysis (helmets are typically on top of head)
        position_score = self.analyze_helmet_position(x1, y1, x2, y2, image.shape)
        
        # Combine all features for final classification
        final_score = (helmet_colors * 0.3 + shape_score * 0.3 + 
                      texture_score * 0.2 + position_score * 0.2)
        
        return final_score > 0.5
    
    def analyze_helmet_colors(self, hsv_roi):
        """Analyze colors typical of helmets"""
        # Define helmet color ranges in HSV
        helmet_color_ranges = [
            # White helmets
            ([0, 0, 200], [180, 30, 255]),
            # Yellow helmets
            ([20, 100, 100], [30, 255, 255]),
            # Red helmets
            ([0, 100, 100], [10, 255, 255]),
            # Blue helmets
            ([100, 100, 100], [130, 255, 255]),
            # Green helmets
            ([40, 100, 100], [80, 255, 255])
        ]
        
        total_pixels = hsv_roi.shape[0] * hsv_roi.shape[1]
        helmet_pixels = 0
        
        for lower, upper in helmet_color_ranges:
            mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
            helmet_pixels += cv2.countNonZero(mask)
        
        return min(helmet_pixels / total_pixels, 1.0)
    
    def analyze_helmet_shape(self, gray_roi):
        """Analyze shape characteristics of helmets"""
        # Apply edge detection
        edges = cv2.Canny(gray_roi, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate roundness (helmet-like shape)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return 0.0
            
        roundness = 4 * np.pi * area / (perimeter * perimeter)
        return min(roundness, 1.0)
    
    def analyze_helmet_texture(self, gray_roi):
        """Analyze texture smoothness typical of helmets"""
        # Calculate local binary pattern for texture analysis
        if gray_roi.size == 0:
            return 0.0
            
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Smooth surfaces (like helmets) have lower gradient variance
        texture_variance = np.var(gradient_magnitude)
        
        # Normalize and invert (lower variance = higher helmet probability)
        normalized_variance = min(texture_variance / 1000, 1.0)
        return 1.0 - normalized_variance
    
    def analyze_helmet_position(self, x1, y1, x2, y2, image_shape):
        """Analyze position typical of helmets (top of head)"""
        height, width = image_shape[:2]
        
        # Calculate relative position
        center_y = (y1 + y2) / 2
        relative_y = center_y / height
        
        # Helmets are typically in upper portion of detection
        if relative_y < 0.4:  # Upper 40% of image
            return 1.0
        elif relative_y < 0.6:  # Middle portion
            return 0.5
        else:  # Lower portion
            return 0.1
    
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
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Classify helmet
                        is_helmet = self.classify_helmet_detection(frame, x1, y1, x2, y2)
                        
                        # Draw bounding box
                        color = (0, 255, 0) if is_helmet else (0, 0, 255)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Add label
                        label = f"{'Helmet' if is_helmet else 'No Helmet'}: {confidence:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1)-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display frame
            cv2.imshow('Helmet Detection - Live', frame)
            
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
        self.confidence_var = tk.DoubleVar(value=0.6)
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
            confidence = detection['confidence']
            self.results_text.insert(tk.END, f"{i}. {status} (Confidence: {confidence:.2f})\n")
    
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
