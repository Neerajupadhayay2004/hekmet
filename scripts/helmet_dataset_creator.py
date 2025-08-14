import cv2
import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path

class HelmetDatasetCreator:
    def __init__(self):
        self.current_image = None
        self.current_image_path = None
        self.annotations = []
        self.image_files = []
        self.current_index = 0
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_class = 1  # 1 for helmet, 2 for no-helmet
        self.setup_gui()
        
    def setup_gui(self):
        """Setup annotation GUI"""
        self.root = tk.Tk()
        self.root.title("Helmet Dataset Creator & Annotator")
        self.root.geometry("1400x900")
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="🏷️ Helmet Dataset Creator", 
                              font=('Arial', 20, 'bold'))
        title_label.pack(pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        # File operations
        ttk.Button(control_frame, text="📁 Load Images", 
                  command=self.load_images).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="💾 Save Annotations", 
                  command=self.save_annotations).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📤 Export Dataset", 
                  command=self.export_dataset).pack(side=tk.LEFT, padx=5)
        
        # Navigation
        nav_frame = ttk.LabelFrame(main_frame, text="Navigation", padding=10)
        nav_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(nav_frame, text="⬅️ Previous", 
                  command=self.previous_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="➡️ Next", 
                  command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        self.image_counter = tk.Label(nav_frame, text="0/0")
        self.image_counter.pack(side=tk.LEFT, padx=20)
        
        # Class selection
        class_frame = ttk.LabelFrame(main_frame, text="Annotation Class", padding=10)
        class_frame.pack(fill=tk.X, pady=5)
        
        self.class_var = tk.IntVar(value=1)
        ttk.Radiobutton(class_frame, text="🛡️ Helmet", variable=self.class_var, 
                       value=1, command=self.update_class).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(class_frame, text="❌ No Helmet", variable=self.class_var, 
                       value=2, command=self.update_class).pack(side=tk.LEFT, padx=10)
        
        # Image and annotation area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Image canvas
        canvas_frame = ttk.LabelFrame(content_frame, text="Image", padding=5)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.canvas = tk.Canvas(canvas_frame, bg='gray', width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events for annotation
        self.canvas.bind("<Button-1>", self.start_annotation)
        self.canvas.bind("<B1-Motion>", self.draw_annotation)
        self.canvas.bind("<ButtonRelease-1>", self.end_annotation)
        
        # Annotations list
        annotations_frame = ttk.LabelFrame(content_frame, text="Annotations", padding=5)
        annotations_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Annotations listbox
        self.annotations_listbox = tk.Listbox(annotations_frame, width=30, height=20)
        self.annotations_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Annotation controls
        ann_control_frame = ttk.Frame(annotations_frame)
        ann_control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(ann_control_frame, text="🗑️ Delete", 
                  command=self.delete_annotation).pack(side=tk.LEFT, padx=2)
        ttk.Button(ann_control_frame, text="🔄 Clear All", 
                  command=self.clear_annotations).pack(side=tk.LEFT, padx=2)
        
        # Instructions
        instructions = """
        Instructions:
        1. Load images using 'Load Images' button
        2. Select annotation class (Helmet/No Helmet)
        3. Click and drag to draw bounding boxes
        4. Use navigation buttons to move between images
        5. Save annotations and export dataset when done
        
        Keyboard Shortcuts:
        - A/D: Previous/Next image
        - 1/2: Switch between Helmet/No Helmet
        - Del: Delete selected annotation
        """
        
        inst_frame = ttk.LabelFrame(main_frame, text="Instructions", padding=5)
        inst_frame.pack(fill=tk.X, pady=5)
        
        inst_label = tk.Label(inst_frame, text=instructions, justify=tk.LEFT, 
                             font=('Arial', 9))
        inst_label.pack()
        
        # Bind keyboard shortcuts
        self.root.bind('<Key-a>', lambda e: self.previous_image())
        self.root.bind('<Key-d>', lambda e: self.next_image())
        self.root.bind('<Key-1>', lambda e: self.set_class(1))
        self.root.bind('<Key-2>', lambda e: self.set_class(2))
        self.root.bind('<Delete>', lambda e: self.delete_annotation())
        self.root.focus_set()
        
    def load_images(self):
        """Load images from directory"""
        directory = filedialog.askdirectory(title="Select Images Directory")
        if not directory:
            return
            
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.image_files = []
        
        for ext in image_extensions:
            self.image_files.extend(Path(directory).glob(f'*{ext}'))
            self.image_files.extend(Path(directory).glob(f'*{ext.upper()}'))
        
        if not self.image_files:
            messagebox.showwarning("No Images", "No image files found in selected directory")
            return
        
        self.image_files.sort()
        self.current_index = 0
        self.load_current_image()
        
        messagebox.showinfo("Images Loaded", f"Loaded {len(self.image_files)} images")
    
    def load_current_image(self):
        """Load and display current image"""
        if not self.image_files or self.current_index >= len(self.image_files):
            return
        
        self.current_image_path = str(self.image_files[self.current_index])
        self.current_image = cv2.imread(self.current_image_path)
        
        if self.current_image is None:
            messagebox.showerror("Error", f"Could not load image: {self.current_image_path}")
            return
        
        # Load existing annotations if available
        self.load_existing_annotations()
        
        # Display image
        self.display_image()
        
        # Update counter
        self.image_counter.config(text=f"{self.current_index + 1}/{len(self.image_files)}")
    
    def display_image(self):
        """Display image on canvas"""
        if self.current_image is None:
            return
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, self.display_image)
            return
        
        height, width = rgb_image.shape[:2]
        
        # Calculate scaling factor
        scale_x = canvas_width / width
        scale_y = canvas_height / height
        self.scale = min(scale_x, scale_y)
        
        new_width = int(width * self.scale)
        new_height = int(height * self.scale)
        
        # Resize image
        resized_image = cv2.resize(rgb_image, (new_width, new_height))
        
        # Convert to PIL and display
        pil_image = Image.fromarray(resized_image)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, 
                               image=self.photo, anchor=tk.CENTER)
        
        # Draw existing annotations
        self.draw_annotations()
    
    def start_annotation(self, event):
        """Start drawing annotation"""
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y
    
    def draw_annotation(self, event):
        """Draw annotation rectangle"""
        if not self.drawing:
            return
        
        # Remove previous temporary rectangle
        self.canvas.delete("temp_rect")
        
        # Draw current rectangle
        color = "green" if self.current_class == 1 else "red"
        self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y,
                                   outline=color, width=2, tags="temp_rect")
    
    def end_annotation(self, event):
        """End drawing annotation"""
        if not self.drawing:
            return
        
        self.drawing = False
        
        # Calculate bounding box in original image coordinates
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Get image dimensions
        height, width = self.current_image.shape[:2]
        
        # Calculate offset (image is centered)
        img_display_width = int(width * self.scale)
        img_display_height = int(height * self.scale)
        
        offset_x = (canvas_width - img_display_width) // 2
        offset_y = (canvas_height - img_display_height) // 2
        
        # Convert canvas coordinates to image coordinates
        x1 = max(0, int((self.start_x - offset_x) / self.scale))
        y1 = max(0, int((self.start_y - offset_y) / self.scale))
        x2 = min(width, int((event.x - offset_x) / self.scale))
        y2 = min(height, int((event.y - offset_y) / self.scale))
        
        # Ensure valid bounding box
        if x2 > x1 and y2 > y1:
            # Convert to YOLO format (center_x, center_y, width, height) normalized
            center_x = (x1 + x2) / 2 / width
            center_y = (y1 + y2) / 2 / height
            bbox_width = (x2 - x1) / width
            bbox_height = (y2 - y1) / height
            
            # Add annotation
            annotation = {
                'class': self.current_class,
                'bbox': [center_x, center_y, bbox_width, bbox_height],
                'pixel_bbox': [x1, y1, x2, y2]
            }
            
            self.annotations.append(annotation)
            self.update_annotations_list()
            self.draw_annotations()
        
        # Remove temporary rectangle
        self.canvas.delete("temp_rect")
    
    def draw_annotations(self):
        """Draw all annotations on canvas"""
        # Remove existing annotation rectangles
        self.canvas.delete("annotation")
        
        if not self.annotations:
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        height, width = self.current_image.shape[:2]
        
        img_display_width = int(width * self.scale)
        img_display_height = int(height * self.scale)
        
        offset_x = (canvas_width - img_display_width) // 2
        offset_y = (canvas_height - img_display_height) // 2
        
        for i, ann in enumerate(self.annotations):
            x1, y1, x2, y2 = ann['pixel_bbox']
            
            # Convert to canvas coordinates
            canvas_x1 = int(x1 * self.scale) + offset_x
            canvas_y1 = int(y1 * self.scale) + offset_y
            canvas_x2 = int(x2 * self.scale) + offset_x
            canvas_y2 = int(y2 * self.scale) + offset_y
            
            # Draw rectangle
            color = "green" if ann['class'] == 1 else "red"
            self.canvas.create_rectangle(canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                                       outline=color, width=2, tags="annotation")
            
            # Add class label
            label = "Helmet" if ann['class'] == 1 else "No Helmet"
            self.canvas.create_text(canvas_x1, canvas_y1 - 10, text=label,
                                  fill=color, anchor=tk.W, tags="annotation")
    
    def update_annotations_list(self):
        """Update annotations listbox"""
        self.annotations_listbox.delete(0, tk.END)
        
        for i, ann in enumerate(self.annotations):
            class_name = "Helmet" if ann['class'] == 1 else "No Helmet"
            self.annotations_listbox.insert(tk.END, f"{i+1}. {class_name}")
    
    def update_class(self):
        """Update current annotation class"""
        self.current_class = self.class_var.get()
    
    def set_class(self, class_id):
        """Set annotation class via keyboard"""
        self.class_var.set(class_id)
        self.current_class = class_id
    
    def delete_annotation(self):
        """Delete selected annotation"""
        selection = self.annotations_listbox.curselection()
        if selection:
            index = selection[0]
            del self.annotations[index]
            self.update_annotations_list()
            self.draw_annotations()
    
    def clear_annotations(self):
        """Clear all annotations"""
        if messagebox.askyesno("Clear Annotations", "Are you sure you want to clear all annotations?"):
            self.annotations = []
            self.update_annotations_list()
            self.draw_annotations()
    
    def previous_image(self):
        """Go to previous image"""
        if self.current_index > 0:
            self.save_current_annotations()
            self.current_index -= 1
            self.load_current_image()
    
    def next_image(self):
        """Go to next image"""
        if self.current_index < len(self.image_files) - 1:
            self.save_current_annotations()
            self.current_index += 1
            self.load_current_image()
    
    def load_existing_annotations(self):
        """Load existing annotations for current image"""
        if not self.current_image_path:
            return
        
        # Look for annotation file
        image_path = Path(self.current_image_path)
        annotation_path = image_path.parent / f"{image_path.stem}.txt"
        
        self.annotations = []
        
        if annotation_path.exists():
            try:
                with open(annotation_path, 'r') as f:
                    lines = f.readlines()
                
                height, width = self.current_image.shape[:2]
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0]) + 1  # Convert from 0-based to 1-based
                        center_x, center_y, bbox_width, bbox_height = map(float, parts[1:])
                        
                        # Convert to pixel coordinates
                        x1 = int((center_x - bbox_width/2) * width)
                        y1 = int((center_y - bbox_height/2) * height)
                        x2 = int((center_x + bbox_width/2) * width)
                        y2 = int((center_y + bbox_height/2) * height)
                        
                        annotation = {
                            'class': class_id,
                            'bbox': [center_x, center_y, bbox_width, bbox_height],
                            'pixel_bbox': [x1, y1, x2, y2]
                        }
                        
                        self.annotations.append(annotation)
                        
            except Exception as e:
                print(f"Error loading annotations: {e}")
        
        self.update_annotations_list()
    
    def save_current_annotations(self):
        """Save annotations for current image"""
        if not self.current_image_path or not self.annotations:
            return
        
        # Create annotation file
        image_path = Path(self.current_image_path)
        annotation_path = image_path.parent / f"{image_path.stem}.txt"
        
        try:
            with open(annotation_path, 'w') as f:
                for ann in self.annotations:
                    class_id = ann['class'] - 1  # Convert to 0-based
                    center_x, center_y, bbox_width, bbox_height = ann['bbox']
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                    
        except Exception as e:
            print(f"Error saving annotations: {e}")
    
    def save_annotations(self):
        """Save all annotations"""
        self.save_current_annotations()
        messagebox.showinfo("Saved", "Annotations saved successfully!")
    
    def export_dataset(self):
        """Export dataset in YOLO format"""
        if not self.image_files:
            messagebox.showwarning("No Data", "No images loaded")
            return
        
        # Select export directory
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if not export_dir:
            return
        
        export_path = Path(export_dir)
        
        # Create directory structure
        (export_path / "images" / "train").mkdir(parents=True, exist_ok=True)
        (export_path / "images" / "val").mkdir(parents=True, exist_ok=True)
        (export_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (export_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # Split data (80% train, 20% val)
        np.random.shuffle(self.image_files)
        split_idx = int(0.8 * len(self.image_files))
        
        train_files = self.image_files[:split_idx]
        val_files = self.image_files[split_idx:]
        
        # Copy files
        self.copy_dataset_files(train_files, export_path, "train")
        self.copy_dataset_files(val_files, export_path, "val")
        
        # Create dataset.yaml
        dataset_config = {
            'path': str(export_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {
                0: 'person',
                1: 'helmet',
                2: 'no-helmet'
            },
            'nc': 3
        }
        
        with open(export_path / "dataset.yaml", 'w') as f:
            yaml.dump(dataset_config, f)
        
        messagebox.showinfo("Export Complete", 
                          f"Dataset exported to {export_dir}\n"
                          f"Train: {len(train_files)} images\n"
                          f"Val: {len(val_files)} images")
    
    def copy_dataset_files(self, file_list, export_path, split):
        """Copy files for dataset export"""
        for img_file in file_list:
            # Copy image
            dest_img = export_path / "images" / split / img_file.name
            shutil.copy2(img_file, dest_img)
            
            # Copy annotation if exists
            annotation_file = img_file.parent / f"{img_file.stem}.txt"
            if annotation_file.exists():
                dest_ann = export_path / "labels" / split / f"{img_file.stem}.txt"
                shutil.copy2(annotation_file, dest_ann)
    
    def run(self):
        """Run the annotation tool"""
        self.root.mainloop()

if __name__ == "__main__":
    import yaml
    import shutil
    
    app = HelmetDatasetCreator()
    app.run()
