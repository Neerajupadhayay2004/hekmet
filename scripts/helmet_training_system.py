import os
import yaml
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import shutil

class HelmetTrainingSystem:
    def __init__(self):
        self.setup_training_environment()
        
    def setup_training_environment(self):
        """Setup training directories and configuration"""
        # Create training directory structure
        directories = [
            'training_data/images/train',
            'training_data/images/val',
            'training_data/labels/train',
            'training_data/labels/val',
            'models/training_runs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Create dataset configuration
        self.create_dataset_config()
    
    def create_dataset_config(self):
        """Create YOLO dataset configuration"""
        config = {
            'path': str(Path('training_data').absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {
                0: 'person',
                1: 'helmet',
                2: 'no-helmet'
            },
            'nc': 3  # number of classes
        }
        
        with open('training_data/dataset.yaml', 'w') as f:
            yaml.dump(config, f)
    
    def prepare_training_data(self, source_images_dir, annotations_dir=None):
        """Prepare training data from source images"""
        print("🔄 Preparing training data...")
        
        if not os.path.exists(source_images_dir):
            print(f"❌ Source directory not found: {source_images_dir}")
            return False
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(source_images_dir).glob(f'*{ext}'))
            image_files.extend(Path(source_images_dir).glob(f'*{ext.upper()}'))
        
        if not image_files:
            print("❌ No image files found in source directory")
            return False
        
        print(f"📁 Found {len(image_files)} images")
        
        # Split data (80% train, 20% validation)
        np.random.shuffle(image_files)
        split_idx = int(0.8 * len(image_files))
        
        train_images = image_files[:split_idx]
        val_images = image_files[split_idx:]
        
        # Copy images and create labels
        self.process_image_set(train_images, 'train', annotations_dir)
        self.process_image_set(val_images, 'val', annotations_dir)
        
        print("✅ Training data preparation complete!")
        return True
    
    def process_image_set(self, image_files, split, annotations_dir):
        """Process a set of images for training"""
        for img_file in image_files:
            # Copy image
            dest_img = f'training_data/images/{split}/{img_file.name}'
            shutil.copy2(img_file, dest_img)
            
            # Create or copy label file
            label_file = f'training_data/labels/{split}/{img_file.stem}.txt'
            
            if annotations_dir and os.path.exists(f'{annotations_dir}/{img_file.stem}.txt'):
                # Copy existing annotation
                shutil.copy2(f'{annotations_dir}/{img_file.stem}.txt', label_file)
            else:
                # Create auto-annotation using pre-trained model
                self.auto_annotate_image(str(img_file), label_file)
    
    def auto_annotate_image(self, image_path, label_path):
        """Auto-annotate image using pre-trained model"""
        try:
            # Load pre-trained model for initial annotation
            model = YOLO('yolov8n.pt')
            
            # Run detection
            results = model(image_path, conf=0.3)
            
            # Get image dimensions
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            
            annotations = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get normalized coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Convert to YOLO format (center_x, center_y, width, height)
                        center_x = ((x1 + x2) / 2) / width
                        center_y = ((y1 + y2) / 2) / height
                        bbox_width = (x2 - x1) / width
                        bbox_height = (y2 - y1) / height
                        
                        # Classify as person (will need manual review)
                        class_id = 0  # person class
                        
                        annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")
            
            # Save annotations
            with open(label_path, 'w') as f:
                f.write('\n'.join(annotations))
                
        except Exception as e:
            print(f"⚠️ Auto-annotation failed for {image_path}: {e}")
            # Create empty annotation file
            with open(label_path, 'w') as f:
                f.write('')
    
    def train_custom_model(self, epochs=100, batch_size=16, image_size=640):
        """Train custom helmet detection model"""
        print("🚀 Starting model training...")
        
        # Check if training data exists
        if not os.path.exists('training_data/dataset.yaml'):
            print("❌ Training data not found. Please prepare training data first.")
            return False
        
        try:
            # Initialize model
            model = YOLO('yolov8n.pt')  # Start with pre-trained weights
            
            # Training parameters
            training_args = {
                'data': 'training_data/dataset.yaml',
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': image_size,
                'project': 'models/training_runs',
                'name': f'helmet_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'save': True,
                'save_period': 10,  # Save every 10 epochs
                'cache': True,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'workers': 4,
                'patience': 20,  # Early stopping patience
                'optimizer': 'AdamW',
                'lr0': 0.01,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'pose': 12.0,
                'kobj': 1.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': 0.0
            }
            
            # Start training
            results = model.train(**training_args)
            
            # Save best model
            best_model_path = f"models/custom_helmet_model.pt"
            shutil.copy2(results.save_dir / 'weights' / 'best.pt', best_model_path)
            
            print(f"✅ Training completed! Best model saved to: {best_model_path}")
            
            # Generate training report
            self.generate_training_report(results)
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def generate_training_report(self, results):
        """Generate training performance report"""
        report = {
            'training_completed': datetime.now().isoformat(),
            'model_path': 'models/custom_helmet_model.pt',
            'training_metrics': {
                'final_map50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                'final_map50_95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                'final_precision': float(results.results_dict.get('metrics/precision(B)', 0)),
                'final_recall': float(results.results_dict.get('metrics/recall(B)', 0))
            },
            'training_config': {
                'epochs': results.args.epochs,
                'batch_size': results.args.batch,
                'image_size': results.args.imgsz,
                'device': results.args.device
            }
        }
        
        # Save report
        with open('models/training_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("📊 Training Report:")
        print(f"   mAP@0.5: {report['training_metrics']['final_map50']:.3f}")
        print(f"   mAP@0.5:0.95: {report['training_metrics']['final_map50_95']:.3f}")
        print(f"   Precision: {report['training_metrics']['final_precision']:.3f}")
        print(f"   Recall: {report['training_metrics']['final_recall']:.3f}")
    
    def validate_model(self, model_path='models/custom_helmet_model.pt'):
        """Validate trained model performance"""
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return False
        
        try:
            model = YOLO(model_path)
            
            # Run validation
            results = model.val(data='training_data/dataset.yaml')
            
            print("✅ Model Validation Results:")
            print(f"   mAP@0.5: {results.box.map50:.3f}")
            print(f"   mAP@0.5:0.95: {results.box.map:.3f}")
            print(f"   Precision: {results.box.mp:.3f}")
            print(f"   Recall: {results.box.mr:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False

def main():
    """Main training interface"""
    trainer = HelmetTrainingSystem()
    
    print("🛡️ Helmet Detection Training System")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Prepare training data")
        print("2. Train custom model")
        print("3. Validate model")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            source_dir = input("Enter source images directory path: ").strip()
            annotations_dir = input("Enter annotations directory (optional, press Enter to skip): ").strip()
            annotations_dir = annotations_dir if annotations_dir else None
            
            trainer.prepare_training_data(source_dir, annotations_dir)
            
        elif choice == '2':
            epochs = int(input("Enter number of epochs (default 100): ") or 100)
            batch_size = int(input("Enter batch size (default 16): ") or 16)
            
            trainer.train_custom_model(epochs=epochs, batch_size=batch_size)
            
        elif choice == '3':
            model_path = input("Enter model path (default: models/custom_helmet_model.pt): ").strip()
            model_path = model_path if model_path else 'models/custom_helmet_model.pt'
            
            trainer.validate_model(model_path)
            
        elif choice == '4':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
