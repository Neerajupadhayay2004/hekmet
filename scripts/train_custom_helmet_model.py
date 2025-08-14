"""
Script to train a custom helmet detection model
This script helps you train a specialized model for better helmet detection accuracy
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import shutil

class HelmetModelTrainer:
    def __init__(self):
        self.project_dir = Path("helmet_training")
        self.project_dir.mkdir(exist_ok=True)
    
    def create_dataset_config(self, train_path, val_path, test_path=None):
        """Create dataset configuration file"""
        config = {
            'path': str(self.project_dir.absolute()),
            'train': str(Path(train_path).relative_to(self.project_dir)),
            'val': str(Path(val_path).relative_to(self.project_dir)),
            'nc': 3,  # number of classes
            'names': {
                0: 'person',
                1: 'helmet', 
                2: 'no_helmet'
            }
        }
        
        if test_path:
            config['test'] = str(Path(test_path).relative_to(self.project_dir))
        
        config_path = self.project_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Dataset config created: {config_path}")
        return config_path
    
    def prepare_sample_dataset(self):
        """Create a sample dataset structure for demonstration"""
        print("Creating sample dataset structure...")
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            (self.project_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.project_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Create sample annotation format guide
        sample_annotation = """# YOLO Format Annotation Guide
# Each line represents one object in the image
# Format: class_id center_x center_y width height
# All coordinates are normalized (0-1)

# Example annotations:
# 0 0.5 0.3 0.2 0.4    # person at center-top
# 1 0.5 0.15 0.1 0.1   # helmet on person's head
# 2 0.7 0.15 0.1 0.1   # no helmet (bare head)

# Classes:
# 0: person
# 1: helmet
# 2: no_helmet (bare head/person without helmet)
"""
        
        with open(self.project_dir / "annotation_guide.txt", 'w') as f:
            f.write(sample_annotation)
        
        print("✅ Sample dataset structure created!")
        print(f"📁 Dataset location: {self.project_dir}")
        print("📝 Add your images and annotations to train/val/test folders")
        
    def train_model(self, dataset_config, model_size='yolov8n', epochs=100, img_size=640):
        """Train the helmet detection model"""
        print(f"🚀 Starting training with {model_size} for {epochs} epochs...")
        
        # Initialize model
        model = YOLO(f'{model_size}.pt')
        
        # Train the model
        results = model.train(
            data=dataset_config,
            epochs=epochs,
            imgsz=img_size,
            project='helmet_detection_training',
            name='helmet_model',
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            device='auto',   # Automatically select GPU if available
            workers=8,
            batch=16,
            patience=50,     # Early stopping patience
            optimizer='AdamW',
            lr0=0.01,
            weight_decay=0.0005,
            warmup_epochs=3,
            box=7.5,         # Box loss gain
            cls=0.5,         # Class loss gain
            dfl=1.5,         # DFL loss gain
            pose=12.0,       # Pose loss gain
            kobj=1.0,        # Keypoint obj loss gain
            label_smoothing=0.0,
            nbs=64,          # Nominal batch size
            hsv_h=0.015,     # HSV-Hue augmentation
            hsv_s=0.7,       # HSV-Saturation augmentation
            hsv_v=0.4,       # HSV-Value augmentation
            degrees=0.0,     # Rotation augmentation
            translate=0.1,   # Translation augmentation
            scale=0.5,       # Scale augmentation
            shear=0.0,       # Shear augmentation
            perspective=0.0, # Perspective augmentation
            flipud=0.0,      # Vertical flip augmentation
            fliplr=0.5,      # Horizontal flip augmentation
            mosaic=1.0,      # Mosaic augmentation
            mixup=0.0,       # Mixup augmentation
            copy_paste=0.0,  # Copy-paste augmentation
        )
        
        print("✅ Training completed!")
        return results
    
    def evaluate_model(self, model_path, dataset_config):
        """Evaluate the trained model"""
        print("📊 Evaluating model performance...")
        
        model = YOLO(model_path)
        results = model.val(data=dataset_config)
        
        print("✅ Evaluation completed!")
        return results
    
    def export_model(self, model_path, formats=['onnx', 'torchscript']):
        """Export model to different formats"""
        print("📦 Exporting model...")
        
        model = YOLO(model_path)
        
        for format in formats:
            model.export(format=format)
            print(f"✅ Exported to {format}")

def main():
    print("🎯 Helmet Detection Model Training")
    print("="*40)
    
    trainer = HelmetModelTrainer()
    
    # Create sample dataset structure
    trainer.prepare_sample_dataset()
    
    print("\n📋 Next Steps:")
    print("1. Add your helmet/no-helmet images to the train/val folders")
    print("2. Create YOLO format annotations (.txt files)")
    print("3. Run training with your dataset")
    print("\nExample training command:")
    print("python train_custom_helmet_model.py --train")
    
    # Uncomment below to start training (after preparing dataset)
    """
    # Create dataset config
    config_path = trainer.create_dataset_config(
        train_path="helmet_training/train",
        val_path="helmet_training/val"
    )
    
    # Train model
    results = trainer.train_model(config_path, epochs=100)
    
    # Evaluate model
    best_model = "helmet_detection_training/helmet_model/weights/best.pt"
    trainer.evaluate_model(best_model, config_path)
    
    # Export model
    trainer.export_model(best_model)
    """

if __name__ == "__main__":
    main()
