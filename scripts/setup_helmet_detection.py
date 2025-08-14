"""
Setup script for helmet detection system
This script helps you set up the environment and download necessary models
"""

import os
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Requirements installed successfully!")

def download_models():
    """Download pre-trained models"""
    print("Downloading YOLOv8 models...")
    from ultralytics import YOLO
    
    # Download different model sizes
    models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
    
    for model in models:
        print(f"Downloading {model}...")
        YOLO(model)
    
    print("✅ Models downloaded successfully!")

def create_directories():
    """Create necessary directories"""
    directories = ['results', 'models', 'datasets', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    print("🚀 Setting up Advanced Helmet Detection System...")
    print("="*50)
    
    try:
        create_directories()
        install_requirements()
        download_models()
        
        print("\n" + "="*50)
        print("🎉 Setup completed successfully!")
        print("\nYou can now run the helmet detection system:")
        print("python helmet_detector.py --mode live")
        print("python helmet_detector.py --mode image --source path/to/image.jpg")
        print("python helmet_detector.py --mode video --source path/to/video.mp4")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
