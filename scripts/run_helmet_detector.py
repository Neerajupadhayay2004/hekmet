#!/usr/bin/env python3
"""
Advanced Helmet Detection System - Main Runner
Usage: python run_helmet_detector.py [options]
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from advanced_helmet_detector import AdvancedHelmetDetector, HelmetDetectorGUI
from helmet_training_system import HelmetTrainingSystem
from helmet_dataset_creator import HelmetDatasetCreator

def main():
    parser = argparse.ArgumentParser(description='Advanced Helmet Detection System')
    parser.add_argument('--mode', choices=['gui', 'cli', 'live', 'train', 'annotate'], 
                       default='gui', help='Operation mode')
    parser.add_argument('--image', type=str, help='Path to image file for CLI mode')
    parser.add_argument('--confidence', type=float, default=0.6, 
                       help='Confidence threshold (0.1-1.0)')
    parser.add_argument('--save', action='store_true', 
                       help='Save detection results')
    
    args = parser.parse_args()
    
    print("🛡️ Advanced Helmet Detection System")
    print("=" * 50)
    
    if args.mode == 'gui':
        print("🖥️ Starting GUI interface...")
        app = HelmetDetectorGUI()
        app.run()
        
    elif args.mode == 'cli':
        if not args.image:
            print("❌ Error: --image required for CLI mode")
            return
        
        print(f"🔍 Processing image: {args.image}")
        detector = AdvancedHelmetDetector()
        detector.confidence_threshold = args.confidence
        
        annotated_image, detections = detector.detect_helmet_in_image(
            args.image, save_result=args.save
        )
        
        if detections:
            print(f"✅ Found {len(detections)} detections:")
            for i, detection in enumerate(detections, 1):
                print(f"   {i}. {detection['class'].upper()} "
                      f"(Confidence: {detection['confidence']:.2f})")
        else:
            print("❌ No detections found")
            
    elif args.mode == 'live':
        print("🎥 Starting live detection...")
        detector = AdvancedHelmetDetector()
        detector.confidence_threshold = args.confidence
        detector.live_detection()
        
    elif args.mode == 'train':
        print("🚀 Starting training system...")
        trainer = HelmetTrainingSystem()
        trainer.train_custom_model()
        
    elif args.mode == 'annotate':
        print("🏷️ Starting annotation tool...")
        annotator = HelmetDatasetCreator()
        annotator.run()

if __name__ == "__main__":
    main()
