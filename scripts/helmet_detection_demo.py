"""
Demo script to test helmet detection with sample images
"""

import cv2
import numpy as np
from helmet_detector import AdvancedHelmetDetector
import os

def create_demo_images():
    """Create sample demo images for testing"""
    print("Creating demo images...")
    
    os.makedirs("demo_images", exist_ok=True)
    
    # Create a simple demo image with text
    demo_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    cv2.putText(demo_img, "HELMET DETECTION DEMO", (150, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(demo_img, "Add your own images to test", (180, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(demo_img, "Place images in demo_images folder", (160, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(demo_img, "Then run: python helmet_detection_demo.py", (120, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    cv2.imwrite("demo_images/demo_placeholder.jpg", demo_img)
    print("✅ Demo placeholder created!")

def run_demo():
    """Run helmet detection demo"""
    print("🚀 Starting Helmet Detection Demo")
    print("="*40)
    
    # Initialize detector
    detector = AdvancedHelmetDetector(confidence_threshold=0.3)
    
    # Check for demo images
    demo_dir = "demo_images"
    if not os.path.exists(demo_dir):
        create_demo_images()
        return
    
    image_files = [f for f in os.listdir(demo_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        create_demo_images()
        print("📁 No demo images found. Placeholder created.")
        print("Add your images to demo_images/ folder and run again.")
        return
    
    print(f"Found {len(image_files)} demo images")
    
    for image_file in image_files:
        if image_file == "demo_placeholder.jpg":
            continue
            
        image_path = os.path.join(demo_dir, image_file)
        print(f"\n🔍 Processing: {image_file}")
        
        # Detect helmets
        result = detector.detect_in_image(image_path, save_result=True)
        
        if result:
            print(f"   👥 Persons detected: {result['person_count']}")
            print(f"   ✅ With helmet: {result['helmet_count']}")
            print(f"   ❌ Without helmet: {result['no_helmet_count']}")
            
            if result['person_count'] > 0:
                compliance = (result['helmet_count'] / result['person_count']) * 100
                print(f"   📊 Compliance rate: {compliance:.1f}%")
        
        # Display result
        if result and 'annotated_image' in result:
            cv2.imshow(f'Helmet Detection - {image_file}', result['annotated_image'])
            print("   👀 Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # Print final statistics
    detector._print_statistics()

if __name__ == "__main__":
    run_demo()
