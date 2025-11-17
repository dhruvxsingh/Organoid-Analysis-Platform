# demo.py
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dilipredict.models import DILIPredict
from dilipredict.image_loader import ImageLoader
from dilipredict.pipelines import DILIPredict as DILIPipeline

def check_environment():
    """Check if all required components are installed"""
    print("=" * 60)
    print("DILITracer Environment Check")
    print("=" * 60)
    
    # Check PyTorch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    
    # Check if model can be initialized
    try:
        model = DILIPredict()
        print("✓ DILITracer model initialized successfully")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        return False
    
    # Check image loader
    try:
        img_loader = ImageLoader()
        print("✓ Image loader initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing image loader: {e}")
        return False
    
    return True

def create_sample_data():
    """Create sample data structure if it doesn't exist"""
    sample_data_path = Path("data/sample_data")
    
    if not sample_data_path.exists():
        print("\nCreating sample data directory structure...")
        sample_data_path.mkdir(parents=True, exist_ok=True)
        
        # Create day folders
        for day in ["D00", "D01", "D02", "D03"]:
            day_path = sample_data_path / day
            day_path.mkdir(exist_ok=True)
            
            # Create dummy images for testing (3 z-stack images per day)
            for z in range(3):
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                img_path = day_path / f"sample_z{z:03d}.tiff"
                img.save(img_path)
        
        print("✓ Sample data structure created")
    
    # Check existing data
    print("\n" + "=" * 60)
    print("Sample Data Structure:")
    print("=" * 60)
    
    for day in ["D00", "D01", "D02", "D03"]:
        day_path = sample_data_path / day
        if day_path.exists():
            files = list(day_path.glob("*.tiff"))
            print(f"  {day}: {len(files)} images")
            if files:
                print(f"    Example: {files[0].name}")

def test_inference():
    """Test the inference pipeline"""
    print("\n" + "=" * 60)
    print("Testing Inference Pipeline")
    print("=" * 60)
    
    try:
        # Initialize components
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DILIPredict()
        img_loader = ImageLoader()
        pipeline = DILIPipeline(img_loader, model, device)
        
        print(f"✓ Pipeline initialized on {device}")
        
        # Prepare sample input
        sample_data_path = Path("data/sample_data")
        img_files = []
        
        for day in ["D00", "D01", "D02", "D03"]:
            day_path = sample_data_path / day
            if day_path.exists():
                day_images = sorted(list(day_path.glob("*.tiff")))[:3]  # Take first 3 images
                img_files.append([str(img) for img in day_images])
        
        if img_files and all(len(day) > 0 for day in img_files):
            print(f"✓ Found images for {len(img_files)} days")
            print(f"  Images per day: {[len(day) for day in img_files]}")
            
            # Run inference
            print("\nRunning inference...")
            label, probabilities = pipeline(img_files)
            
            # Map labels to DILI categories
            label_map = {0: "No-DILI", 1: "Less-DILI", 2: "Most-DILI"}
            
            print("\n" + "=" * 60)
            print("Inference Results:")
            print("=" * 60)
            print(f"  Predicted Label: {label_map.get(label, 'Unknown')}")
            print(f"  Probabilities:")
            print(f"    No-DILI:   {probabilities[0]:.4f}")
            print(f"    Less-DILI: {probabilities[1]:.4f}")
            print(f"    Most-DILI: {probabilities[2]:.4f}")
            
            return True
        else:
            print("✗ No sample images found. Please add images to data/sample_data/")
            return False
            
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demo function"""
    print("\n" * 2)
    print("*" * 60)
    print(" " * 15 + "DILITracer Demo System")
    print("*" * 60)
    
    # Check environment
    if not check_environment():
        print("\n⚠ Environment check failed. Please install required dependencies.")
        return
    
    # Create/check sample data
    create_sample_data()
    
    # Test inference
    if test_inference():
        print("\n" + "=" * 60)
        print("✅ DILITracer is ready for use!")
        print("=" * 60)
        print("\nTo use with your own images:")
        print("1. Organize images in folders: D00/, D01/, D02/, D03/")
        print("2. Each folder should contain z-stack images (.tiff format)")
        print("3. Run inference using the pipeline")
        
        print("\nExample usage:")
        print("-" * 40)
        print("""
from dilipredict.models import DILIPredict
from dilipredict.image_loader import ImageLoader
from dilipredict.pipelines import DILIPredict as DILIPipeline

# Initialize
model = DILIPredict()
img_loader = ImageLoader()
pipeline = DILIPipeline(img_loader, model)

# Prepare image paths
img_files = [
    ['D00/img1.tiff', 'D00/img2.tiff'],  # Day 0
    ['D01/img1.tiff', 'D01/img2.tiff'],  # Day 1
    ['D02/img1.tiff', 'D02/img2.tiff'],  # Day 2
    ['D03/img1.tiff', 'D03/img2.tiff'],  # Day 3
]

# Run inference
label, probabilities = pipeline(img_files)
""")
        print("-" * 40)
    else:
        print("\n⚠ Inference test failed. Please check the error messages above.")

if __name__ == "__main__":
    main()