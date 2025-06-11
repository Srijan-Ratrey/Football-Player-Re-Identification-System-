#!/usr/bin/env python3
"""
Environment Setup Script for Football Player Re-Identification Project

This script helps set up the environment and checks dependencies for the
player re-identification project.
"""

import os
import sys
import subprocess
import importlib
import pkg_resources
from typing import List, Tuple

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("  Required: Python 3.8 or higher")
        return False

def check_package(package_name: str, import_name: str = None) -> bool:
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        try:
            version = pkg_resources.get_distribution(package_name).version
            print(f"✓ {package_name} ({version})")
        except:
            print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} - not installed")
        return False

def install_package(package: str) -> bool:
    """Install a package using pip."""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")
        return False

def check_dependencies() -> Tuple[List[str], List[str]]:
    """Check all required dependencies."""
    print("Checking dependencies...")
    
    required_packages = [
        ("ultralytics", "ultralytics"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("Pillow", "PIL"),
        ("tqdm", "tqdm"),
        ("seaborn", "seaborn"),
        ("pandas", "pandas")
    ]
    
    missing = []
    installed = []
    
    for package_name, import_name in required_packages:
        if check_package(package_name, import_name):
            installed.append(package_name)
        else:
            missing.append(package_name)
    
    return installed, missing

def check_model_file() -> bool:
    """Check if the model file exists."""
    model_path = "best.pt"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Model file found: {model_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"✗ Model file not found: {model_path}")
        print("  Please download the model from the provided link")
        return False

def check_video_files() -> bool:
    """Check if video files exist."""
    video_files = [
        "data/15sec_input_720p.mp4",
        "data/broadcast.mp4",
        "data/tacticam.mp4"
    ]
    
    found_files = []
    missing_files = []
    
    for video_file in video_files:
        if os.path.exists(video_file):
            size_mb = os.path.getsize(video_file) / (1024 * 1024)
            print(f"✓ Video file found: {video_file} ({size_mb:.1f} MB)")
            found_files.append(video_file)
        else:
            print(f"✗ Video file not found: {video_file}")
            missing_files.append(video_file)
    
    return len(missing_files) == 0

def check_directories() -> bool:
    """Check and create necessary directories."""
    directories = ["data", "src", "results"]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✓ Directory exists: {directory}")
        else:
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"✓ Directory created: {directory}")
            except Exception as e:
                print(f"✗ Failed to create directory {directory}: {e}")
                return False
    
    return True

def check_gpu_support() -> bool:
    """Check if GPU support is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU support available: {gpu_name}")
            return True
        else:
            print("! GPU support not available - will use CPU")
            return False
    except ImportError:
        print("! Cannot check GPU support - PyTorch not installed")
        return False

def run_basic_test() -> bool:
    """Run a basic test to ensure everything works."""
    print("\nRunning basic functionality test...")
    
    try:
        # Test imports
        sys.path.append("src")
        from player_detector import PlayerDetector
        from feature_extractor import FeatureExtractor
        from tracker import PlayerTracker
        
        print("✓ All modules can be imported")
        
        # Test model loading if available
        if os.path.exists("best.pt"):
            try:
                detector = PlayerDetector("best.pt")
                print("✓ Model loads successfully")
            except Exception as e:
                print(f"✗ Model loading failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Basic test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("="*60)
    print("FOOTBALL PLAYER RE-IDENTIFICATION SETUP")
    print("="*60)
    
    success = True
    
    # Check Python version
    print("\n1. Checking Python version...")
    if not check_python_version():
        success = False
    
    # Check directories
    print("\n2. Checking directories...")
    if not check_directories():
        success = False
    
    # Check dependencies
    print("\n3. Checking dependencies...")
    installed, missing = check_dependencies()
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        response = input("Would you like to install missing packages? (y/n): ")
        
        if response.lower() == 'y':
            failed_installs = []
            for package in missing:
                if not install_package(package):
                    failed_installs.append(package)
            
            if failed_installs:
                print(f"Failed to install: {', '.join(failed_installs)}")
                success = False
        else:
            print("Please install missing packages manually:")
            print("pip install -r requirements.txt")
            success = False
    
    # Check model file
    print("\n4. Checking model file...")
    if not check_model_file():
        print("Please download the model file from:")
        print("https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view")
        success = False
    
    # Check video files
    print("\n5. Checking video files...")
    if not check_video_files():
        print("Please ensure video files are in the data/ directory")
        success = False
    
    # Check GPU support
    print("\n6. Checking GPU support...")
    check_gpu_support()
    
    # Run basic test
    if success:
        if not run_basic_test():
            success = False
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("✓ SETUP COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("You can now run the player re-identification system:")
        print()
        print("For single feed re-identification:")
        print("  python single_feed_reidentification.py")
        print()
        print("For cross-camera mapping:")
        print("  python cross_camera_mapping.py")
    else:
        print("✗ SETUP INCOMPLETE")
        print("="*60)
        print("Please resolve the issues above before running the system.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 