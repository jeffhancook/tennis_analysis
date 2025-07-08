#!/usr/bin/env python3
"""
Tennis Analysis Setup Script
Automatically installs dependencies and sets up the system
"""

import subprocess
import sys
import os
import shutil

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def install_package(package_name):
    """Install a Python package"""
    return run_command(f"{sys.executable} -m pip install {package_name}", f"Installing {package_name}")

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7+ is required")
        return False
    
    print("✅ Python version is compatible")
    return True

def setup_directories():
    """Create necessary directories"""
    directories = ['input_videos', 'output_videos']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")
        else:
            print(f"📁 Directory already exists: {directory}")

def copy_sample_video():
    """Copy sample video if available"""
    sample_paths = [
        '../Input_Videos/input_video.mp4',
        '../input_video.mp4',
        '../../Input_Videos/input_video.mp4'
    ]
    
    for path in sample_paths:
        if os.path.exists(path):
            try:
                shutil.copy2(path, './sample_video.mp4')
                print(f"✅ Copied sample video from {path}")
                return True
            except Exception as e:
                print(f"⚠️ Could not copy video from {path}: {e}")
    
    print("📹 No sample video found - you'll need to provide your own video file")
    return False

def test_opencv():
    """Test if OpenCV is working"""
    try:
        import cv2
        print(f"✅ OpenCV installed successfully (version: {cv2.__version__})")
        return True
    except ImportError:
        print("❌ OpenCV import failed")
        return False

def main():
    print("🎾 Tennis Analysis System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Update pip first
    print("\n📦 Updating pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Updating pip")
    
    # Install required packages
    packages = [
        "opencv-python>=4.5.0",
        "numpy>=1.20.0", 
        "pandas>=1.3.0"
    ]
    
    print("\n📦 Installing required packages...")
    all_success = True
    
    for package in packages:
        if not install_package(package):
            all_success = False
    
    # Test OpenCV specifically
    print("\n🧪 Testing OpenCV installation...")
    if not test_opencv():
        print("🔧 Trying alternative OpenCV installation...")
        install_package("opencv-python-headless")
        if not test_opencv():
            print("❌ OpenCV installation failed. Please install manually:")
            print("   pip install opencv-python")
            all_success = False
    
    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Copy sample video if available
    print("\n📹 Looking for sample video...")
    copy_sample_video()
    
    # Final status
    print("\n" + "=" * 50)
    if all_success:
        print("🎉 Setup completed successfully!")
        print("\n🚀 Ready to run:")
        print("   python main_standalone.py")
        print("\n📹 To use your own video:")
        print("   1. Copy your video to this directory")
        print("   2. Run: python main_standalone.py your_video.mp4")
    else:
        print("⚠️ Setup completed with some issues")
        print("Please check the error messages above")
    
    print("=" * 50)

if __name__ == "__main__":
    main()