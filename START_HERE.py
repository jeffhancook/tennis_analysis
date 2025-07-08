#!/usr/bin/env python3
"""
🎾 TENNIS ANALYSIS - START HERE
===============================

This script does EVERYTHING automatically:
1. Installs required packages
2. Finds or copies a video file
3. Runs the complete tennis analysis
4. Shows you the results

Just run: python START_HERE.py
"""

import subprocess
import sys
import os

def install_if_missing(package):
    """Install package if it's not available"""
    try:
        __import__(package.replace('-', '_').split('>=')[0])
        return True
    except ImportError:
        print(f"📦 Installing {package}...")
        try:
            # Try pip3 first, then pip
            try:
                subprocess.check_call(['pip3', 'install', package], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
        except:
            return False

def main():
    print("🎾 TENNIS ANALYSIS - AUTO INSTALLER & RUNNER")
    print("=" * 60)
    print("This will automatically:")
    print("1. Install required packages (opencv, numpy, pandas)")
    print("2. Find or copy a tennis video")
    print("3. Run the complete analysis")
    print("4. Generate results with statistics")
    print("=" * 60)
    
    # Step 1: Install packages
    print("\n📦 STEP 1: Installing required packages...")
    packages = ['opencv-python', 'numpy', 'pandas']
    
    for package in packages:
        if install_if_missing(package):
            print(f"✅ {package}")
        else:
            print(f"❌ Failed to install {package}")
            print("Please run manually: pip install opencv-python numpy pandas")
            return
    
    # Step 2: Check for video
    print("\n📹 STEP 2: Looking for tennis video...")
    
    # Look for videos in current directory
    video_files = []
    for file in os.listdir('.'):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_files.append(file)
    
    video_to_use = None
    
    if video_files:
        video_to_use = video_files[0]
        print(f"✅ Found video: {video_to_use}")
    else:
        # Try to copy from parent directory
        possible_videos = [
            '../Input_Videos/input_video.mp4',
            '../input_video.mp4',
            '../../Input_Videos/input_video.mp4'
        ]
        
        for video_path in possible_videos:
            if os.path.exists(video_path):
                try:
                    import shutil
                    shutil.copy2(video_path, './tennis_video.mp4')
                    video_to_use = 'tennis_video.mp4'
                    print(f"✅ Copied video from {video_path}")
                    break
                except:
                    continue
    
    if not video_to_use:
        print("❌ No tennis video found!")
        print("\nPlease:")
        print("1. Place a tennis video file in this directory")
        print("2. Name it something like 'tennis.mp4'")
        print("3. Run this script again")
        return
    
    # Step 3: Run analysis
    print(f"\n🚀 STEP 3: Running analysis on {video_to_use}...")
    print("\nIMPORTANT: When windows open, you need to:")
    print("1. 🎯 COURT CALIBRATION: Click on 4 court corners, then press 'q'")
    print("2. 👥 PLAYER SELECTION: Click on 2 players, then press 'q'")
    print("3. ⏳ WAIT: Let the system process (may take a few minutes)")
    
    input("\n🎮 Press Enter when you're ready to start...")
    
    try:
        # Import and run the analysis
        print("\n" + "="*60)
        from main_standalone import TennisAnalysisSystem
        
        analyzer = TennisAnalysisSystem(video_to_use)
        success = analyzer.run_full_analysis()
        
        if success:
            print("\n🎉 SUCCESS! Analysis completed!")
            print(f"📁 Output video saved as: {video_to_use.replace('.mp4', '_analyzed.avi')}")
            print("📊 Statistics are shown above")
        else:
            print("\n❌ Analysis failed - check error messages above")
            
    except KeyboardInterrupt:
        print("\n👋 Analysis cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your video shows a tennis court")
        print("2. Try clicking more precisely on court corners")
        print("3. Ensure players are clearly visible")

if __name__ == "__main__":
    main()