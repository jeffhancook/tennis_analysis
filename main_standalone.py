#!/usr/bin/env python3
"""
Standalone Tennis Analysis System
A complete, self-contained tennis video analysis system for casual users.

Usage:
    python main_standalone.py [video_path]
    
If no video path is provided, it will look for video files in the current directory.
"""

import cv2
import os
import sys
import argparse
import numpy as np
from typing import List, Dict, Tuple, Optional

# Import our standalone modules
from video_utils import read_video, save_video
from manual_court_detector import ManualCourtDetector
from simple_tracker import SimplePlayerTracker, SimpleBallTracker, draw_tracking_results
from tennis_stats import TennisStatsCalculator
import tennis_constants as const

class TennisAnalysisSystem:
    """
    Complete tennis analysis system for casual videos
    """
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.frames = []
        self.court_detector = ManualCourtDetector()
        self.player_tracker = SimplePlayerTracker()
        self.ball_tracker = SimpleBallTracker()
        self.stats_calculator = TennisStatsCalculator()
        
        # Analysis results
        self.court_corners = []
        self.player_detections = []
        self.ball_detections = []
        self.stats = {}
        
    def load_video(self) -> bool:
        """Load and validate video file"""
        print(f"📹 Loading video: {self.video_path}")
        
        if not os.path.exists(self.video_path):
            print(f"❌ Video file not found: {self.video_path}")
            return False
        
        try:
            self.frames = read_video(self.video_path)
            if not self.frames:
                print("❌ Could not read video frames")
                return False
                
            print(f"✅ Loaded {len(self.frames)} frames")
            print(f"📐 Frame size: {self.frames[0].shape[1]}x{self.frames[0].shape[0]}")
            
            # Calculate video info
            duration = len(self.frames) / const.DEFAULT_FPS
            print(f"⏱️  Estimated duration: {duration:.1f} seconds")
            
            return True
        except Exception as e:
            print(f"❌ Error loading video: {e}")
            return False
    
    def calibrate_court(self) -> bool:
        """Manual court calibration"""
        print(f"\n🎯 Court Calibration")
        print("You'll need to click on 4 visible court corners...")
        
        try:
            points = self.court_detector.get_manual_keypoints(self.frames[0])
            
            if len(points) >= 4:
                self.court_corners = points
                court_info = self.court_detector.create_court_reference_system(points)
                
                # Set reference for stats calculator
                self.stats_calculator.set_court_reference(points)
                
                print("✅ Court calibration successful!")
                return True
            else:
                print(f"⚠️ Only {len(points)} points selected, need at least 4")
                return False
                
        except Exception as e:
            print(f"❌ Court calibration failed: {e}")
            return False
    
    def track_players(self) -> bool:
        """Track players throughout the video"""
        print(f"\n👥 Player Tracking")
        
        try:
            # Manual player selection
            initial_players = self.player_tracker.manual_player_selection(self.frames[0])
            
            if not initial_players:
                print("⚠️ No players selected")
                return False
            
            # Sample frames for faster processing
            sample_frames = self.frames[::const.ANALYSIS_SAMPLE_RATE]
            print(f"📊 Processing {len(sample_frames)} sample frames (every {const.ANALYSIS_SAMPLE_RATE}th frame)")
            
            # Track players
            sample_detections = self.player_tracker.simple_tracking(sample_frames, initial_players)
            
            # Expand back to full video
            self.player_detections = self._expand_detections(sample_detections, len(self.frames))
            
            print("✅ Player tracking completed!")
            return True
            
        except Exception as e:
            print(f"❌ Player tracking failed: {e}")
            return False
    
    def track_ball(self) -> bool:
        """Track tennis ball (optional)"""
        print(f"\n🎾 Ball Tracking")
        
        try:
            # Sample frames for faster processing
            sample_frames = self.frames[::const.ANALYSIS_SAMPLE_RATE]
            
            sample_ball_detections = self.ball_tracker.track_ball_in_frames(sample_frames)
            
            # Expand back to full video
            self.ball_detections = self._expand_detections(sample_ball_detections, len(self.frames))
            
            # Count successful detections
            successful_detections = sum(1 for frame in self.ball_detections if frame)
            detection_rate = successful_detections / len(self.ball_detections)
            
            print(f"✅ Ball tracking completed! Detection rate: {detection_rate:.1%}")
            return True
            
        except Exception as e:
            print(f"⚠️ Ball tracking failed: {e}")
            print("📊 Continuing without ball analysis...")
            self.ball_detections = [{} for _ in self.frames]
            return False
    
    def calculate_statistics(self):
        """Calculate comprehensive tennis statistics"""
        print(f"\n📊 Calculating Statistics")
        
        try:
            # Check if we have ball detections
            has_ball_data = any(frame for frame in self.ball_detections)
            
            if has_ball_data:
                self.stats = self.stats_calculator.calculate_comprehensive_stats(
                    self.player_detections, self.ball_detections
                )
            else:
                self.stats = self.stats_calculator.calculate_comprehensive_stats(
                    self.player_detections
                )
            
            print("✅ Statistics calculated!")
            
        except Exception as e:
            print(f"❌ Statistics calculation failed: {e}")
            self.stats = {}
    
    def generate_output_video(self, output_path: str = None) -> str:
        """Generate annotated output video"""
        print(f"\n🎬 Generating Output Video")
        
        if output_path is None:
            # Create output filename in output_videos directory
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            os.makedirs("output_videos", exist_ok=True)
            output_path = f"output_videos/{base_name}_analyzed.avi"
        
        try:
            # Draw all tracking results
            output_frames = draw_tracking_results(
                self.frames, self.player_detections, self.ball_detections
            )
            
            # Add court corners
            for frame in output_frames:
                for i, point in enumerate(self.court_corners):
                    cv2.circle(frame, tuple(map(int, point)), 8, (0, 255, 0), -1)
                    cv2.putText(frame, f"{i+1}", tuple(map(int, point)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add statistics overlay
            output_frames = self._add_stats_overlay(output_frames)
            
            # Save video
            save_video(output_frames, output_path)
            
            print(f"✅ Output video saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Output video generation failed: {e}")
            return ""
    
    def run_full_analysis(self, output_path: str = None) -> bool:
        """Run complete tennis analysis pipeline"""
        print("🎾 STARTING TENNIS ANALYSIS")
        print("="*50)
        
        # Step 1: Load video
        if not self.load_video():
            return False
        
        # Step 2: Court calibration
        if not self.calibrate_court():
            print("⚠️ Proceeding without court calibration...")
        
        # Step 3: Player tracking
        if not self.track_players():
            print("❌ Cannot proceed without player tracking")
            return False
        
        # Step 4: Ball tracking (optional)
        self.track_ball()  # Continue even if this fails
        
        # Step 5: Calculate statistics
        self.calculate_statistics()
        
        # Step 6: Generate output
        output_file = self.generate_output_video(output_path)
        
        # Step 7: Display results
        if self.stats:
            self.stats_calculator.print_stats_summary(self.stats)
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        if output_file:
            print(f"📁 Output saved to: {output_file}")
        
        return True
    
    def _expand_detections(self, sample_detections: List[Dict], target_length: int) -> List[Dict]:
        """Expand sampled detections to full video length"""
        if not sample_detections:
            return [{} for _ in range(target_length)]
        
        expanded = []
        sample_ratio = len(sample_detections) / target_length
        
        for i in range(target_length):
            sample_index = min(int(i * sample_ratio), len(sample_detections) - 1)
            expanded.append(sample_detections[sample_index])
        
        return expanded
    
    def _add_stats_overlay(self, frames: List) -> List:
        """Add statistics overlay to frames"""
        if not self.stats:
            return frames
        
        output_frames = []
        session_info = self.stats.get('session_info', {})
        
        for frame_idx, frame in enumerate(frames):
            output_frame = frame.copy()
            
            # Create stats overlay background
            overlay_height = 150
            overlay = np.zeros((overlay_height, frame.shape[1], 3), dtype=np.uint8)
            overlay[:] = (0, 0, 0)  # Black background
            
            # Add text
            stats_text = [
                "🎾 Tennis Analysis Results",
                f"Frame: {frame_idx + 1}/{len(frames)}",
                f"Duration: {session_info.get('duration_minutes', 0):.1f} min",
                f"Detection Rate: {session_info.get('detection_rate', 0):.1%}"
            ]
            
            for i, text in enumerate(stats_text):
                y_pos = 25 + (i * 25)
                cv2.putText(overlay, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Combine with frame
            output_frame[:overlay_height] = cv2.addWeighted(
                output_frame[:overlay_height], 0.7, overlay, 0.3, 0
            )
            
            output_frames.append(output_frame)
        
        return output_frames

def find_video_files(directory: str = ".") -> List[str]:
    """Find video files in directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(directory, file))
    
    return video_files

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Standalone Tennis Video Analysis')
    parser.add_argument('video_path', nargs='?', help='Path to tennis video file')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--list-videos', action='store_true', help='List available video files')
    
    args = parser.parse_args()
    
    # List videos if requested
    if args.list_videos:
        videos = find_video_files()
        if videos:
            print("📹 Available video files:")
            for i, video in enumerate(videos, 1):
                print(f"  {i}. {video}")
        else:
            print("❌ No video files found in current directory")
        return
    
    # Determine video path
    video_path = args.video_path
    
    if not video_path:
        # Try to find video files
        videos = find_video_files()
        if not videos:
            print("❌ No video file specified and none found in current directory")
            print("Usage: python main_standalone.py <video_path>")
            print("   or: python main_standalone.py --list-videos")
            return
        elif len(videos) == 1:
            video_path = videos[0]
            print(f"📹 Using found video: {video_path}")
        else:
            print("📹 Multiple videos found:")
            for i, video in enumerate(videos, 1):
                print(f"  {i}. {video}")
            
            try:
                choice = int(input("Select video (number): ")) - 1
                if 0 <= choice < len(videos):
                    video_path = videos[choice]
                else:
                    print("❌ Invalid selection")
                    return
            except (ValueError, KeyboardInterrupt):
                print("\n❌ No video selected")
                return
    
    # Run analysis
    try:
        analyzer = TennisAnalysisSystem(video_path)
        success = analyzer.run_full_analysis(args.output)
        
        if success:
            print("\n🎉 Analysis completed successfully!")
        else:
            print("\n❌ Analysis failed")
            
    except KeyboardInterrupt:
        print("\n👋 Analysis cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()