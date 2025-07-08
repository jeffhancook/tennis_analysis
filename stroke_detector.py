#!/usr/bin/env python3
"""
Tennis Stroke Detection System
Ball-free approach using audio analysis and pose estimation
"""

import cv2
import numpy as np
import librosa
import mediapipe as mp
import soundfile as sf
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import subprocess
import os
import tempfile

# Import existing utilities
from video_utils import read_video, save_video
from bbox_utils import get_center_of_bbox, measure_distance


@dataclass
class StrokeEvent:
    """Represents a detected tennis stroke"""
    frame_number: int
    timestamp: float
    stroke_type: str  # 'forehand', 'backhand', 'serve', 'volley'
    player_id: int
    confidence: float
    position: Tuple[int, int]  # Player position on court
    power_estimate: float  # Estimated stroke power (0-100)


class AudioStrokeDetector:
    """
    Detects tennis strokes using audio analysis
    Ball-racket contact creates distinctive transient sounds
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.min_time_between_hits = 0.3  # Minimum 300ms between strokes
        
    def extract_audio_from_video(self, video_path: str) -> Tuple[np.ndarray, int]:
        """Extract audio track from video file"""
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        try:
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg', '-i', video_path,
                '-ac', '1',  # Mono
                '-ar', str(self.sample_rate),  # Sample rate
                '-y',  # Overwrite
                temp_audio_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Load audio
            audio, sr = librosa.load(temp_audio_path, sr=self.sample_rate)
            return audio, sr
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
    
    def detect_ball_hits(self, audio: np.ndarray, sr: int) -> List[float]:
        """
        Detect ball-racket contact moments using onset detection
        Returns list of timestamps in seconds
        """
        # Apply high-pass filter to emphasize ball hit frequencies
        # Tennis ball hits are typically in 1-4 kHz range
        audio_filtered = librosa.effects.preemphasis(audio)
        
        # Detect onsets (sudden energy bursts)
        onset_frames = librosa.onset.onset_detect(
            y=audio_filtered,
            sr=sr,
            pre_max=20,  # 20ms pre-window
            post_max=20,  # 20ms post-window
            pre_avg=100,  # 100ms for background noise
            post_avg=100,
            delta=0.3,  # Sensitivity threshold
            wait=int(sr * self.min_time_between_hits)  # Minimum frames between hits
        )
        
        # Convert frames to time
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Additional filtering based on spectral characteristics
        filtered_times = []
        for i, onset_frame in enumerate(onset_frames):
            # Get audio segment around onset
            start = max(0, onset_frame - sr // 100)  # 10ms before
            end = min(len(audio), onset_frame + sr // 100)  # 10ms after
            segment = audio[start:end]
            
            # Check if it has tennis ball hit characteristics
            if self._is_tennis_hit(segment, sr):
                filtered_times.append(onset_times[i])
        
        return filtered_times
    
    def _is_tennis_hit(self, audio_segment: np.ndarray, sr: int) -> bool:
        """Check if audio segment has tennis ball hit characteristics"""
        if len(audio_segment) < sr // 100:  # Too short
            return False
        
        # Compute spectral centroid (tennis hits have high frequency content)
        centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)
        mean_centroid = np.mean(centroid)
        
        # Tennis hits typically have centroid > 2000 Hz
        if mean_centroid < 2000:
            return False
        
        # Check for sharp attack (sudden energy increase)
        energy = librosa.feature.rms(y=audio_segment)[0]
        if len(energy) > 1:
            attack_ratio = np.max(energy) / (np.mean(energy) + 1e-6)
            if attack_ratio < 2.0:  # Not sharp enough
                return False
        
        return True
    
    def process_video(self, video_path: str, fps: float = 24.0) -> List[int]:
        """
        Process video and return frame numbers where strokes occur
        """
        print("🎵 Extracting audio from video...")
        audio, sr = self.extract_audio_from_video(video_path)
        
        print("🎾 Detecting ball hits from audio...")
        hit_times = self.detect_ball_hits(audio, sr)
        
        # Convert times to frame numbers
        stroke_frames = [int(time * fps) for time in hit_times]
        
        print(f"✅ Detected {len(stroke_frames)} potential strokes from audio")
        return stroke_frames


class PoseStrokeClassifier:
    """
    Classifies tennis strokes using pose estimation
    Identifies forehand, backhand, serve, and volley
    """
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Tracking state
        self.pose_history = deque(maxlen=10)  # Keep last 10 poses
        
    def analyze_player_pose(self, frame: np.ndarray, player_bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Analyze player pose within bounding box
        Returns pose landmarks and derived features
        """
        x1, y1, x2, y2 = [int(coord) for coord in player_bbox]
        
        # Extract player region
        player_region = frame[y1:y2, x1:x2]
        if player_region.size == 0:
            return None
        
        # Run pose estimation
        results = self.pose.process(cv2.cvtColor(player_region, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            return None
        
        # Extract key landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Calculate features for stroke classification
        features = {
            'landmarks': landmarks,
            'shoulder_angle': self._calculate_shoulder_angle(landmarks),
            'elbow_angle': self._calculate_elbow_angle(landmarks, 'right'),
            'wrist_height': landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y,
            'hip_rotation': self._calculate_hip_rotation(landmarks),
            'arm_extension': self._calculate_arm_extension(landmarks, 'right')
        }
        
        return features
    
    def classify_stroke(self, pose_features: Dict, motion_vector: Optional[Tuple[float, float]] = None) -> Tuple[str, float]:
        """
        Classify stroke type based on pose features
        Returns (stroke_type, confidence)
        """
        if not pose_features:
            return 'unknown', 0.0
        
        landmarks = pose_features['landmarks']
        
        # Get key positions
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        
        # Serve detection (high wrist position)
        if right_wrist.y < right_shoulder.y - 0.3:  # Wrist well above shoulder
            return 'serve', 0.85
        
        # Determine which hand is dominant based on arm extension
        right_extended = pose_features['arm_extension'] > 0.7
        left_extended = self._calculate_arm_extension(landmarks, 'left') > 0.7
        
        # Volley detection (both arms forward, close to net)
        if right_wrist.y > 0.3 and right_wrist.y < 0.6:  # Mid-height
            if right_extended or left_extended:
                return 'volley', 0.75
        
        # Forehand vs Backhand
        if right_extended:
            # Check body rotation for forehand/backhand
            hip_rotation = pose_features['hip_rotation']
            if hip_rotation > 0.2:  # Body rotated to the right
                return 'forehand', 0.80
            else:
                return 'backhand', 0.80
        
        return 'unknown', 0.3
    
    def estimate_stroke_power(self, current_pose: Dict, previous_poses: List[Dict]) -> float:
        """
        Estimate stroke power based on motion speed and arm extension
        Returns power estimate 0-100
        """
        if not current_pose or not previous_poses:
            return 0.0
        
        # Calculate wrist velocity
        if len(previous_poses) >= 2:
            current_wrist = current_pose['landmarks'][self.mp_pose.PoseLandmark.RIGHT_WRIST]
            prev_wrist = previous_poses[-1]['landmarks'][self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            # Calculate movement distance
            wrist_movement = np.sqrt(
                (current_wrist.x - prev_wrist.x)**2 + 
                (current_wrist.y - prev_wrist.y)**2
            )
            
            # Combine with arm extension
            arm_extension = current_pose['arm_extension']
            
            # Power estimate (0-100 scale)
            power = min(100, wrist_movement * 500 * arm_extension)
            return power
        
        return 0.0
    
    def _calculate_shoulder_angle(self, landmarks) -> float:
        """Calculate angle between shoulders (body rotation indicator)"""
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        
        angle = np.arctan2(
            right_shoulder.y - left_shoulder.y,
            right_shoulder.x - left_shoulder.x
        )
        return angle
    
    def _calculate_elbow_angle(self, landmarks, side: str) -> float:
        """Calculate elbow angle (arm bend)"""
        if side == 'right':
            shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        else:
            shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        
        # Calculate angle at elbow
        v1 = np.array([shoulder.x - elbow.x, shoulder.y - elbow.y])
        v2 = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
        
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
        return angle
    
    def _calculate_hip_rotation(self, landmarks) -> float:
        """Calculate hip rotation (body orientation)"""
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        
        rotation = right_hip.z - left_hip.z  # Z-axis indicates depth
        return rotation
    
    def _calculate_arm_extension(self, landmarks, side: str) -> float:
        """Calculate how extended the arm is (0=bent, 1=fully extended)"""
        elbow_angle = self._calculate_elbow_angle(landmarks, side)
        extension = (elbow_angle / np.pi)  # Normalize to 0-1
        return extension


class TennisStrokeAnalyzer:
    """
    Main analyzer combining audio and pose analysis for stroke detection
    Works without ball detection
    """
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.audio_detector = AudioStrokeDetector()
        self.pose_classifier = PoseStrokeClassifier()
        
        # Results storage
        self.stroke_events = []
        self.player_detections = []
        
    def analyze_strokes(self, player_detections: List[Dict[int, Tuple[int, int, int, int]]]) -> List[StrokeEvent]:
        """
        Main analysis pipeline combining audio and pose detection
        """
        print("\n🎾 TENNIS STROKE ANALYSIS (Ball-Free Approach)")
        print("="*50)
        
        # Step 1: Get stroke timings from audio
        print("\n📍 Step 1: Audio-based stroke detection")
        stroke_frames = self.audio_detector.process_video(self.video_path)
        
        if not stroke_frames:
            print("⚠️ No strokes detected from audio")
            return []
        
        # Step 2: Load video for pose analysis
        print("\n📍 Step 2: Loading video for pose analysis")
        frames = read_video(self.video_path)
        fps = 24.0  # Assume 24 fps (can be detected from video)
        
        # Step 3: Analyze pose around each stroke event
        print("\n📍 Step 3: Analyzing player poses at stroke moments")
        stroke_events = []
        
        for stroke_frame in stroke_frames:
            # Analyze frames around the stroke (±5 frames)
            start_frame = max(0, stroke_frame - 5)
            end_frame = min(len(frames), stroke_frame + 5)
            
            best_event = None
            best_confidence = 0.0
            
            for frame_idx in range(start_frame, end_frame):
                if frame_idx >= len(player_detections):
                    continue
                
                frame = frames[frame_idx]
                players = player_detections[frame_idx]
                
                # Analyze each player
                for player_id, bbox in players.items():
                    # Get pose features
                    pose_features = self.pose_classifier.analyze_player_pose(frame, bbox)
                    
                    if pose_features:
                        # Classify stroke
                        stroke_type, confidence = self.pose_classifier.classify_stroke(pose_features)
                        
                        # Estimate power
                        power = self.pose_classifier.estimate_stroke_power(
                            pose_features, 
                            list(self.pose_classifier.pose_history)
                        )
                        
                        # Create stroke event
                        if confidence > best_confidence:
                            best_event = StrokeEvent(
                                frame_number=frame_idx,
                                timestamp=frame_idx / fps,
                                stroke_type=stroke_type,
                                player_id=player_id,
                                confidence=confidence,
                                position=get_center_of_bbox(bbox),
                                power_estimate=power
                            )
                            best_confidence = confidence
                        
                        # Update pose history
                        self.pose_classifier.pose_history.append(pose_features)
            
            if best_event and best_event.stroke_type != 'unknown':
                stroke_events.append(best_event)
        
        print(f"\n✅ Detected {len(stroke_events)} strokes with classification")
        self.stroke_events = stroke_events
        return stroke_events
    
    def generate_analysis_video(self, frames: List[np.ndarray], player_detections: List[Dict], 
                              stroke_events: List[StrokeEvent], output_path: str):
        """Generate video with stroke analysis overlay"""
        output_frames = []
        
        # Create event lookup by frame
        events_by_frame = {}
        for event in stroke_events:
            events_by_frame[event.frame_number] = event
        
        for i, frame in enumerate(frames):
            output_frame = frame.copy()
            
            # Draw players
            if i < len(player_detections):
                for player_id, bbox in player_detections[i].items():
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    color = (0, 255, 0) if player_id == 1 else (255, 0, 0)
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(output_frame, f"Player {player_id}", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw stroke event if present
            if i in events_by_frame:
                event = events_by_frame[i]
                
                # Highlight player with stroke
                if i < len(player_detections) and event.player_id in player_detections[i]:
                    bbox = player_detections[i][event.player_id]
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    
                    # Draw thick border for stroke
                    cv2.rectangle(output_frame, (x1-5, y1-5), (x2+5, y2+5), (0, 255, 255), 4)
                    
                    # Add stroke info
                    info_text = f"{event.stroke_type.upper()} (Power: {event.power_estimate:.0f}%)"
                    cv2.putText(output_frame, info_text, (x1, y1-25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Add timestamp
            timestamp = i / 24.0  # Assuming 24 fps
            cv2.putText(output_frame, f"Time: {timestamp:.2f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            output_frames.append(output_frame)
        
        save_video(output_frames, output_path)
        print(f"✅ Analysis video saved to: {output_path}")
    
    def print_analysis_summary(self):
        """Print summary of detected strokes"""
        if not self.stroke_events:
            print("No strokes detected")
            return
        
        print("\n" + "="*60)
        print("🎾 STROKE ANALYSIS SUMMARY")
        print("="*60)
        
        # Count by type
        stroke_counts = {}
        total_power = 0
        
        for event in self.stroke_events:
            stroke_counts[event.stroke_type] = stroke_counts.get(event.stroke_type, 0) + 1
            total_power += event.power_estimate
        
        print(f"\n📊 Total Strokes: {len(self.stroke_events)}")
        print("\n🏓 Stroke Breakdown:")
        for stroke_type, count in stroke_counts.items():
            percentage = (count / len(self.stroke_events)) * 100
            print(f"  {stroke_type.capitalize()}: {count} ({percentage:.1f}%)")
        
        print(f"\n💪 Average Power: {total_power / len(self.stroke_events):.1f}%")
        
        # Player breakdown
        player_strokes = {}
        for event in self.stroke_events:
            player_strokes[event.player_id] = player_strokes.get(event.player_id, 0) + 1
        
        print("\n👥 By Player:")
        for player_id, count in player_strokes.items():
            print(f"  Player {player_id}: {count} strokes")
        
        print("\n⏱️ Timing Analysis:")
        if len(self.stroke_events) > 1:
            intervals = []
            for i in range(1, len(self.stroke_events)):
                interval = self.stroke_events[i].timestamp - self.stroke_events[i-1].timestamp
                intervals.append(interval)
            
            avg_interval = np.mean(intervals)
            print(f"  Average time between strokes: {avg_interval:.2f}s")
            print(f"  Estimated rally pace: {60/avg_interval:.1f} shots/minute")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tennis Stroke Detection (Ball-Free)')
    parser.add_argument('video_path', help='Path to tennis video')
    parser.add_argument('--player_detections', help='Path to player detections pickle', 
                       default='stubs/player_detections.pkl')
    parser.add_argument('--output', help='Output video path', 
                       default='output_videos/stroke_analysis.avi')
    
    args = parser.parse_args()
    
    # Load player detections (from your existing tracker)
    import pickle
    with open(args.player_detections, 'rb') as f:
        player_detections = pickle.load(f)
    
    # Run analysis
    analyzer = TennisStrokeAnalyzer(args.video_path)
    stroke_events = analyzer.analyze_strokes(player_detections)
    
    # Generate output video
    frames = read_video(args.video_path)
    analyzer.generate_analysis_video(frames, player_detections, stroke_events, args.output)
    
    # Print summary
    analyzer.print_analysis_summary()


if __name__ == "__main__":
    main()