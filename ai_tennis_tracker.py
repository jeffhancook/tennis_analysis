#!/usr/bin/env python3
"""
AI-Powered Tennis Tracker
- Advanced player tracking that handles all poses (including back-turned)
- Smart ball detection for small green pixelated moving objects
- Motion prediction and tracking recovery
"""

import cv2
import numpy as np
import os
import sys
from typing import List, Dict, Tuple, Optional
from collections import deque
import math

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ YOLO not available. Install with: pip3 install ultralytics")
    YOLO_AVAILABLE = False

from video_utils import read_video, save_video
from bbox_utils import get_center_of_bbox, measure_distance

class AIPlayerTracker:
    """
    Advanced AI player tracker that handles occlusions and pose variations
    """
    
    def __init__(self, model_path: str):
        if not YOLO_AVAILABLE:
            raise ValueError("YOLO required")
        
        self.model = YOLO(model_path)
        
        # Tracking state
        self.players = {
            1: {'bbox': None, 'velocity': (0, 0), 'lost_frames': 0, 'template': None, 'history': deque(maxlen=30)},
            2: {'bbox': None, 'velocity': (0, 0), 'lost_frames': 0, 'template': None, 'history': deque(maxlen=30)}
        }
        self.initialized = False
        self.frame_count = 0
        
        # Tracking parameters
        self.MAX_LOST_FRAMES = 30  # How long to predict before giving up
        self.MAX_MOVEMENT = 200    # Max reasonable movement per frame
        self.MIN_CONFIDENCE = 0.3  # Lower confidence for difficult poses
        
    def track_frame(self, frame: np.ndarray) -> Dict[int, List[float]]:
        """Advanced tracking with motion prediction and recovery"""
        self.frame_count += 1
        
        # Detect all people with lower confidence to catch difficult poses
        results = self.model(frame, conf=self.MIN_CONFIDENCE, verbose=False)
        
        detected_people = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    if box.cls == 0:  # Person class
                        bbox = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        
                        # More flexible size filter
                        if area > 1000:  # Lower threshold
                            detected_people.append({
                                'bbox': bbox,
                                'confidence': conf,
                                'area': area
                            })
        
        # Sort by confidence and area
        detected_people.sort(key=lambda x: x['confidence'] * x['area'], reverse=True)
        
        if not self.initialized:
            # Initialize with first 2 detections
            if len(detected_people) >= 2:
                self._initialize_players(frame, detected_people[:2])
                self.initialized = True
            return self._get_current_positions()
        
        # Match detections to existing players
        self._match_detections_to_players(frame, detected_people)
        
        # Predict positions for lost players
        self._predict_lost_players(frame)
        
        # Update templates and history
        self._update_tracking_data(frame)
        
        return self._get_current_positions()
    
    def _initialize_players(self, frame: np.ndarray, detections: List[Dict]):
        """Initialize player tracking"""
        bbox1 = detections[0]['bbox']
        bbox2 = detections[1]['bbox']
        
        # Assign based on x position (left = player 1, right = player 2)
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        
        if center1_x < center2_x:
            self.players[1]['bbox'] = bbox1
            self.players[2]['bbox'] = bbox2
        else:
            self.players[1]['bbox'] = bbox2
            self.players[2]['bbox'] = bbox1
        
        # Extract templates
        for player_id in [1, 2]:
            bbox = self.players[player_id]['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            template = frame[y1:y2, x1:x2]
            if template.size > 0:
                self.players[player_id]['template'] = template
            
            # Initialize history
            center = get_center_of_bbox(bbox)
            self.players[player_id]['history'].append(center)
        
        print("🎯 Players initialized with AI tracking")
    
    def _match_detections_to_players(self, frame: np.ndarray, detections: List[Dict]):
        """Match new detections to existing players"""
        used_detections = set()
        
        for player_id in [1, 2]:
            best_match = None
            best_score = 0
            
            if self.players[player_id]['bbox'] is None:
                continue
            
            last_center = get_center_of_bbox(self.players[player_id]['bbox'])
            
            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue
                
                det_center = get_center_of_bbox(detection['bbox'])
                distance = measure_distance(last_center, det_center)
                
                # Score based on distance and confidence
                if distance < self.MAX_MOVEMENT:
                    score = detection['confidence'] * (1 - distance / self.MAX_MOVEMENT)
                    
                    if score > best_score:
                        best_score = score
                        best_match = i
            
            if best_match is not None:
                # Found a good match
                self.players[player_id]['bbox'] = detections[best_match]['bbox']
                self.players[player_id]['lost_frames'] = 0
                used_detections.add(best_match)
                
                # Update velocity
                current_center = get_center_of_bbox(detections[best_match]['bbox'])
                if self.players[player_id]['history']:
                    last_center = self.players[player_id]['history'][-1]
                    velocity = (current_center[0] - last_center[0], 
                              current_center[1] - last_center[1])
                    self.players[player_id]['velocity'] = velocity
            else:
                # No match found - try template matching
                template_match = self._template_match_recovery(frame, player_id)
                if template_match:
                    self.players[player_id]['bbox'] = template_match
                    self.players[player_id]['lost_frames'] = 0
                else:
                    self.players[player_id]['lost_frames'] += 1
    
    def _template_match_recovery(self, frame: np.ndarray, player_id: int) -> Optional[List[float]]:
        """Try to recover lost player using template matching"""
        if self.players[player_id]['template'] is None:
            return None
        
        template = self.players[player_id]['template']
        last_bbox = self.players[player_id]['bbox']
        
        # Define search area around last known position
        search_margin = 150
        x1 = max(0, int(last_bbox[0] - search_margin))
        y1 = max(0, int(last_bbox[1] - search_margin))
        x2 = min(frame.shape[1], int(last_bbox[2] + search_margin))
        y2 = min(frame.shape[0], int(last_bbox[3] + search_margin))
        
        search_area = frame[y1:y2, x1:x2]
        if search_area.size == 0:
            return None
        
        # Template matching
        try:
            result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.5:  # Good template match
                # Convert back to frame coordinates
                match_x = x1 + max_loc[0]
                match_y = y1 + max_loc[1]
                
                new_bbox = [
                    match_x,
                    match_y,
                    match_x + template.shape[1],
                    match_y + template.shape[0]
                ]
                
                return new_bbox
        except:
            pass
        
        return None
    
    def _predict_lost_players(self, frame: np.ndarray):
        """Predict position of lost players using motion"""
        for player_id in [1, 2]:
            if self.players[player_id]['lost_frames'] > 0 and self.players[player_id]['lost_frames'] <= self.MAX_LOST_FRAMES:
                # Predict position based on velocity
                if self.players[player_id]['bbox'] and self.players[player_id]['velocity']:
                    bbox = self.players[player_id]['bbox']
                    velocity = self.players[player_id]['velocity']
                    
                    # Apply velocity with dampening
                    damping = 0.9 ** self.players[player_id]['lost_frames']
                    pred_velocity = (velocity[0] * damping, velocity[1] * damping)
                    
                    predicted_bbox = [
                        bbox[0] + pred_velocity[0],
                        bbox[1] + pred_velocity[1],
                        bbox[2] + pred_velocity[0],
                        bbox[3] + pred_velocity[1]
                    ]
                    
                    # Keep prediction within frame bounds
                    predicted_bbox[0] = max(0, predicted_bbox[0])
                    predicted_bbox[1] = max(0, predicted_bbox[1])
                    predicted_bbox[2] = min(frame.shape[1], predicted_bbox[2])
                    predicted_bbox[3] = min(frame.shape[0], predicted_bbox[3])
                    
                    self.players[player_id]['bbox'] = predicted_bbox
    
    def _update_tracking_data(self, frame: np.ndarray):
        """Update templates and position history"""
        for player_id in [1, 2]:
            if self.players[player_id]['bbox'] and self.players[player_id]['lost_frames'] == 0:
                # Update template occasionally with good detections
                if self.frame_count % 30 == 0:  # Every 30 frames
                    bbox = self.players[player_id]['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    template = frame[y1:y2, x1:x2]
                    if template.size > 0:
                        self.players[player_id]['template'] = template
                
                # Update position history
                center = get_center_of_bbox(self.players[player_id]['bbox'])
                self.players[player_id]['history'].append(center)
    
    def _get_current_positions(self) -> Dict[int, List[float]]:
        """Get current player positions"""
        positions = {}
        for player_id in [1, 2]:
            if (self.players[player_id]['bbox'] and 
                self.players[player_id]['lost_frames'] <= self.MAX_LOST_FRAMES):
                positions[player_id] = self.players[player_id]['bbox']
        return positions

class SmartBallDetector:
    """
    Smart ball detector that finds small green pixelated moving objects
    """
    
    def __init__(self):
        self.ball_history = deque(maxlen=20)  # Track ball movement
        self.motion_tracker = None
        
    def detect_ball(self, frame: np.ndarray, player_bboxes: List) -> Optional[Tuple[int, int, int, int]]:
        """Detect small moving green pixelated ball"""
        
        # Convert to different color spaces for better detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Create mask for bright yellow-green (tennis ball color)
        # Tennis balls appear as bright, somewhat saturated yellow-green
        lower_ball = np.array([25, 100, 150])  # More flexible
        upper_ball = np.array([45, 255, 255])
        ball_mask = cv2.inRange(hsv, lower_ball, upper_ball)
        
        # Also try LAB color space for brightness
        # Tennis balls are bright in L channel
        l_channel = lab[:, :, 0]
        bright_mask = cv2.threshold(l_channel, 200, 255, cv2.THRESH_BINARY)[1]
        
        # Combine masks
        combined_mask = cv2.bitwise_and(ball_mask, bright_mask)
        
        # Remove UI areas
        h, w = combined_mask.shape
        combined_mask[:int(h*0.12), :] = 0    # Top
        combined_mask[int(h*0.88):, :] = 0    # Bottom
        combined_mask[:, :int(w*0.05)] = 0    # Left
        combined_mask[:, int(w*0.95):] = 0    # Right
        
        # Remove player areas (with larger margin)
        for bbox in player_bboxes:
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                margin = 60  # Larger margin
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                combined_mask[y1:y2, x1:x2] = 0
        
        # Clean up mask - focus on small objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Tennis ball is SMALL - typically 10-80 pixels on broadcast video
            if 8 <= area <= 80:
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w//2, y + h//2)
                
                # Check if roughly circular (tennis balls are round)
                if w > 0 and h > 0:
                    aspect_ratio = min(w, h) / max(w, h)
                    if aspect_ratio < 0.6:  # Too elongated
                        continue
                
                # Check average brightness in the region
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    avg_brightness = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                    if avg_brightness < 100:  # Too dark
                        continue
                
                candidates.append({
                    'bbox': (x, y, x+w, y+h),
                    'center': center,
                    'area': area,
                    'brightness': avg_brightness
                })
        
        if not candidates:
            return None
        
        # Select best candidate based on motion and history
        best_candidate = self._select_best_ball_candidate(candidates)
        
        if best_candidate:
            self.ball_history.append(best_candidate['center'])
            return best_candidate['bbox']
        
        return None
    
    def _select_best_ball_candidate(self, candidates: List[Dict]) -> Optional[Dict]:
        """Select the best ball candidate based on motion patterns"""
        
        if len(candidates) == 1:
            return candidates[0]
        
        if not self.ball_history:
            # No history, pick brightest small object
            return max(candidates, key=lambda c: c['brightness'])
        
        # Score candidates based on realistic tennis ball motion
        scored_candidates = []
        
        for candidate in candidates:
            score = 0
            center = candidate['center']
            
            # Check motion consistency
            if len(self.ball_history) >= 2:
                # Calculate expected position based on recent motion
                recent_positions = list(self.ball_history)[-3:]
                
                if len(recent_positions) >= 2:
                    # Calculate velocity
                    velocity = (
                        recent_positions[-1][0] - recent_positions[-2][0],
                        recent_positions[-1][1] - recent_positions[-2][1]
                    )
                    
                    # Predict next position
                    predicted_pos = (
                        recent_positions[-1][0] + velocity[0],
                        recent_positions[-1][1] + velocity[1]
                    )
                    
                    # Score based on how close candidate is to prediction
                    distance_to_prediction = measure_distance(center, predicted_pos)
                    
                    # Tennis ball shouldn't move too fast or too slow
                    speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
                    if 2 <= speed <= 100:  # Reasonable ball speed range
                        score += 50
                        
                        # Bonus for being close to prediction
                        if distance_to_prediction < 30:
                            score += 30
                        elif distance_to_prediction < 60:
                            score += 15
            
            # Distance from last known position
            last_pos = self.ball_history[-1]
            distance = measure_distance(center, last_pos)
            
            # Reasonable movement (ball can't teleport)
            if distance < 80:
                score += 20
                if distance < 40:
                    score += 20
            
            # Prefer smaller, brighter objects
            score += (100 - candidate['area'])  # Smaller is better
            score += candidate['brightness'] / 10  # Brighter is better
            
            scored_candidates.append((candidate, score))
        
        if scored_candidates:
            # Return candidate with highest score
            best = max(scored_candidates, key=lambda x: x[1])
            return best[0]
        
        return None

class AITennisAnalyzer:
    """
    AI-powered tennis analyzer with advanced tracking
    """
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.frames = []
        
        # Model paths
        self.player_model = self._find_model("yolov8x.pt")
        
        # Components
        self.player_tracker = None
        self.ball_detector = SmartBallDetector()
        
        # Results
        self.player_detections = []
        self.ball_detections = []
    
    def _find_model(self, model_name: str) -> str:
        """Find model file"""
        paths = [".", "..", "../models", "models"]
        for path in paths:
            full_path = os.path.join(path, model_name)
            if os.path.exists(full_path):
                return full_path
        return model_name
    
    def load_video(self) -> bool:
        """Load video"""
        print(f"📹 Loading: {self.video_path}")
        
        try:
            self.frames = read_video(self.video_path)
            if not self.frames:
                return False
            
            print(f"✅ Loaded {len(self.frames)} frames")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def initialize_components(self) -> bool:
        """Initialize AI components"""
        if not YOLO_AVAILABLE:
            return False
        
        try:
            print("🤖 Initializing AI tracker...")
            self.player_tracker = AIPlayerTracker(self.player_model)
            print("✅ AI tracker ready")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def process_video(self):
        """Process video with AI tracking"""
        print(f"🧠 AI processing {len(self.frames)} frames...")
        
        for i, frame in enumerate(self.frames):
            # AI player tracking (handles back-turned poses)
            player_positions = self.player_tracker.track_frame(frame)
            self.player_detections.append(player_positions)
            
            # Smart ball detection (small moving green object)
            player_bboxes = list(player_positions.values())
            ball_bbox = self.ball_detector.detect_ball(frame, player_bboxes)
            
            if ball_bbox:
                self.ball_detections.append({1: ball_bbox})
            else:
                self.ball_detections.append({})
            
            if i % 50 == 0:
                print(f"  Frame {i}/{len(self.frames)} ({i/len(self.frames)*100:.1f}%)")
        
        # Results summary
        p1_frames = sum(1 for d in self.player_detections if 1 in d)
        p2_frames = sum(1 for d in self.player_detections if 2 in d)
        ball_frames = sum(1 for d in self.ball_detections if d)
        
        print(f"🎯 AI Tracking Results:")
        print(f"  Player 1: {p1_frames}/{len(self.frames)} ({p1_frames/len(self.frames)*100:.1f}%)")
        print(f"  Player 2: {p2_frames}/{len(self.frames)} ({p2_frames/len(self.frames)*100:.1f}%)")
        print(f"  Ball: {ball_frames}/{len(self.frames)} ({ball_frames/len(self.frames)*100:.1f}%)")
    
    def generate_output(self, output_path: str = None) -> str:
        """Generate output with AI tracking results"""
        print("🎬 Generating AI-tracked output...")
        
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            os.makedirs("output_videos", exist_ok=True)
            output_path = f"output_videos/{base_name}_ai_tracked.avi"
        
        output_frames = []
        
        for i, frame in enumerate(self.frames):
            output_frame = frame.copy()
            
            # Draw AI-tracked players
            if i < len(self.player_detections):
                players = self.player_detections[i]
                
                # Player 1 - BLUE (with AI tracking status)
                if 1 in players:
                    bbox = players[1]
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Check if this is a prediction vs detection
                    lost_frames = self.player_tracker.players[1]['lost_frames']
                    if lost_frames == 0:
                        color = (255, 0, 0)  # Solid blue - good detection
                        label = "Player 1"
                    else:
                        color = (150, 0, 150)  # Purple - predicted
                        label = f"Player 1 (pred)"
                    
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(output_frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Player 2 - RED (with AI tracking status)
                if 2 in players:
                    bbox = players[2]
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    lost_frames = self.player_tracker.players[2]['lost_frames']
                    if lost_frames == 0:
                        color = (0, 0, 255)  # Solid red - good detection
                        label = "Player 2"
                    else:
                        color = (0, 150, 150)  # Orange - predicted
                        label = f"Player 2 (pred)"
                    
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(output_frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw ball (small green circle for pixelated moving object)
            if i < len(self.ball_detections) and self.ball_detections[i]:
                ball_dict = self.ball_detections[i]
                if 1 in ball_dict:
                    bbox = ball_dict[1]
                    x1, y1, x2, y2 = map(int, bbox)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Small green circle for ball
                    cv2.circle(output_frame, (center_x, center_y), 6, (0, 255, 0), 2)
                    cv2.putText(output_frame, "Ball", (center_x + 8, center_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Frame info
            cv2.putText(output_frame, f"Frame: {i+1} | AI Tracking", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            output_frames.append(output_frame)
        
        save_video(output_frames, output_path)
        print(f"✅ Saved: {output_path}")
        return output_path
    
    def run_analysis(self) -> bool:
        """Run AI-powered tennis analysis"""
        print("🧠 AI TENNIS TRACKER")
        print("="*40)
        
        if not self.load_video():
            return False
        
        if not self.initialize_components():
            return False
        
        self.process_video()
        
        output_path = self.generate_output()
        
        print(f"\n🎉 AI Analysis Complete!")
        print(f"📁 Output: {output_path}")
        print("\n🧠 AI Features:")
        print("  - Tracks players even when back is turned")
        print("  - Motion prediction during occlusions")
        print("  - Smart ball detection for small moving objects")
        print("  - Template matching recovery")
        print("  - Purple/orange boxes = AI prediction")
        print("  - Blue/red boxes = Direct detection")
        
        return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Tennis Tracker')
    parser.add_argument('video_path', nargs='?', help='Tennis video path')
    args = parser.parse_args()
    
    # Find video
    video_path = args.video_path
    if not video_path:
        video_files = []
        
        for f in os.listdir('.'):
            if f.lower().endswith(('.mp4', '.avi', '.mov')):
                video_files.append(f)
        
        if not video_files and os.path.exists('input_videos'):
            for f in os.listdir('input_videos'):
                if f.lower().endswith(('.mp4', '.avi', '.mov')):
                    video_files.append(os.path.join('input_videos', f))
        
        if video_files:
            video_path = video_files[0]
            print(f"📹 Using: {video_path}")
        else:
            print("❌ No video found")
            return
    
    if not YOLO_AVAILABLE:
        print("📦 Installing YOLO...")
        os.system("pip3 install ultralytics")
        print("🔄 Please restart")
        return
    
    analyzer = AITennisAnalyzer(video_path)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()