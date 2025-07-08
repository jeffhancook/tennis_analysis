import cv2
import numpy as np
from bbox_utils import get_center_of_bbox, measure_distance
from typing import List, Dict, Tuple, Optional

class SimplePlayerTracker:
    """
    Simplified player tracking for casual tennis videos.
    Uses basic object detection and manual assistance.
    """
    
    def __init__(self):
        self.players = {}
        self.player_count = 0
        self.tracking_started = False
        
    def manual_player_selection(self, frame: np.ndarray) -> Dict[int, Tuple[int, int, int, int]]:
        """
        Manual player selection by clicking on players in first frame
        
        Args:
            frame: First frame of video
            
        Returns:
            Dictionary with player_id -> bbox mapping
        """
        print("\n👥 Manual Player Selection")
        print("Click on each player to select them (up to 2 players)")
        print("Press 'q' when done")
        
        selected_players = {}
        click_points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(click_points) < 2:
                click_points.append((x, y))
                # Create a simple bounding box around click point
                bbox_size = 100  # Default bbox size
                x1 = max(0, x - bbox_size//2)
                y1 = max(0, y - bbox_size//2)
                x2 = min(frame.shape[1], x + bbox_size//2)
                y2 = min(frame.shape[0], y + bbox_size//2)
                
                player_id = len(click_points)
                selected_players[player_id] = (x1, y1, x2, y2)
                
                # Draw the selection
                display_frame = param['display_frame']
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Player {player_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Select Players", display_frame)
                
                print(f"Selected Player {player_id} at ({x}, {y})")
        
        # Setup window
        display_frame = frame.copy()
        cv2.namedWindow("Select Players", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Players", 800, 600)
        cv2.setMouseCallback("Select Players", mouse_callback, {'display_frame': display_frame})
        
        cv2.imshow("Select Players", display_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or len(click_points) >= 2:
                break
        
        cv2.destroyAllWindows()
        
        print(f"✅ Selected {len(selected_players)} players")
        return selected_players
    
    def simple_tracking(self, frames: List[np.ndarray], initial_players: Dict[int, Tuple[int, int, int, int]]) -> List[Dict[int, Tuple[int, int, int, int]]]:
        """
        Simple tracking using template matching
        
        Args:
            frames: List of video frames
            initial_players: Initial player bounding boxes from first frame
            
        Returns:
            List of player detections for each frame
        """
        print(f"\n🔍 Tracking {len(initial_players)} players across {len(frames)} frames...")
        
        all_detections = []
        current_players = initial_players.copy()
        
        # Extract templates from first frame
        templates = {}
        for player_id, bbox in initial_players.items():
            x1, y1, x2, y2 = bbox
            template = frames[0][y1:y2, x1:x2]
            templates[player_id] = template
        
        for frame_idx, frame in enumerate(frames):
            frame_detections = {}
            
            # Track each player
            for player_id, template in templates.items():
                if player_id in current_players:
                    # Get search area around last known position
                    last_bbox = current_players[player_id]
                    new_bbox = self._track_template(frame, template, last_bbox)
                    
                    if new_bbox:
                        frame_detections[player_id] = new_bbox
                        current_players[player_id] = new_bbox
            
            all_detections.append(frame_detections)
            
            # Progress indicator
            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx}/{len(frames)} frames...")
        
        print("✅ Tracking complete!")
        return all_detections
    
    def _track_template(self, frame: np.ndarray, template: np.ndarray, last_bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        Track using template matching around last known position
        """
        if template.size == 0:
            return None
            
        x1, y1, x2, y2 = last_bbox
        template_h, template_w = template.shape[:2]
        
        # Define search area (expand around last position)
        search_expansion = 50
        search_x1 = max(0, x1 - search_expansion)
        search_y1 = max(0, y1 - search_expansion)
        search_x2 = min(frame.shape[1], x2 + search_expansion)
        search_y2 = min(frame.shape[0], y2 + search_expansion)
        
        search_area = frame[search_y1:search_y2, search_x1:search_x2]
        
        if search_area.size == 0:
            return last_bbox  # Return last known position
        
        # Template matching
        try:
            result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            # If match is good enough
            if max_val > 0.5:  # Threshold for good match
                # Convert back to frame coordinates
                match_x, match_y = max_loc
                new_x1 = search_x1 + match_x
                new_y1 = search_y1 + match_y
                new_x2 = new_x1 + template_w
                new_y2 = new_y1 + template_h
                
                return (new_x1, new_y1, new_x2, new_y2)
        except:
            pass
        
        # If tracking failed, return last known position
        return last_bbox

class SimpleBallTracker:
    """
    Simplified ball tracking using color detection and motion prediction
    """
    
    def __init__(self):
        self.ball_color_range = {
            'lower': np.array([0, 100, 100]),    # Yellow-green lower bound
            'upper': np.array([40, 255, 255])    # Yellow-green upper bound
        }
        self.min_ball_area = 10
        self.max_ball_area = 500
    
    def detect_ball_by_color(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect tennis ball using color detection
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for yellow/green ball
        mask = cv2.inRange(hsv, self.ball_color_range['lower'], self.ball_color_range['upper'])
        
        # Clean up mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour within size constraints
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_ball_area <= area <= self.max_ball_area:
                    valid_contours.append((contour, area))
            
            if valid_contours:
                # Get largest valid contour
                best_contour = max(valid_contours, key=lambda x: x[1])[0]
                x, y, w, h = cv2.boundingRect(best_contour)
                return (x, y, x + w, y + h)
        
        return None
    
    def track_ball_in_frames(self, frames: List[np.ndarray]) -> List[Optional[Tuple[int, int, int, int]]]:
        """
        Track ball across all frames
        """
        print(f"\n🎾 Tracking tennis ball across {len(frames)} frames...")
        
        ball_detections = []
        
        for frame_idx, frame in enumerate(frames):
            ball_bbox = self.detect_ball_by_color(frame)
            ball_detections.append({1: ball_bbox} if ball_bbox else {})
            
            # Progress indicator
            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx}/{len(frames)} frames...")
        
        print("✅ Ball tracking complete!")
        return ball_detections

def draw_tracking_results(frames: List[np.ndarray], 
                         player_detections: List[Dict[int, Tuple[int, int, int, int]]], 
                         ball_detections: List[Dict[int, Tuple[int, int, int, int]]]) -> List[np.ndarray]:
    """
    Draw tracking results on frames
    """
    output_frames = []
    
    for frame_idx, frame in enumerate(frames):
        output_frame = frame.copy()
        
        # Draw players
        if frame_idx < len(player_detections):
            for player_id, bbox in player_detections[frame_idx].items():
                x1, y1, x2, y2 = bbox
                color = (255, 0, 0) if player_id == 1 else (0, 0, 255)
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(output_frame, f"Player {player_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw ball
        if frame_idx < len(ball_detections) and ball_detections[frame_idx]:
            for ball_id, bbox in ball_detections[frame_idx].items():
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(output_frame, "Ball", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        output_frames.append(output_frame)
    
    return output_frames