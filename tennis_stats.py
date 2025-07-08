import numpy as np
from typing import List, Dict, Tuple, Optional
from bbox_utils import get_center_of_bbox, measure_distance
import tennis_constants as const

class TennisStatsCalculator:
    """
    Calculate tennis statistics from tracking data.
    Focuses on reliable metrics for casual videos.
    """
    
    def __init__(self, fps: float = const.DEFAULT_FPS):
        self.fps = fps
        self.court_reference = None
        self.pixel_to_meter_ratio = None
        
    def set_court_reference(self, court_corners: List[Tuple[int, int]]):
        """
        Set court reference for pixel-to-meter conversion
        
        Args:
            court_corners: List of 4 court corner coordinates [(x1,y1), (x2,y2), ...]
        """
        if len(court_corners) >= 4:
            self.court_reference = court_corners
            # Estimate pixel to meter ratio from court width
            court_width_pixels = max([p[0] for p in court_corners]) - min([p[0] for p in court_corners])
            self.pixel_to_meter_ratio = const.DOUBLE_LINE_WIDTH / court_width_pixels
    
    def calculate_comprehensive_stats(self, 
                                    player_detections: List[Dict[int, Tuple[int, int, int, int]]], 
                                    ball_detections: List[Dict[int, Tuple[int, int, int, int]]] = None) -> Dict:
        """
        Calculate comprehensive tennis statistics
        
        Args:
            player_detections: List of player detection dictionaries for each frame
            ball_detections: List of ball detection dictionaries for each frame (optional)
            
        Returns:
            Dictionary containing all calculated statistics
        """
        stats = {
            'session_info': self._calculate_session_info(player_detections),
            'player_stats': self._calculate_player_stats(player_detections),
            'movement_stats': self._calculate_movement_stats(player_detections),
            'court_coverage': self._calculate_court_coverage(player_detections)
        }
        
        # Add ball statistics if available
        if ball_detections:
            stats['ball_stats'] = self._calculate_ball_stats(ball_detections, player_detections)
            stats['rally_stats'] = self._calculate_rally_stats(ball_detections, player_detections)
        
        return stats
    
    def _calculate_session_info(self, player_detections: List[Dict]) -> Dict:
        """Calculate basic session information"""
        total_frames = len(player_detections)
        duration_seconds = total_frames / self.fps
        
        # Count frames where players are detected
        frames_with_players = sum(1 for frame in player_detections if frame)
        detection_rate = frames_with_players / total_frames if total_frames > 0 else 0
        
        return {
            'total_frames': total_frames,
            'duration_seconds': duration_seconds,
            'duration_minutes': duration_seconds / 60,
            'fps': self.fps,
            'detection_rate': detection_rate,
            'frames_analyzed': frames_with_players
        }
    
    def _calculate_player_stats(self, player_detections: List[Dict]) -> Dict:
        """Calculate per-player statistics"""
        player_stats = {}
        
        # Identify all players
        all_players = set()
        for frame in player_detections:
            all_players.update(frame.keys())
        
        for player_id in all_players:
            # Count appearances
            appearances = sum(1 for frame in player_detections if player_id in frame)
            activity_rate = appearances / len(player_detections)
            
            # Get all positions
            positions = []
            for frame in player_detections:
                if player_id in frame:
                    center = get_center_of_bbox(frame[player_id])
                    positions.append(center)
            
            # Calculate position statistics
            if positions:
                x_positions = [p[0] for p in positions]
                y_positions = [p[1] for p in positions]
                
                player_stats[player_id] = {
                    'appearances': appearances,
                    'activity_rate': activity_rate,
                    'avg_x_position': np.mean(x_positions),
                    'avg_y_position': np.mean(y_positions),
                    'position_variance_x': np.var(x_positions),
                    'position_variance_y': np.var(y_positions),
                    'court_coverage_area': self._calculate_coverage_area(positions)
                }
        
        return player_stats
    
    def _calculate_movement_stats(self, player_detections: List[Dict]) -> Dict:
        """Calculate movement-related statistics"""
        movement_stats = {}
        
        for player_id in [1, 2]:  # Assume max 2 players
            distances = []
            speeds = []
            previous_position = None
            
            for frame in player_detections:
                if player_id in frame:
                    current_position = get_center_of_bbox(frame[player_id])
                    
                    if previous_position:
                        # Calculate distance moved
                        distance_pixels = measure_distance(previous_position, current_position)
                        distances.append(distance_pixels)
                        
                        # Calculate speed (pixels per second, convert to m/s if we have reference)
                        speed_pixels_per_second = distance_pixels * self.fps
                        if self.pixel_to_meter_ratio:
                            speed_ms = speed_pixels_per_second * self.pixel_to_meter_ratio
                            speed_kmh = speed_ms * 3.6
                            speeds.append(speed_kmh)
                        else:
                            speeds.append(speed_pixels_per_second)
                    
                    previous_position = current_position
                else:
                    previous_position = None
            
            if distances:
                movement_stats[player_id] = {
                    'total_distance_pixels': sum(distances),
                    'avg_speed': np.mean(speeds) if speeds else 0,
                    'max_speed': max(speeds) if speeds else 0,
                    'movement_intensity': np.std(speeds) if speeds else 0,
                    'total_movements': len(distances)
                }
                
                if self.pixel_to_meter_ratio:
                    movement_stats[player_id]['total_distance_meters'] = sum(distances) * self.pixel_to_meter_ratio
        
        return movement_stats
    
    def _calculate_court_coverage(self, player_detections: List[Dict]) -> Dict:
        """Calculate court coverage statistics"""
        coverage_stats = {}
        
        for player_id in [1, 2]:
            positions = []
            for frame in player_detections:
                if player_id in frame:
                    center = get_center_of_bbox(frame[player_id])
                    positions.append(center)
            
            if positions:
                x_positions = [p[0] for p in positions]
                y_positions = [p[1] for p in positions]
                
                coverage_stats[player_id] = {
                    'min_x': min(x_positions),
                    'max_x': max(x_positions),
                    'min_y': min(y_positions),
                    'max_y': max(y_positions),
                    'court_width_used': max(x_positions) - min(x_positions),
                    'court_height_used': max(y_positions) - min(y_positions),
                    'positions_sampled': len(positions)
                }
        
        return coverage_stats
    
    def _calculate_ball_stats(self, ball_detections: List[Dict], player_detections: List[Dict]) -> Dict:
        """Calculate ball-related statistics"""
        ball_stats = {
            'total_ball_detections': 0,
            'ball_detection_rate': 0,
            'estimated_shots': 0,
            'avg_ball_speed': 0
        }
        
        ball_positions = []
        for frame in ball_detections:
            if frame and 1 in frame:  # Assuming ball has ID 1
                ball_stats['total_ball_detections'] += 1
                center = get_center_of_bbox(frame[1])
                ball_positions.append(center)
        
        if ball_positions:
            ball_stats['ball_detection_rate'] = ball_stats['total_ball_detections'] / len(ball_detections)
            
            # Estimate shots by counting direction changes
            direction_changes = 0
            for i in range(2, len(ball_positions)):
                # Simple direction change detection
                prev_dx = ball_positions[i-1][0] - ball_positions[i-2][0]
                curr_dx = ball_positions[i][0] - ball_positions[i-1][0]
                
                if prev_dx * curr_dx < 0:  # Sign change indicates direction change
                    direction_changes += 1
            
            ball_stats['estimated_shots'] = direction_changes
            
            # Calculate ball speeds
            speeds = []
            for i in range(1, len(ball_positions)):
                distance = measure_distance(ball_positions[i-1], ball_positions[i])
                speed = distance * self.fps  # pixels per second
                if self.pixel_to_meter_ratio:
                    speed_ms = speed * self.pixel_to_meter_ratio
                    speed_kmh = speed_ms * 3.6
                    speeds.append(speed_kmh)
            
            if speeds:
                ball_stats['avg_ball_speed'] = np.mean(speeds)
                ball_stats['max_ball_speed'] = max(speeds)
        
        return ball_stats
    
    def _calculate_rally_stats(self, ball_detections: List[Dict], player_detections: List[Dict]) -> Dict:
        """Calculate rally-related statistics"""
        rally_stats = {
            'total_rallies': 0,
            'avg_rally_length': 0,
            'longest_rally': 0,
            'rally_lengths': []
        }
        
        # Simple rally detection based on ball movement patterns
        ball_positions = []
        for frame in ball_detections:
            if frame and 1 in frame:
                center = get_center_of_bbox(frame[1])
                ball_positions.append(center)
            else:
                ball_positions.append(None)
        
        # Find continuous ball tracking segments (rallies)
        rally_lengths = []
        current_rally_length = 0
        
        for pos in ball_positions:
            if pos is not None:
                current_rally_length += 1
            else:
                if current_rally_length > 10:  # Minimum rally length
                    rally_lengths.append(current_rally_length)
                current_rally_length = 0
        
        if current_rally_length > 10:
            rally_lengths.append(current_rally_length)
        
        if rally_lengths:
            rally_stats['total_rallies'] = len(rally_lengths)
            rally_stats['avg_rally_length'] = np.mean(rally_lengths) / self.fps  # Convert to seconds
            rally_stats['longest_rally'] = max(rally_lengths) / self.fps
            rally_stats['rally_lengths'] = [r / self.fps for r in rally_lengths]
        
        return rally_stats
    
    def _calculate_coverage_area(self, positions: List[Tuple[int, int]]) -> float:
        """Calculate approximate area covered by player positions"""
        if len(positions) < 3:
            return 0
        
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        
        # Simple bounding box area
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        return width * height
    
    def print_stats_summary(self, stats: Dict):
        """Print a formatted summary of statistics"""
        print("\n" + "="*60)
        print("🎾 TENNIS ANALYSIS SUMMARY")
        print("="*60)
        
        # Session info
        session = stats['session_info']
        print(f"📹 Session Duration: {session['duration_minutes']:.1f} minutes")
        print(f"🎬 Total Frames: {session['total_frames']}")
        print(f"🎯 Detection Rate: {session['detection_rate']:.1%}")
        
        # Player stats
        print(f"\n👥 PLAYER STATISTICS:")
        for player_id, player_stats in stats['player_stats'].items():
            print(f"  Player {player_id}:")
            print(f"    Activity Rate: {player_stats['activity_rate']:.1%}")
            print(f"    Court Coverage: {player_stats['court_coverage_area']:.0f} px²")
        
        # Movement stats
        if 'movement_stats' in stats:
            print(f"\n🏃 MOVEMENT STATISTICS:")
            for player_id, movement in stats['movement_stats'].items():
                print(f"  Player {player_id}:")
                if 'total_distance_meters' in movement:
                    print(f"    Distance Covered: {movement['total_distance_meters']:.1f}m")
                    print(f"    Average Speed: {movement['avg_speed']:.1f} km/h")
                    print(f"    Max Speed: {movement['max_speed']:.1f} km/h")
                else:
                    print(f"    Total Distance: {movement['total_distance_pixels']:.0f} pixels")
                    print(f"    Movement Count: {movement['total_movements']}")
        
        # Ball stats
        if 'ball_stats' in stats:
            ball = stats['ball_stats']
            print(f"\n🎾 BALL STATISTICS:")
            print(f"    Ball Detection Rate: {ball['ball_detection_rate']:.1%}")
            print(f"    Estimated Shots: {ball['estimated_shots']}")
            if ball['avg_ball_speed'] > 0:
                print(f"    Average Ball Speed: {ball['avg_ball_speed']:.1f} km/h")
        
        # Rally stats
        if 'rally_stats' in stats and stats['rally_stats']['total_rallies'] > 0:
            rally = stats['rally_stats']
            print(f"\n🏓 RALLY STATISTICS:")
            print(f"    Total Rallies: {rally['total_rallies']}")
            print(f"    Average Rally: {rally['avg_rally_length']:.1f} seconds")
            print(f"    Longest Rally: {rally['longest_rally']:.1f} seconds")
        
        print("="*60)