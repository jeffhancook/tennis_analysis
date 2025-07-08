import math

def get_center_of_bbox(bbox):
    """Get center point of bounding box"""
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def measure_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_foot_position(bbox):
    """Get foot position (bottom center) of bounding box"""
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_height_of_bbox(bbox):
    """Get height of bounding box"""
    return bbox[3] - bbox[1]

def get_width_of_bbox(bbox):
    """Get width of bounding box"""
    return bbox[2] - bbox[0]

def measure_xy_distance(p1, p2):
    """Calculate separate x and y distances between points"""
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])

def is_point_in_bbox(point, bbox):
    """Check if point is inside bounding box"""
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def expand_bbox(bbox, expansion_factor=1.1):
    """Expand bounding box by given factor"""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    
    new_width = width * expansion_factor
    new_height = height * expansion_factor
    
    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2
    
    return (int(new_x1), int(new_y1), int(new_x2), int(new_y2))