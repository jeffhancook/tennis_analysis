# Tennis court dimensions in meters (official measurements)
# These are used for converting pixel distances to real-world measurements

# Court lines
SINGLE_LINE_WIDTH = 8.23  # Width of singles court
DOUBLE_LINE_WIDTH = 10.97  # Width of doubles court including alleys
HALF_COURT_LINE_HEIGHT = 11.88  # Half court length
SERVICE_LINE_WIDTH = 6.4  # Width of service box
DOUBLE_ALLY_DIFFERENCE = 1.37  # Width of doubles alley
NO_MANS_LAND_HEIGHT = 5.48  # Distance from service line to baseline

# Player heights (for scaling reference)
PLAYER_1_HEIGHT_METERS = 1.88
PLAYER_2_HEIGHT_METERS = 1.91
AVERAGE_PLAYER_HEIGHT = 1.8  # Default assumption

# Court colors (for detection assistance)
COURT_COLORS = {
    'hard_court_blue': {'lower': [100, 50, 50], 'upper': [130, 255, 255]},
    'hard_court_green': {'lower': [40, 50, 50], 'upper': [80, 255, 255]},
    'clay_court': {'lower': [5, 50, 50], 'upper': [25, 255, 255]}
}

# Ball characteristics
TENNIS_BALL_DIAMETER_CM = 6.7  # Official tennis ball diameter
TENNIS_BALL_COLOR_YELLOW = {'lower': [20, 100, 100], 'upper': [40, 255, 255]}

# Video processing
DEFAULT_FPS = 24
ANALYSIS_SAMPLE_RATE = 5  # Process every Nth frame for speed