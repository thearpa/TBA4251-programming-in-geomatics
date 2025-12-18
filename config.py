import os
import numpy as np


LAZ_DIR = "sample_roofdata_50"          # Directory with LiDAR point clouds
FOOTPRINT_PATH = "footprints_25832.gpkg" # Building footprint dataset
FOOTPRINT_LAYER = "buildings"            # Footprint layer name
CRS_EPSG = 25832                         # Coordinate reference system

# MATCHING 

MAX_JOIN_DIST = 6.0                      # Max distance for footprint–point cloud matching
TOP_K = 50                               # Number of buildings to process

# PLANE EXTRACTION 

VOXEL_SIZE = 0.08                        # Voxel size for downsampling (m)
MAX_PLANES = 10                          # Max number of roof planes
RANSAC_DIST = 0.08                       # Initial RANSAC distance threshold (m)
RANSAC_N = 3                             # RANSAC sample size
RANSAC_ITERS = 5000                      # RANSAC iterations
MIN_INLIERS = 80                         # Minimum inlier points
MIN_INLIERS_REL = 0.01                   # Minimum inlier ratio

# ROOF POST-PROCESSING 

CONCAVE_RATIO = 0.25                     # Concave hull tightness
ALPHA_Q = 2.2                            # Alpha-shape parameter
INNER_CLIP = 0.05                        # Inward polygon buffer (m)
MIN_FACE_AREA = 0.8                      # Minimum roof face area (m²)
MAX_SLOPE_DEG = 60.0                     # Maximum roof slope (degrees)
MERGE_ANG_DEG = 5.0                      # Coplanar merge angle threshold
MERGE_DZ = 0.1                           # Coplanar height difference (m)

#  RIDGE MODEL 

EDGE_MIN_LEN = 0.6                       # Minimum edge length (m)
RIDGE_ANG_THR = 10.0                     # Ridge angle threshold (degrees)

#  OUTPUT 

MAX_POINTS_PLOT = 40_000                 # Max points shown in plots
WALL_HEIGHT = 8.0                        # Default wall height (m)
OUT_DIR = "output8"                      # Output directory
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(0)           # Random generator for plotting
