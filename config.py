# config.py
import os
import numpy as np

# ================== INPUT / PARAMS ==================

LAZ_DIR = "sample_roofdata_50"

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FOOTPRINT_PATH =  "footprints_25832.gpkg"
FOOTPRINT_LAYER = "buildings"   # eller None
 # None hvis SHP

CRS_EPSG = 25832

# Matching / utvalg
MAX_JOIN_DIST = 6.0
TOP_K = 20                           # None = alle

# Open3D planes
VOXEL_SIZE = 0.12                    # nedprøving (m)
MAX_PLANES = 10
RANSAC_DIST = 0.12                   # min-gulv – økes adaptivt
RANSAC_N = 3
RANSAC_ITERS = 5000
MIN_INLIERS = 150                    # min absolutte inliers
MIN_INLIERS_REL = 0.01            # 0.5% av downsampled

# Post-prosess flater
CONCAVE_RATIO = 0.25                 # 0..1 (lavere => strammere)
ALPHA_Q = 2.2                        # alpha-shape terskel fallback
INNER_CLIP = 0.05                    # liten innbuffer
MIN_FACE_AREA = 0.8                  # m² – dropp små flater
MAX_SLOPE_DEG = 60.0                 # maks tak-skråning (vegger ~90°)
MERGE_ANG_DEG = 5.0                  # koplanar-merge vinkel
MERGE_DZ = 0.1                      # koplanar-merge høydeforskjell

# Kant / ridge model
EDGE_MIN_LEN = 0.6                   # min delt kantlengde (m)
RIDGE_ANG_THR = 10.0                 # vinkelgrense for "ridge-lignende" (deg)

# Plot / IO
MAX_POINTS_PLOT = 40_000
WALL_HEIGHT = 8.0
OUT_DIR = "output/output_roof_recon"
os.makedirs(OUT_DIR, exist_ok=True)

# RNG til sampling i plott
rng = np.random.default_rng(0)
