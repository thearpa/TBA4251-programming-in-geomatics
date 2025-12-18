# TBA4251-programming-in-geomatics

This project implements a plane-first, model-driven approach for reconstructing
LoD2 building roofs from segmented LiDAR point clouds and building footprints.
The output is exported as CityGML.

Requirements:
- Python 3.10+
- numpy, shapely, geopandas, open3d, fiona

Run:
python run_roof_recon.py

The program outputs LoD2 CityGML building models and optional visualizations.
