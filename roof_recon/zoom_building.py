"""
Standalone script to load a CityGML file and visualise buildings properly zoomed
and centred in a local coordinate system.

Features:
- Parses CityGML (RoofSurface + WallSurface)
- Extracts all gml:Polygon exterior rings
- Converts to local coordinates (centered + ground-normalised)
- Auto-zooms tightly around geometry
- Renders solid 3D faces with edges

Usage:
    python zoom_building.py path/to/file.gml

Optional:
    python zoom_building.py output1/_all_roofs.gml --building b_000
"""

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse

NS = {
    "gml": "http://www.opengis.net/gml",
    "core": "http://www.opengis.net/citygml/2.0",
    "bldg": "http://www.opengis.net/citygml/building/2.0",
}


def extract_polygons_from_citygml(path, building_id=None):
    tree = ET.parse(path)
    root = tree.getroot()

    polygons = []

    for building in root.findall(".//bldg:Building", NS):
        bid = building.attrib.get("{http://www.opengis.net/gml}id")
        if building_id and bid != building_id:
            continue

        for poly in building.findall(".//gml:Polygon", NS):
            pos = poly.find(".//gml:exterior//gml:posList", NS)
            if pos is None or not pos.text:
                continue

            vals = pos.text.split()
            if len(vals) % 3 != 0:
                continue

            coords = []
            for i in range(0, len(vals), 3):
                coords.append((float(vals[i]), float(vals[i+1]), float(vals[i+2])))

            polygons.append(np.array(coords))

    return polygons


def normalise_local(polygons):
    pts = np.vstack(polygons)
    cx = pts[:,0].mean()
    cy = pts[:,1].mean()
    cz = pts[:,2].min()

    norm_polys = []
    for poly in polygons:
        p = poly.copy()
        p[:,0] -= cx
        p[:,1] -= cy
        p[:,2] -= cz
        norm_polys.append(p)

    return norm_polys


def plot_polygons(polygons):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    all_pts = np.vstack(polygons)

    for poly in polygons:
        pc = Poly3DCollection([poly], alpha=0.7, edgecolor='k', linewidth=0.8)
        ax.add_collection3d(pc)

    # Automatic tight zoom
    min_xyz = all_pts.min(axis=0)
    max_xyz = all_pts.max(axis=0)

    ax.set_xlim(min_xyz[0], max_xyz[0])
    ax.set_ylim(min_xyz[1], max_xyz[1])
    ax.set_zlim(min_xyz[2], max_xyz[2])

    ax.set_xlabel("Local Easting (m)")
    ax.set_ylabel("Local Northing (m)")
    ax.set_zlabel("Height (m)")
    ax.set_box_aspect([
        max_xyz[0]-min_xyz[0],
        max_xyz[1]-min_xyz[1],
        max_xyz[2]-min_xyz[2]
    ])

    plt.tight_layout()
    plt.show()



def main():
    parser = argparse.ArgumentParser(description="Zoomed 3D preview of CityGML buildings")
    parser.add_argument("gml", help="Path to CityGML file")
    parser.add_argument("--building", help="Specific building id (e.g. b_000)")
    args = parser.parse_args()

    polys = extract_polygons_from_citygml(args.gml, args.building)
    if not polys:
        print("No polygons found in file")
        return

    polys = normalise_local(polys)
    plot_polygons(polys)


if __name__ == "__main__":
    main()
