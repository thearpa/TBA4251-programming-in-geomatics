import json
import csv
import logging
import math
import os
import warnings
import numpy as np
import geopandas as gpd
import laspy
import shapely
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from citygml_writer import CityGMLWriter
gpd.options.io_engine = "fiona"


from config import (
    LAZ_DIR, FOOTPRINT_PATH, FOOTPRINT_LAYER, CRS_EPSG,
    MAX_JOIN_DIST, TOP_K, OUT_DIR, WALL_HEIGHT
)
from geometry_utils import to2d, largest_part
from roof_model import extract_roof_planes, classify_roof_type
from plots import (
     plot_building_2d, plot_building_3d, plot_histograms
)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Footprints
    if FOOTPRINT_LAYER:
        fp = gpd.read_file(FOOTPRINT_PATH, layer=FOOTPRINT_LAYER)
    else:
        fp = gpd.read_file(FOOTPRINT_PATH)
    fp = fp.to_crs(CRS_EPSG)
    fp["geometry"] = fp.geometry.apply(to2d)

    # LAZ → GeoDataFrame
    rows = []
    for name in sorted(os.listdir(LAZ_DIR)):
        if not name.lower().endswith(".laz"):
            continue
        path = os.path.join(LAZ_DIR, name)
        try:
            las = laspy.read(path)
        except Exception as e:
            logging.warning(f"Skipper {name}: {e}")
            continue
        x = np.asarray(las.x)
        y = np.asarray(las.y)
        z = np.asarray(las.z)
        if x.size == 0:
            continue
        rows.append({
            "file_name": name,
            "x": x, "y": y, "z": z,
            "min_z": float(np.min(z)),
            "max_z": float(np.max(z))
        })
    if not rows:
        raise SystemExit("Ingen LAZ-filer i LAZ_DIR.")

    pc = gpd.GeoDataFrame(rows)
    pc["geometry"] = pc.apply(
        lambda r: Point(float(np.mean(r["x"])), float(np.mean(r["y"]))),
        axis=1
    )
    pc = pc.set_geometry("geometry").set_crs(CRS_EPSG)

    # sjoin_nearest
    combined = fp.sjoin_nearest(
        pc, how="inner",
        distance_col="distance",
        max_distance=MAX_JOIN_DIST
    )
    combined = combined.reset_index(drop=True)
    logging.info(f"Footprints: {len(fp)} | LAZ: {len(pc)} | Matchede bygg: {len(combined)}")
    if len(combined) == 0:
        raise SystemExit("Ingen matches - sjekk CRS/MAX_JOIN_DIST.")

    # ranger etter antall takpunkt inne i footprint
    def count_pts_in_poly(row):
        poly = largest_part(row["geometry"])
        pts = shapely.points(np.column_stack([row["x"], row["y"]]))
        return int(shapely.contains(poly, pts).sum())

    combined["_pts_in"] = combined.apply(count_pts_in_poly, axis=1)
    combined = combined.sort_values("_pts_in", ascending=False).reset_index(drop=True)
    if TOP_K is not None:
        combined = combined.head(TOP_K).copy()
        logging.info(f"Begrenser til TOP_K={TOP_K}")

    reports = []
    citygml_all = CityGMLWriter(srs_epsg=CRS_EPSG)


    for idx, row in combined.iterrows():
        poly = largest_part(row["geometry"])
        x_pts = np.asarray(row["x"])
        y_pts = np.asarray(row["y"])
        z_pts = np.asarray(row["z"])
        zmin = float(row["min_z"])
        tag = row.get("file_name", f"roof_{idx:03d}")

        faces = extract_roof_planes(x_pts, y_pts, z_pts, poly)
        faces_rings3d = [f.rings3d for f in faces]

        roof_info = classify_roof_type(faces)
        roof_type = roof_info.get("type", "unknown")

        pts_all2d = shapely.points(np.column_stack([x_pts, y_pts]))
        rms_list = []
        ang_list = []

        for f in faces:
            mask = shapely.contains(f.poly2d, pts_all2d)
            Pin = np.c_[x_pts[mask], y_pts[mask], z_pts[mask]]
            if len(Pin) < 25:
                continue
            n, d = f.n, f.d
            z_pred = (-d - n[0] * Pin[:, 0] - n[1] * Pin[:, 1]) / n[2]
            rms = float(np.sqrt(np.mean((Pin[:, 2] - z_pred) ** 2)))
            rms_list.append(rms)
            ang = math.degrees(math.acos(min(1.0, max(-1.0, abs(n[2])))))
            ang_list.append(ang)

        faces2d = [f.poly2d for f in faces if isinstance(f.poly2d, Polygon)]
        cov_area = float(unary_union(faces2d).area / poly.area) if faces2d else 0.0

        logging.info(
            f"[{idx:02d}] {tag} -> faces={len(faces_rings3d)}  "
            f"roof_type={roof_type:<7}  "
            f"rms_mean={np.mean(rms_list) if rms_list else float('nan'):.3f}  "
            f"angle_dev_mean={np.mean(ang_list) if ang_list else float('nan'):.2f}°  "
            f"cov={cov_area:.2f}"
        )


        if faces_rings3d:
            bid = f"b_{idx:03d}"
           
            # -------- CityGML (roofs + walls) --------
            citygml_all.add_building(
                bid,
                faces_rings3d=faces_rings3d,
                footprint=poly,      # 2D footprint polygon
                ground_z=zmin,       # min_z for this building
                default_wall_height=WALL_HEIGHT  # or e.g. 3.0
            )

        plot_building_2d(
            idx, tag, poly, x_pts, y_pts, faces, int(row["_pts_in"]),
            os.path.join(OUT_DIR, f"{idx:02d}_{tag.replace('.laz','')}_2d.png")
        )

        plot_building_3d(
            idx, tag, poly, x_pts, y_pts, z_pts, faces_rings3d, zmin,
            os.path.join(OUT_DIR, f"{idx:02d}_{tag.replace('.laz','')}_3d.png")
        )

        reports.append({
            "tag": tag,
            "n_faces": len(faces_rings3d),
            "rms_mean": float(np.mean(rms_list)) if rms_list else None,
            "rms_median": float(np.median(rms_list)) if rms_list else None,
            "angle_dev_mean_deg": float(np.mean(ang_list)) if ang_list else None,
            "coverage": cov_area,
            "roof_type": roof_type
        })

    citygml_all.write(os.path.join(OUT_DIR, "_all_roofs.gml"))

    with open(os.path.join(OUT_DIR, "_report.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "tag", "n_faces", "rms_mean", "rms_median",
            "angle_dev_mean_deg", "coverage", "roof_type"
        ])
        w.writeheader()
        for r in reports:
            w.writerow(r)

    params = {
        "LAZ_DIR": LAZ_DIR,
        "FOOTPRINT_PATH": FOOTPRINT_PATH,
        "FOOTPRINT_LAYER": FOOTPRINT_LAYER,
        "CRS_EPSG": CRS_EPSG,
        "MAX_JOIN_DIST": MAX_JOIN_DIST,
        "TOP_K": TOP_K,
    }
    with open(os.path.join(OUT_DIR, "_params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    plot_histograms(reports)


    


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
