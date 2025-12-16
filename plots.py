# plots.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import Polygon

from config import WALL_HEIGHT, MAX_POINTS_PLOT, OUT_DIR, rng


def plot_building_debug(points: np.ndarray,
                        footprint_polygon: Polygon,
                        faces_3d: list,
                        title: str = "Building Debug View"):
    fig = plt.figure(figsize=(12, 9), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    if points is not None and len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   s=1, c='gray', alpha=0.4, label="Point cloud")

    if footprint_polygon is not None and not footprint_polygon.is_empty:
        xs, ys = footprint_polygon.exterior.xy
        z_min = np.min(points[:, 2]) if len(points) > 0 else 0.0
        ax.plot(xs, ys, zs=z_min, color='black', linewidth=2, label="Footprint")

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan',
              'yellow', 'magenta', 'lime', 'pink']

    for i, face in enumerate(faces_3d):
        if len(face) < 3:
            continue
        color = colors[i % len(colors)]
        poly = Poly3DCollection([face], alpha=0.65, facecolor=color)
        poly.set_edgecolor('k')
        ax.add_collection3d(poly)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=70, azim=-120)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_building_2d(idx, tag, poly, x_pts, y_pts, faces, pts_in, path_out):
    take = min(len(x_pts), MAX_POINTS_PLOT)
    sel = rng.choice(len(x_pts), size=take, replace=False)
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    if isinstance(poly, Polygon):
        ax2.add_patch(MplPolygon(
            np.array(poly.exterior.coords),
            fill=False, lw=2, edgecolor="blue", label="Footprint"
        ))
    ax2.scatter(x_pts[sel], y_pts[sel], s=2, alpha=0.25, label="Roof points")
    for f in faces:
        ax2.add_patch(MplPolygon(
            np.array(f.poly2d.exterior.coords),
            fill=False, lw=1.0, edgecolor="crimson"
        ))
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title(f"{idx:02d} – {tag}  (pts_in={pts_in})")
    ax2.set_xlabel("Easting (m)")
    ax2.set_ylabel("Northing (m)")
    handles, labels = ax2.get_legend_handles_labels()
    if handles:
        seen = set(); H2 = []; L2 = []
        for h, l in zip(handles, labels):
            if l not in seen:
                H2.append(h); L2.append(l); seen.add(l)
        ax2.legend(H2, L2, loc="best")
    plt.tight_layout()
    plt.savefig(path_out, dpi=300)
    plt.close(fig2)


def plot_building_3d(idx, tag, poly, x_pts, y_pts, z_pts, faces_rings3d, zmin, path_out):
    fig3 = plt.figure(figsize=(9, 9))
    ax3 = fig3.add_subplot(111, projection="3d")
    if isinstance(poly, Polygon):
        xf, yf = poly.exterior.xy
        ax3.plot(xf, yf, zs=zmin - WALL_HEIGHT)
        ax3.plot(xf, yf, zs=zmin)
        for (px, py) in poly.exterior.coords:
            ax3.plot([px, px], [py, py],
                     [zmin - WALL_HEIGHT, zmin],
                     linewidth=0.5)
    base_cols = np.array([
        [0.85, 0.25, 0.25, 0.9],
        [0.25, 0.55, 0.85, 0.9],
        [0.25, 0.75, 0.35, 0.9],
        [0.85, 0.65, 0.20, 0.9]
    ])
    VE = 2.0
    for i_f, rings in enumerate(faces_rings3d):
        if not rings:
            continue
        outer = [(x, y, (z - zmin) * VE + zmin) for (x, y, z) in rings[0]]
        if len(outer) < 3:
            continue
        tris = []
        b0 = outer[0]
        for j in range(1, len(outer) - 1):
            tris.append([b0, outer[j], outer[j + 1]])
        pcoll = Poly3DCollection(
            tris,
            facecolors=[base_cols[i_f % 4]],
            edgecolors='none',
            linewidths=0.0
        )
        ax3.add_collection3d(pcoll)

    mins = np.array([x_pts.min(), y_pts.min(), z_pts.min()])
    maxs = np.array([x_pts.max(), y_pts.max(), z_pts.max()])
    ax3.set_box_aspect(maxs - mins)
    ax3.view_init(elev=25, azim=35)
    ax3.set_xlabel("Easting (m)")
    ax3.set_ylabel("Northing (m)")
    ax3.set_zlabel("Z (m)")
    ax3.set_title(f"{idx:02d} – {tag} (LoD2-ish roof)")
    plt.tight_layout()
    plt.savefig(path_out, dpi=300)
    plt.close(fig3)


def plot_histograms(reports):
    import os
    rms_vals = [r["rms_mean"] for r in reports if r["rms_mean"] is not None]
    cov_vals = [r["coverage"] for r in reports if r["coverage"] is not None]

    if rms_vals:
        fig_rms, ax_rms = plt.subplots()
        ax_rms.hist(rms_vals, bins=20)
        ax_rms.set_xlabel("RMS (m)")
        ax_rms.set_ylabel("Antall bygg")
        ax_rms.set_title("Fordeling av tak-RMS")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "_rms_hist.png"), dpi=200)
        plt.close(fig_rms)

    if cov_vals:
        fig_cov, ax_cov = plt.subplots()
        ax_cov.hist(cov_vals, bins=20)
        ax_cov.set_xlabel("Coverage (takflateunion / footprint)")
        ax_cov.set_ylabel("Antall bygg")
        ax_cov.set_title("Fordeling av coverage")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "_coverage_hist.png"), dpi=200)
        plt.close(fig_cov)


