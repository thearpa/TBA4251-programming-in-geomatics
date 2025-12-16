# geometry_utils.py
import math
from typing import List, Tuple

import numpy as np
import shapely
from shapely.geometry import (
    Point, Polygon, MultiPolygon, MultiPoint,
    LineString, MultiLineString
)
from shapely.ops import unary_union, triangulate, polygonize_full
from shapely.validation import make_valid

from config import CONCAVE_RATIO, ALPHA_Q, INNER_CLIP


def to2d(geom):
    if isinstance(geom, Polygon):
        return Polygon(
            np.asarray(geom.exterior.coords)[:, :2],
            [np.asarray(r.coords)[:, :2] for r in geom.interiors]
        )
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([to2d(g) for g in geom.geoms])
    return geom


def largest_part(geom):
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon) and len(geom.geoms) > 0:
        return max(list(geom.geoms), key=lambda p: p.area)
    return geom


def alpha_shape_polygon(points_xy: np.ndarray, alpha: float) -> Polygon:
    if len(points_xy) < 3:
        return MultiPoint(points_xy).convex_hull
    mp = MultiPoint([tuple(p) for p in points_xy])
    tris = triangulate(mp)
    if not tris:
        return mp.convex_hull

    keep = []
    for t in tris:
        if t.area <= 0:
            continue
        ratio = t.length / t.area
        if ratio < alpha:
            keep.append(t)

    if not keep:
        return mp.convex_hull

    merged = unary_union(keep)
    m, holes, dangles, cuts = polygonize_full(merged)
    poly = unary_union(list(m.geoms)) if hasattr(m, "geoms") else m
    return largest_part(poly) if (poly and not poly.is_empty) else mp.convex_hull


def hull2d(points_xy: np.ndarray, concave_ratio: float = CONCAVE_RATIO) -> Polygon:
    if len(points_xy) < 3:
        return MultiPoint(points_xy).convex_hull
    mp = MultiPoint([tuple(p) for p in points_xy])
    try:
        return shapely.concave_hull(mp, ratio=concave_ratio)
    except Exception:
        return alpha_shape_polygon(points_xy, alpha=ALPHA_Q)


def simplify_poly(poly: Polygon, tol: float = 0.20) -> Polygon:
    if poly.is_empty or not isinstance(poly, Polygon):
        return poly
    simp = poly.simplify(tol, preserve_topology=True)
    if simp.is_empty or not isinstance(simp, Polygon):
        return poly
    return simp


def adaptive_inner_clip(area: float) -> float:
    if area < 5.0:
        return 0.02
    if area < 20.0:
        return 0.035
    return INNER_CLIP


def adaptive_alpha_q(npts: int, base=ALPHA_Q) -> float:
    if npts < 200:
        return base + 0.8
    if npts < 800:
        return base + 0.4
    return base


def compute_coverage(faces_polys, footprint_poly) -> float:
    if not faces_polys:
        return 0.0
    try:
        union = unary_union([p for p in faces_polys if isinstance(p, Polygon)])
        if union.is_empty:
            return 0.0
        return float(union.area / footprint_poly.area)
    except Exception:
        return 0.0


# ===== Plan-lokalt hull =====

def _plane_basis_from_normal(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = np.asarray(n, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    if abs(n[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(ref, n)
    e1_norm = np.linalg.norm(e1)
    if e1_norm < 1e-12:
        e1 = np.array([1.0, 0.0, 0.0])
        e1_norm = 1.0
    e1 = e1 / e1_norm
    e2 = np.cross(n, e1)
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)
    return e1, e2


def plane_local_hull(Pxyz: np.ndarray, n: np.ndarray,
                     concave_ratio: float = CONCAVE_RATIO) -> Polygon:
    Pxyz = np.asarray(Pxyz, float)
    if len(Pxyz) < 3:
        return MultiPoint(Pxyz[:, :2]).convex_hull

    e1, e2 = _plane_basis_from_normal(n)
    p0 = Pxyz.mean(axis=0)

    dP = Pxyz - p0
    u = dP @ e1
    v = dP @ e2
    uv = np.column_stack([u, v])

    mp_uv = MultiPoint([tuple(p) for p in uv])
    try:
        hull_uv = shapely.concave_hull(mp_uv, ratio=concave_ratio)
    except Exception:
        hull_uv = alpha_shape_polygon(uv, alpha=adaptive_alpha_q(len(Pxyz)))

    if hull_uv.is_empty:
        return MultiPoint(Pxyz[:, :2]).convex_hull

    if not isinstance(hull_uv, Polygon):
        hull_uv = largest_part(hull_uv)

    if hull_uv.is_empty or not isinstance(hull_uv, Polygon):
        return MultiPoint(Pxyz[:, :2]).convex_hull

    coords_uv = np.asarray(hull_uv.exterior.coords)
    pts3d = p0 + coords_uv[:, 0:1] * e1 + coords_uv[:, 1:2] * e2
    poly_xy = Polygon(pts3d[:, :2])
    return poly_xy


def snap_ring_to_footprint(
    ring: List[Tuple[float, float]],
    footprint: Polygon,
    tol: float = 0.2
) -> List[Tuple[float, float]]:
    if not isinstance(footprint, Polygon):
        return ring
    fp_line = footprint.exterior
    snapped = []
    for (x, y) in ring:
        p = Point(x, y)
        d = p.distance(fp_line)
        if d < tol:
            s = fp_line.interpolate(fp_line.project(p))
            snapped.append((s.x, s.y))
        else:
            snapped.append((x, y))
    return snapped

