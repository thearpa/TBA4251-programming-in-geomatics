import logging
import math
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import open3d as o3d
import shapely
from shapely.geometry import Polygon, MultiLineString, LineString
from shapely.ops import unary_union, split
from shapely.validation import make_valid

from config import (
    VOXEL_SIZE, MAX_PLANES, RANSAC_DIST, RANSAC_N, RANSAC_ITERS,
    MIN_INLIERS, MIN_INLIERS_REL, MAX_SLOPE_DEG, MERGE_ANG_DEG,
    MERGE_DZ, EDGE_MIN_LEN, RIDGE_ANG_THR, MIN_FACE_AREA
)
from geometry_utils import (
    plane_local_hull, hull2d, alpha_shape_polygon, adaptive_alpha_q,
    adaptive_inner_clip, simplify_poly, compute_coverage, snap_ring_to_footprint
)


@dataclass
class FaceRec:
    poly2d: Polygon
    n: np.ndarray
    d: float
    rings3d: List[List[tuple]]  # [outer, hole1, ...]


# ===== Planefit / grunnleggende =====

def fit_plane_ls_irls(Pxyz: np.ndarray, iters: int = 3):
    X = np.c_[Pxyz[:, 0], Pxyz[:, 1], np.ones(len(Pxyz))]
    z = Pxyz[:, 2]
    w = np.ones(len(Pxyz))

    for _ in range(iters):
        W = np.diag(w)
        a, b, c = np.linalg.lstsq(W @ X, W @ z, rcond=None)[0]
        pred = a * Pxyz[:, 0] + b * Pxyz[:, 1] + c
        resid = z - pred
        med = np.median(resid)
        mad = np.median(np.abs(resid - med)) + 1e-12
        sigma = 1.4826 * mad
        k = 1.345 * sigma
        absr = np.abs(resid)
        w = np.where(absr <= k, 1.0, k / absr)

    n_raw = np.array([a, b, -1.0], float)
    scale = np.linalg.norm(n_raw) + 1e-12
    n = n_raw / scale
    d = -c / scale
    if n[2] < 0:
        n, d = -n, -d

    z_pred = (-d - n[0] * Pxyz[:, 0] - n[1] * Pxyz[:, 1]) / n[2]
    resid = Pxyz[:, 2] - z_pred
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) + 1e-12
    sigma = 1.4826 * mad
    return n, d, sigma, resid


def is_roofish(nz: float) -> bool:
    nz_clamped = min(1.0, max(0.0, abs(nz)))
    slope_deg = math.degrees(math.acos(nz_clamped))
    return slope_deg <= MAX_SLOPE_DEG


def z_on_plane(n: np.ndarray, d: float, x: float, y: float) -> float:
    nx, ny, nz = n
    if abs(nz) < 1e-10:
        return np.nan
    return (-d - nx * x - ny * y) / nz


def lift_ring(n, d, ring_xy):
    return [(x, y, z_on_plane(n, d, x, y)) for (x, y) in ring_xy]


def polygon_to_rings3d(poly: Polygon, n, d):
    ext = lift_ring(n, d, list(poly.exterior.coords)[:-1])
    holes = [lift_ring(n, d, list(r.coords)[:-1]) for r in poly.interiors]
    return [ext] + holes


# ===== Taktype / modellregler =====

def classify_roof_type(faces: List[FaceRec]) -> Dict[str, Any]:
    if not faces:
        return {"type": "unknown", "clusters": []}

    normals = np.array([f.n for f in faces])
    nz = np.abs(normals[:, 2])

    if len(faces) == 1 or float(np.mean(nz)) > 0.97:
        return {"type": "flat", "clusters": [0] * len(faces)}

    az = []
    for n in normals:
        vx, vy = n[0], n[1]
        if vx == 0 and vy == 0:
            az.append(0.0)
        else:
            ang = math.degrees(math.atan2(vy, vx)) % 180.0
            az.append(ang)
    az = np.array(az)

    bucket = (az / 30.0).round().astype(int)
    unique = sorted(set(bucket))
    mapping = {b: i for i, b in enumerate(unique)}
    clusters = [mapping[b] for b in bucket]
    k = len(unique)

    if k == 1:
        roof_type = "shed"
    elif k == 2:
        roof_type = "gable"
    elif k == 3:
        roof_type = "hip"
    else:
        roof_type = "complex"

    return {"type": roof_type, "clusters": clusters}


def compute_ridge_segment_2d(f1: FaceRec, f2: FaceRec, footprint: Polygon) -> LineString | None:
    n1, d1 = f1.n, f1.d
    n2, d2 = f2.n, f2.d

    dir3 = np.cross(n1, n2)
    if np.linalg.norm(dir3) < 1e-6:
        return None

    dir_xy = dir3[:2]
    if np.linalg.norm(dir_xy) < 1e-6:
        return None
    dir_xy = dir_xy / np.linalg.norm(dir_xy)

    cx, cy = footprint.centroid.x, footprint.centroid.y
    minx, miny, maxx, maxy = footprint.bounds
    half_len = 2.0 * max(maxx - minx, maxy - miny)

    p1 = (cx - dir_xy[0] * half_len, cy - dir_xy[1] * half_len)
    p2 = (cx + dir_xy[0] * half_len, cy + dir_xy[1] * half_len)
    big_line = LineString([p1, p2])

    inter = footprint.intersection(big_line)
    if inter.is_empty:
        return None

    if isinstance(inter, LineString):
        seg = inter
    elif isinstance(inter, MultiLineString):
        seg = max(list(inter.geoms), key=lambda g: g.length)
    else:
        return None

    if seg.length < EDGE_MIN_LEN:
        return None
    return seg


def refine_simple_gable(
    faces: List[FaceRec],
    footprint: Polygon
) -> List[FaceRec]:
    if len(faces) != 2:
        return faces

    info = classify_roof_type(faces)
    if info.get("type") != "gable":
        return faces

    f1, f2 = faces

    cosang = float(np.clip(np.dot(f1.n, f2.n), -1, 1))
    ang = math.degrees(math.acos(cosang))
    if not (5.0 <= ang <= 45.0):
        return faces

    ridge = compute_ridge_segment_2d(f1, f2, footprint)
    if ridge is None:
        return faces

    try:
        sp = split(footprint, ridge)
    except Exception:
        return faces

    parts = [g for g in sp.geoms if isinstance(g, Polygon)]
    if len(parts) != 2:
        return faces

    parts = [simplify_poly(p, tol=0.15) for p in parts]

    new_faces: List[FaceRec] = []
    for base, poly in zip((f1, f2), parts):
        if poly.is_empty or not poly.is_valid:
            continue
        rings3d = polygon_to_rings3d(poly, base.n, base.d)
        new_faces.append(FaceRec(poly2d=poly, n=base.n, d=base.d, rings3d=rings3d))

    return new_faces if len(new_faces) == 2 else faces


def force_simple_gable_ridge_height(faces: List[FaceRec], footprint: Polygon):
    if len(faces) != 2:
        return
    f1, f2 = faces

    ridge = compute_ridge_segment_2d(f1, f2, footprint)
    if ridge is None or ridge.length < EDGE_MIN_LEN:
        return

    c0 = ridge.coords[0]
    c1 = ridge.coords[-1]
    cm = ridge.interpolate(0.5, normalized=True).coords[0]
    samples = [c0, cm, c1]

    z_targets = []
    for (x, y) in samples:
        z1 = z_on_plane(f1.n, f1.d, x, y)
        z2 = z_on_plane(f2.n, f2.d, x, y)
        if np.isfinite(z1) and np.isfinite(z2):
            z_targets.append(0.5 * (z1 + z2))

    if not z_targets:
        return

    z_target = float(np.mean(z_targets))
    mx, my = cm

    for f in (f1, f2):
        f.d = -(f.n[0]*mx + f.n[1]*my + f.n[2]*z_target)
        f.rings3d = polygon_to_rings3d(f.poly2d, f.n, f.d)


def is_horizontal_ridge_line(line: LineString, fi: FaceRec, fj: FaceRec,
                             dz_thr: float = 0.3, slope_thr: float = 0.1) -> bool:
    coords = list(line.coords)
    if len(coords) < 2:
        return False
    (x1, y1) = coords[0]
    (x2, y2) = coords[-1]
    L = line.length + 1e-9

    z1_i = z_on_plane(fi.n, fi.d, x1, y1)
    z2_i = z_on_plane(fi.n, fi.d, x2, y2)
    z1_j = z_on_plane(fj.n, fj.d, x1, y1)
    z2_j = z_on_plane(fj.n, fj.d, x2, y2)

    if not (np.isfinite(z1_i) and np.isfinite(z2_i) and
            np.isfinite(z1_j) and np.isfinite(z2_j)):
        return False

    if max(abs(z1_i - z1_j), abs(z2_i - z2_j)) > dz_thr:
        return False

    slope_i = abs(z2_i - z1_i) / L
    slope_j = abs(z2_j - z1_j) / L
    if max(slope_i, slope_j) > slope_thr:
        return False

    return True


def extract_shared_lines(p1: Polygon, p2: Polygon) -> List[LineString]:
    inter = p1.boundary.intersection(p2.boundary)
    out = []
    if isinstance(inter, LineString):
        if inter.length >= EDGE_MIN_LEN:
            out.append(inter)
    elif isinstance(inter, MultiLineString):
        out.extend([ls for ls in inter.geoms if ls.length >= EDGE_MIN_LEN])
    return out



def classify_edges(faces: List[FaceRec]) -> List[dict]:
    edges = []
    nF = len(faces)
    if nF < 2:
        return edges

    for i in range(nF):
        fi = faces[i]
        for j in range(i + 1, nF):
            fj = faces[j]

            cosang = float(np.clip(np.dot(fi.n, fj.n), -1, 1))
            ang = math.degrees(math.acos(cosang))

            if ang < RIDGE_ANG_THR or ang > 180.0 - RIDGE_ANG_THR:
                continue

            shared = extract_shared_lines(fi.poly2d, fj.poly2d)
            if shared:
                for ls in shared:
                    if ls.length < EDGE_MIN_LEN:
                        continue
                    if not is_horizontal_ridge_line(ls, fi, fj):
                        continue
                    edges.append({
                        "type": "ridge_line",
                        "i": i, "j": j,
                        "geom": ls,
                        "angle": ang
                    })
                continue

            inter = fi.poly2d.intersection(fj.poly2d)
            if isinstance(inter, Polygon) and inter.area > 0:
                Ai = fi.poly2d.area
                Aj = fj.poly2d.area
                if inter.area > 0.15 * min(Ai, Aj):
                    edges.append({
                        "type": "ridge_overlap",
                        "i": i, "j": j,
                        "geom": inter,
                        "angle": ang
                    })

    return edges


def drop_small_or_slim_faces(
    faces: List[FaceRec],
    min_area: float = 3.0,
    min_fill_ratio: float = 0.25
) -> List[FaceRec]:
    out = []
    for f in faces:
        A = f.poly2d.area
        if A < min_area:
            continue
        minx, miny, maxx, maxy = f.poly2d.bounds
        w = maxx - minx
        h = maxy - miny
        if w <= 0 or h <= 0:
            continue
        fill = A / (w * h)
        if fill < min_fill_ratio:
            continue
        out.append(f)
    return out


def apply_roof_rules(faces: List[FaceRec], edges: List[dict],
                     reg_lambda: float = 0.1) -> None:
    if not faces or not edges:
        return

    nF = len(faces)
    eqs = []  # (i, j, Ai, Aj, b, w)

    for e in edges:
        etype = e["type"]
        i, j = e["i"], e["j"]
        fi, fj = faces[i], faces[j]
        geom = e["geom"]

        if etype == "ridge_line" and isinstance(geom, LineString):
            if geom.length < EDGE_MIN_LEN:
                continue
            c0 = geom.coords[0]
            c1 = geom.coords[-1]
            cm = geom.interpolate(0.5, normalized=True).coords[0]
            sample_pts = [c0, c1, cm]
            base_weight = geom.length
        elif etype == "ridge_overlap" and isinstance(geom, Polygon):
            c = geom.representative_point()
            sample_pts = [(c.x, c.y)]
            base_weight = math.sqrt(geom.area + 1e-6)
        else:
            continue

        ni = fi.n
        nj = fj.n
        niz = ni[2]
        njz = nj[2]
        if abs(niz) < 1e-6 or abs(njz) < 1e-6:
            continue

        for (mx, my) in sample_pts:
            nix, niy = ni[0], ni[1]
            njx, njy = nj[0], nj[1]
            Ai = -njz
            Aj = niz
            b = -((niz * njx - njz * nix) * mx + (niz * njy - njz * niy) * my)
            w = max(base_weight, EDGE_MIN_LEN)
            eqs.append((i, j, Ai, Aj, b, w))

    if not eqs:
        return

    m = len(eqs)
    A = np.zeros((m + nF, nF), float)
    b = np.zeros(m + nF, float)

    for row_idx, (i, j, Ai, Aj, bi, w) in enumerate(eqs):
        ww = math.sqrt(max(w, 1e-6))
        A[row_idx, i] = Ai * ww
        A[row_idx, j] = Aj * ww
        b[row_idx] = bi * ww

    for k, f in enumerate(faces):
        row_idx = m + k
        A[row_idx, k] = reg_lambda
        b[row_idx] = reg_lambda * f.d

    try:
        d_new, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception as ex:
        logging.warning(f"LS ridge-løsning feilet: {ex}")
        return

    for k, f in enumerate(faces):
        f.d = float(d_new[k])
        f.rings3d = polygon_to_rings3d(f.poly2d, f.n, f.d)


def merge_coplanar_faces(
    faces: List[FaceRec],
    ang_thr: float = MERGE_ANG_DEG,
    d_thr: float = MERGE_DZ
) -> List[FaceRec]:
    if not faces:
        return faces
    out: List[FaceRec] = []
    used = [False] * len(faces)

    def plane_offset_diff(fi: FaceRec, fj: FaceRec) -> float:
        inter = fi.poly2d.intersection(fj.poly2d)
        if inter.is_empty:
            return float("inf")
        samples = []
        if isinstance(inter, Polygon):
            samples.append(inter.representative_point().coords[0])
        else:
            try:
                for g in inter.geoms[:5]:
                    if isinstance(g, Polygon):
                        samples.append(g.representative_point().coords[0])
            except Exception:
                pass
        if not samples:
            samples.append(fi.poly2d.representative_point().coords[0])
        diffs = []
        for (cx, cy) in samples[:5]:
            zi = z_on_plane(fi.n, fi.d, cx, cy)
            zj = z_on_plane(fj.n, fj.d, cx, cy)
            if np.isfinite(zi) and np.isfinite(zj):
                diffs.append(abs(zi - zj))
        return float(np.median(diffs)) if diffs else float("inf")

    for i, fi in enumerate(faces):
        if used[i]:
            continue
        group = [fi]
        for j, fj in enumerate(faces[i + 1:], i + 1):
            if used[j]:
                continue
            cosang = float(np.clip(np.dot(fi.n, fj.n), -1, 1))
            ang = math.degrees(math.acos(cosang))
            if ang > ang_thr:
                continue
            if not (fi.poly2d.intersects(fj.poly2d) or fi.poly2d.touches(fj.poly2d)):
                continue
            dz = plane_offset_diff(fi, fj)
            if dz > d_thr:
                continue
            group.append(fj)
            used[j] = True

        U = unary_union([g.poly2d for g in group])
        if not isinstance(U, Polygon):
            U = max(U.geoms, key=lambda p: p.area)

        U = U.buffer(0)
        rings3d = polygon_to_rings3d(U, fi.n, fi.d)
        out.append(FaceRec(poly2d=U, n=fi.n, d=fi.d, rings3d=rings3d))

    return out


def collapse_almost_coplanar_pair(
    faces: List[FaceRec],
    ang_thr: float = 3.0,
    dz_thr: float = 0.30
) -> List[FaceRec]:
    if len(faces) != 2:
        return faces

    f1, f2 = faces
    cosang = float(np.clip(np.dot(f1.n, f2.n), -1, 1))
    ang = math.degrees(math.acos(cosang))
    if ang > ang_thr:
        return faces

    U = unary_union([f1.poly2d, f2.poly2d])
    if U.is_empty or not isinstance(U, Polygon):
        return faces

    coords = list(U.exterior.coords)
    if len(coords) < 3:
        return faces
    sample_idx = np.linspace(0, len(coords) - 1, 8, dtype=int)
    samples = [coords[i] for i in sample_idx]

    dz_list = []
    for (x, y) in samples:
        z1 = z_on_plane(f1.n, f1.d, x, y)
        z2 = z_on_plane(f2.n, f2.d, x, y)
        if np.isfinite(z1) and np.isfinite(z2):
            dz_list.append(abs(z1 - z2))

    if not dz_list or np.median(dz_list) > dz_thr:
        return faces

    A1, A2 = f1.poly2d.area, f2.poly2d.area
    base = f1 if A1 >= A2 else f2
    poly = U.buffer(0)
    rings3d = polygon_to_rings3d(poly, base.n, base.d)
    return [FaceRec(poly2d=poly, n=base.n, d=base.d, rings3d=rings3d)]


def drop_nested_faces(faces: List[FaceRec], area_ratio: float = 0.9) -> List[FaceRec]:
    out = []
    for i, fi in enumerate(faces):
        poly_i = fi.poly2d
        Ai = poly_i.area
        remove = False
        for j, fj in enumerate(faces):
            if i == j:
                continue
            poly_j = fj.poly2d
            Aj = poly_j.area
            if Aj <= Ai:
                continue
            if poly_i.within(poly_j) and Ai < area_ratio * Aj:
                remove = True
                break
        if not remove:
            out.append(fi)
    return out


def _rms_on_points(face: FaceRec, P: np.ndarray) -> float:
    """
    RMS mellom plane (face.n, face.d) og punkter P (N x 3).
    Brukes bare lokalt i overlapps-regioner.
    """
    if len(P) == 0:
        return float("inf")
    n, d = face.n, face.d
    z_pred = (-d - n[0] * P[:, 0] - n[1] * P[:, 1]) / n[2]
    return float(np.sqrt(np.mean((P[:, 2] - z_pred) ** 2)))



def resolve_overlaps_by_rms(
    faces: List[FaceRec],
    P_xyz: np.ndarray,
    min_overlap_area: float = 0.5,
    min_pts_region: int = 30
) -> List[FaceRec]:
    """
    Fjerner overlapp mellom takflater:
      - Finn overlapps-polygon mellom to flater
      - Bruk punkter i overlapps-området til å beregne RMS for hver flate
      - Flaten med dårligst RMS taper overlappet (klippes med difference)
      - Hvis diff gir flere polygoner -> lag flere faces av samme plan

    P_xyz: alle punkter (x,y,z) inne i footprinten for dette bygget.
    """
    if len(faces) < 2 or len(P_xyz) == 0:
        return faces

    pts2d = shapely.points(P_xyz[:, :2])

    changed = True
    while changed:
        changed = False
        nF = len(faces)
        if nF < 2:
            break

        # Vi må kunne bygge lista på nytt når vi splitter / fjerner
        for i in range(nF):
            fi = faces[i]
            for j in range(i + 1, nF):
                fj = faces[j]

                inter = fi.poly2d.intersection(fj.poly2d)
                if inter.is_empty:
                    continue

                # Håndter Polygon / MultiPolygon / GeometryCollection
                if isinstance(inter, Polygon):
                    inter_polys = [inter]
                else:
                    try:
                        inter_polys = [
                            g for g in inter.geoms if isinstance(g, Polygon)
                        ]
                    except Exception:
                        inter_polys = []

                if not inter_polys:
                    continue

                # Bruk største overlapps-polygon som representant
                inter_poly = max(inter_polys, key=lambda g: g.area)

                if inter_poly.area < min_overlap_area:
                    continue

                # Punkter i overlapps-området
                mask = shapely.contains(inter_poly, pts2d)
                P_reg = P_xyz[mask]
           
                # Hvis for få punkt -> fall tilbake til areal-basert prioritering
                if len(P_reg) < min_pts_region:
                    keep_i = fi.poly2d.area >= fj.poly2d.area
                else:
                    rms_i = _rms_on_points(fi, P_reg)
                    rms_j = _rms_on_points(fj, P_reg)
                    keep_i = rms_i <= rms_j

                winner_idx, loser_idx = (i, j) if keep_i else (j, i)
                loser = faces[loser_idx]

                diff = loser.poly2d.difference(inter)
                new_faces: List[FaceRec] = []

                if diff.is_empty:
                    # taperen forsvinner helt
                    pass
                else:
                    if isinstance(diff, Polygon):
                        polys = [diff]
                    else:
                        polys = [g for g in diff.geoms if isinstance(g, Polygon)]

                    for poly in polys:
                        poly = poly.buffer(0)
                        if poly.is_empty or poly.area < MIN_FACE_AREA:
                            continue
                        rings3d = polygon_to_rings3d(poly, loser.n, loser.d)
                        new_faces.append(
                            FaceRec(poly2d=poly, n=loser.n, d=loser.d, rings3d=rings3d)
                        )

                # Bygg ny faces-liste: vi fjerner taperen og legger inn evt. nye biter
                faces = [f for k, f in enumerate(faces) if k != loser_idx]
                faces.extend(new_faces)

                changed = True
                break  # restart pga endret liste
            if changed:
                break

    return faces


# ===== Hoved-funksjon: plane-first + model-driven =====

def extract_roof_planes(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, footprint: Polygon,
) -> List[FaceRec]:
    allP = np.c_[x, y, z]

    pts2d = shapely.points(allP[:, :2])
    mask_fp = shapely.contains(footprint, pts2d)
    P = allP[mask_fp]
    if len(P) < MIN_INLIERS:
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    if VOXEL_SIZE and VOXEL_SIZE > 0:
        pcd = pcd.voxel_down_sample(VOXEL_SIZE)

    nn = pcd.compute_nearest_neighbor_distance()
    avg_nn = float(np.mean(nn)) if len(nn) else RANSAC_DIST
    dist_thr = max(RANSAC_DIST, 2.5 * avg_nn)
    min_inliers_eff = max(MIN_INLIERS, int(MIN_INLIERS_REL * len(pcd.points)))

    faces: List[FaceRec] = []
    faces2d_for_cov: List[Polygon] = []
    work = pcd
    max_planes_eff = MAX_PLANES

    for k in range(max_planes_eff + 10):
        if len(work.points) < min_inliers_eff:
            break

        plane_model, inliers = work.segment_plane(
            distance_threshold=dist_thr,
            ransac_n=RANSAC_N,
            num_iterations=RANSAC_ITERS
        )
        if len(inliers) < min_inliers_eff:
            break

        inlier_cloud = work.select_by_index(inliers)
        work = work.select_by_index(inliers, invert=True)
        P_in = np.asarray(inlier_cloud.points)

        n, d, sigma, resid = fit_plane_ls_irls(P_in)
        if not is_roofish(n[2]):
            continue

        thr = max(2.5 * sigma, 0.02)
        keep = np.abs(resid) <= thr
        P_in = P_in[keep]
        if len(P_in) < min_inliers_eff:
            continue

        hull = plane_local_hull(P_in, n)
        if hull.is_empty:
            hull = hull2d(P_in[:, :2])
            if hull.is_empty:
                hull = alpha_shape_polygon(P_in[:, :2], alpha=adaptive_alpha_q(len(P_in)))
                if hull.is_empty:
                    continue

        poly2d_int = hull.intersection(footprint)
        if poly2d_int.is_empty:
            continue

        if isinstance(poly2d_int, Polygon):
            polys = [poly2d_int]
        else:
            polys = [g for g in poly2d_int.geoms if isinstance(g, Polygon)]

        new_any = False
        for F in polys:
            F = make_valid(F).buffer(0)
            if F.is_empty or not isinstance(F, Polygon):
                continue

            Fq = F.buffer(-adaptive_inner_clip(F.area))
            if Fq.is_empty or not isinstance(Fq, Polygon):
                Fq = F

            if Fq.area < MIN_FACE_AREA:
                continue

            Fq = simplify_poly(Fq, tol=0.20)

            rings3d = polygon_to_rings3d(Fq, n, d)
            faces.append(FaceRec(poly2d=Fq, n=n, d=d, rings3d=rings3d))

            faces2d_for_cov.append(Fq)
            new_any = True

        cov_now = compute_coverage(faces2d_for_cov, footprint)

        if not new_any:
            if cov_now < 0.90 and k < max_planes_eff + 6:
                max_planes_eff += 1
                continue
            else:
                break

        if cov_now >= 0.96:
            break

        if not faces:
            return []

    # 5) koplanar-merge
    faces = merge_coplanar_faces(faces, ang_thr=MERGE_ANG_DEG, d_thr=MERGE_DZ)

    # 5b) enkel saltak-spesialregel
    faces = refine_simple_gable(faces, footprint)
    force_simple_gable_ridge_height(faces, footprint)

    # 6) model-driven ridge-regel
    edges = classify_edges(faces)
    apply_roof_rules(faces, edges)

    # 7) nested + små/slanke flater
    faces = drop_nested_faces(faces)
    faces = drop_small_or_slim_faces(faces)

    # 7b) fjern overlappende takflater basert på RMS i overlapps-soner
    faces = resolve_overlaps_by_rms(faces, P)

 
    # 8) snapping mot footprint + hjørner
    snapped_faces: List[FaceRec] = []
    for f in faces:
        outer = list(f.poly2d.exterior.coords)[:-1]

        outer_snapped = snap_ring_to_footprint(outer, footprint)
     

        holes = [list(r.coords)[:-1] for r in f.poly2d.interiors]

        poly_snapped = Polygon(outer_snapped, holes)
        if poly_snapped.is_empty or not poly_snapped.is_valid:
            snapped_faces.append(f)
            continue

        poly_snapped = simplify_poly(poly_snapped, tol=0.15)
        rings3d = polygon_to_rings3d(poly_snapped, f.n, f.d)
        snapped_faces.append(
            FaceRec(poly2d=poly_snapped, n=f.n, d=f.d, rings3d=rings3d)
        )

    faces = snapped_faces

    # 8b) FJERN OVERLAPP ETTER SNAPPING
    #    – nå jobber vi på de endelige konturene
    faces = resolve_overlaps_by_rms(
        faces,
        P,
        min_overlap_area=0.05,   # litt mer aggressiv
        min_pts_region=20
    )

    # 9) Collaps very simple 2-face cases
    faces = collapse_almost_coplanar_pair(faces)

    return faces

    
