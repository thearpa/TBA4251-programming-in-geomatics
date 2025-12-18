from dataclasses import dataclass
from typing import List, Tuple, Dict
from shapely.geometry import Polygon


@dataclass
class FaceGeom:
    """Simple holder for one polygon surface."""
    rings3d: List[List[Tuple[float, float, float]]]  # [outer, hole1, ...]


class CityGMLWriter:
    def __init__(self, srs_epsg: int = 25832):
        self.srs_epsg = srs_epsg
        # per building: {"roofs": [...], "walls": [...]}
        self.buildings: Dict[str, Dict[str, List[FaceGeom]]] = {}
        self._poly_counter = 0

    # small helpers

    @staticmethod
    def _ring_area2_xy(ring: List[Tuple[float, float, float]]) -> float:
        """Signed area in XY (for orientation)."""
        if len(ring) < 3:
            return 0.0
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        area2 = 0.0
        n = len(xs)
        for i in range(n):
            j = (i + 1) % n
            area2 += xs[i] * ys[j] - ys[i] * xs[j]
        return area2 / 2.0

    def _ensure_closed_oriented_ring(
        self, ring: List[Tuple[float, float, float]], outer: bool
    ) -> List[Tuple[float, float, float]]:
        """Close ring + orient outer CCW, inner CW in XY."""
        if len(ring) < 3:
            return ring
        r = list(ring)
        if r[0] != r[-1]:
            r.append(r[0])
        area = self._ring_area2_xy(r)
        if outer and area < 0:
            r = list(reversed(r))
        if not outer and area > 0:
            r = list(reversed(r))
        return r

    @staticmethod
    def _format_poslist(ring: List[Tuple[float, float, float]]) -> str:
        """x y z x y z ..."""
        coords = []
        for (x, y, z) in ring:
            coords.append(f"{x:.3f} {y:.3f} {z:.3f}")
        return " ".join(coords)

    # roof handling 

    def _make_roof_geoms(
        self,
        faces_rings3d: List[List[List[Tuple[float, float, float]]]]
    ) -> List[FaceGeom]:
        geoms: List[FaceGeom] = []
        for rings in faces_rings3d:
            if not rings:
                continue

            normalized: List[List[Tuple[float, float, float]]] = []

            # outer
            outer = self._ensure_closed_oriented_ring(rings[0], outer=True)
            if len(outer) < 4:
                continue
            normalized.append(outer)

            # holes
            for h in rings[1:]:
                hole = self._ensure_closed_oriented_ring(h, outer=False)
                if len(hole) < 4:
                    continue
                normalized.append(hole)

            geoms.append(FaceGeom(rings3d=normalized))
        return geoms

    # wall generation from footprint + roofs 

    @staticmethod
    def _collect_roof_vertices(
        faces_rings3d: List[List[List[Tuple[float, float, float]]]]
    ) -> List[Tuple[float, float, float]]:
        vertices = []
        for rings in faces_rings3d:
            for ring in rings:
                vertices.extend(ring)
        return vertices

    @staticmethod
    def _nearest_roof_z(
        x: float,
        y: float,
        roof_vertices: List[Tuple[float, float, float]],
        default_height: float
    ) -> float:
        if not roof_vertices:
            return default_height
        best_d2 = float("inf")
        best_z = default_height
        for (rx, ry, rz) in roof_vertices:
            dx = x - rx
            dy = y - ry
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_z = rz
        return best_z

    def _make_wall_geoms_from_footprint(
        self,
        footprint: Polygon,
        faces_rings3d,
        ground_z: float,
        default_wall_height: float = 3.0
    ):
        if footprint is None or footprint.is_empty:
            return []

        coords = list(footprint.exterior.coords)
        if len(coords) < 4:
            return []

        roof_vertices = self._collect_roof_vertices(faces_rings3d)
        default_top = ground_z + default_wall_height

        geoms = []

        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]

            z1_top = self._nearest_roof_z(x1, y1, roof_vertices, default_top)
            z2_top = self._nearest_roof_z(x2, y2, roof_vertices, default_top)

            ring = [
                (x1, y1, ground_z),
                (x2, y2, ground_z),
                (x2, y2, z2_top),
                (x1, y1, z1_top),
                (x1, y1, ground_z),
            ]

            geoms.append(FaceGeom(rings3d=[ring]))

        return geoms


    #  public API 

    def add_building(
        self,
        bid: str,
        faces_rings3d: List[List[List[Tuple[float, float, float]]]],
        footprint: Polygon,
        ground_z: float,
        default_wall_height: float = 3.0
    ):
        """
        Legger til et bygg med tak- og veggflater.

        faces_rings3d: takflater fra rekonstruksjonen.
        footprint: 2D footprint-polygon.
        ground_z: typisk zmin for byggets punktsky.
        """
        roofs = self._make_roof_geoms(faces_rings3d)
        walls = self._make_wall_geoms_from_footprint(
            footprint, faces_rings3d, ground_z, default_wall_height
        )

        if not roofs and not walls:
            return

        self.buildings[bid] = {
            "roofs": roofs,
            "walls": walls
        }

    #  XML generation 

    def _build_polygon_xml(
        self, poly_id: str, geom: FaceGeom
    ) -> str:
        srs = f"urn:ogc:def:crs:EPSG::{self.srs_epsg}"
        outer = geom.rings3d[0]
        ou_pos = self._format_poslist(outer)

        xml = []
        xml.append(f'          <gml:Polygon gml:id="{poly_id}" srsName="{srs}">')
        xml.append(f'            <gml:exterior>')
        xml.append(f'              <gml:LinearRing>')
        xml.append(f'                <gml:posList>{ou_pos}</gml:posList>')
        xml.append(f'              </gml:LinearRing>')
        xml.append(f'            </gml:exterior>')

        # holes
        for hole in geom.rings3d[1:]:
            ho_pos = self._format_poslist(hole)
            xml.append(f'            <gml:interior>')
            xml.append(f'              <gml:LinearRing>')
            xml.append(f'                <gml:posList>{ho_pos}</gml:posList>')
            xml.append(f'              </gml:LinearRing>')
            xml.append(f'            </gml:interior>')

        xml.append(f'          </gml:Polygon>')
        return "\n".join(xml)

    def _build_multisurface_xml(
        self, bid: str, geoms: List[FaceGeom], surf_kind: str
    ) -> str:
        """
        surf_kind: 'RoofSurface' or 'WallSurface'
        """
        if not geoms:
            return ""

        xml = []
        xml.append('    <bldg:boundedBy>')
        xml.append(f'      <bldg:{surf_kind}>')
        xml.append('        <bldg:lod2MultiSurface>')
        xml.append('          <gml:MultiSurface>')

        for idx, geom in enumerate(geoms):
            self._poly_counter += 1
            pid = f"{bid}_{surf_kind.lower()}_{idx}"
            xml.append('            <gml:surfaceMember>')
            xml.append(self._build_polygon_xml(pid, geom))
            xml.append('            </gml:surfaceMember>')

        xml.append('          </gml:MultiSurface>')
        xml.append('        </bldg:lod2MultiSurface>')
        xml.append(f'      </bldg:{surf_kind}>')
        xml.append('    </bldg:boundedBy>')
        return "\n".join(xml)

    def _build_building_xml(self, bid: str, data: Dict[str, List[FaceGeom]]) -> str:
        roofs = data.get("roofs", [])
        walls = data.get("walls", [])

        xml = []
        xml.append('  <core:cityObjectMember>')
        xml.append(f'    <bldg:Building gml:id="{bid}">')

        if roofs:
            xml.append(self._build_multisurface_xml(bid, roofs, "RoofSurface"))
        if walls:
            xml.append(self._build_multisurface_xml(bid, walls, "WallSurface"))

        xml.append('    </bldg:Building>')
        xml.append('  </core:cityObjectMember>')
        return "\n".join(xml)

    def to_xml(self) -> str:
        header = """<?xml version="1.0" encoding="UTF-8"?>
<core:CityModel
    xmlns:gml="http://www.opengis.net/gml"
    xmlns:core="http://www.opengis.net/citygml/2.0"
    xmlns:bldg="http://www.opengis.net/citygml/building/2.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="
      http://www.opengis.net/citygml/2.0 http://schemas.opengis.net/citygml/2.0/cityGMLBase.xsd
      http://www.opengis.net/citygml/building/2.0 http://schemas.opengis.net/citygml/building/2.0/building.xsd">
"""
        parts = [header]
        for bid, data in self.buildings.items():
            parts.append(self._build_building_xml(bid, data))
        parts.append("</core:CityModel>")
        return "\n".join(parts)

    def write(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_xml())
