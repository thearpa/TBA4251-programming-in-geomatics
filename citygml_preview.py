import xml.etree.ElementTree as ET
import numpy as np



def preview_citygml_3d(gml_path: str):
    import xml.etree.ElementTree as ET
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ns = {
        "gml": "http://www.opengis.net/gml",
        "core": "http://www.opengis.net/citygml/2.0",
        "bldg": "http://www.opengis.net/citygml/building/2.0",
    }

    tree = ET.parse(gml_path)
    root = tree.getroot()

    rings = []
    for poly in root.findall(".//gml:Polygon", ns):
        pos = poly.find(".//gml:posList", ns)
        if pos is None or not pos.text:
            continue

        vals = list(map(float, pos.text.split()))
        coords = np.array(vals).reshape(-1, 3)
        rings.append(coords)

    if not rings:
        print("Ingen polygoner funnet")
        return

    pts = np.vstack(rings)

    # 🔥 lokal referanse
    x0, y0, z0 = pts[:,0].mean(), pts[:,1].mean(), pts[:,2].min()
    pts_local = pts - np.array([x0, y0, z0])

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    for ring in rings:
        ring_local = ring - np.array([x0, y0, z0])
        poly = Poly3DCollection(
            [ring_local],
            facecolor="lightblue",
            edgecolor="black",
            alpha=0.8
        )
        ax.add_collection3d(poly)

    # ✅ Zoom automatisk rundt geometrien
    margin = 10
    ax.set_xlim(np.min(pts_local[:,0]) - margin, np.max(pts_local[:,0]) + margin)
    ax.set_ylim(np.min(pts_local[:,1]) - margin, np.max(pts_local[:,1]) + margin)
    ax.set_zlim(0, np.max(pts_local[:,2]) + margin)

    ax.set_xlabel("Local Easting (m)")
    ax.set_ylabel("Local Northing (m)")
    ax.set_zlabel("Height (m)")

    ax.set_box_aspect([1,1,0.4])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    preview_citygml_3d("output/output_roof_recon/_all_roofs.gml")
