import geopandas as gpd
import matplotlib.pyplot as plt

gdf = gpd.read_file("footprints_25832.gpkg", layer="buildings")

fig, ax = plt.subplots(figsize=(8, 8))
gdf.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=0.6)

subset = gdf.sample(10, random_state=1)

xmin, ymin, xmax, ymax = subset.total_bounds
pad = 30
north_offset = 100 
ax.set_xlim(xmin - pad, xmax + pad)
ax.set_ylim(ymin - pad + north_offset, ymax + pad + north_offset)


ax.set_xlabel("Easting [m]")
ax.set_ylabel("Northing [m]")
ax.set_title("Building footprints (ETRS89 / UTM 32N, EPSG:25832)")

plt.tight_layout()
plt.savefig("fig.png", dpi=300, bbox_inches="tight")
plt.show()
