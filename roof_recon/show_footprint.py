import fiona
import geopandas as gpd

# 1) List layers in the FGDB to inspect available datasets
fgdb = "Trondheim_fkb_bygning.gdb"
print(fiona.listlayers(fgdb))
# ['fkb_bygning_omrade', 'fkb_bygning_grense', 'fkb_bygning_posisjon', 'fkb_bygning_senterlinje']

# 2) Read building footprint layer
gdf = gpd.read_file(fgdb, layer="fkb_bygning_omrade")

# 3) Inspect dataset
print(gdf.crs)                  # Coordinate reference system
print(gdf.columns.tolist())     # Attribute fields
print(len(gdf), "buildings")

# 4) Clean data and reproject to UTM 32N (EPSG:25832)
gdf = gdf[gdf.geometry.notna()]
gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
gdf = gdf.to_crs("EPSG:25832")

# 5) Export as GeoPackage for LoD2 roof reconstruction
out = "footprints_25832.gpkg"
gdf.to_file(out, layer="buildings", driver="GPKG")
print("Written to:", out)
