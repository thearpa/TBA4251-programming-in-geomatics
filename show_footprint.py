import fiona
import geopandas as gpd

# 1) List lagene i FGDB for å se hva som finnes
fgdb = "Trondheim_fkb_bygning.gdb"
print(fiona.listlayers(fgdb))
# ['fkb_bygning_omrade', 'fkb_bygning_grense', 'fkb_bygning_posisjon', 'fkb_bygning_senterlinje']

# 2) Les inn footprint-laget 
gdf = gpd.read_file(fgdb, layer="fkb_bygning_omrade")

# 3) Undersøk dataene
print(gdf.crs)                  # viser koordinatsystemet
print(gdf.columns.tolist())     # viser feltnavn (f.eks. 'lokalId', 'bygningsnummer', 'gml_id')
print(len(gdf), "bygninger")

# 4) Rens dataene og reprojiser til UTM 32N (EPSG:25832)
gdf = gdf[gdf.geometry.notna()]
gdf = gdf[gdf.geometry.geom_type.isin(["Polygon","MultiPolygon"])]
gdf = gdf.to_crs("EPSG:25832")

# 5) Lagre som GeoPackage for bruk i videre LoD2-rekonstruksjon
out = "sample_roofdata_50/footprints_25832.gpkg"
gdf.to_file(out, layer="buildings", driver="GPKG")
print("Skrev til:", out)
