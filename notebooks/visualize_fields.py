# %%

from geodataframefilter import *
import pyproj
import shapely
from pyproj import Geod, CRS
import pandas as pd
from geomet import wkt
from shapely import wkt
import sys
from dateutil import parser
import shapefile
import matplotlib.pyplot as plt
import geopandas as gpd
import ee
import geojson
import salem
# %%

centralasia = gpd.read_file(
    "../data/cropdata/CentralAsia/CAWa_CropType_samples.shp")

durnast = gpd.read_file(
    "/Volumes/Untitled 1/CropTypes2.0/data/cropdata/Bavaria/Test_area.shp")


kenya1 = gpd.read_file(
    '/Volumes/Untitled 1/CropTypes2.0/data/cropdata/Kenya/ref_african_crops_kenya_01_labels/ref_african_crops_kenya_01_labels_00/labels.geojson')
kenya2 = gpd.read_file(
    '../data/cropdata/Kenya/ref_african_crops_kenya_01_labels/ref_african_crops_kenya_01_labels_01/labels.geojson')
kenya3 = gpd.read_file(
    '../data/cropdata/Kenya/ref_african_crops_kenya_01_labels/ref_african_crops_kenya_01_labels_02/labels.geojson')
kenya_merged = pd.concat([kenya1, kenya2, kenya3], axis=0)

bavaria = gpd.read_file(
    "/Volumes/Untitled 1/Sabine Erbe/BY_Antragsshape_2017/Nutzungsschlag_Oberbayern_2017.shp")


# load shapefiles
country = salem.read_shapefile(
    '../data/shapefiles/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
counties_france = salem.read_shapefile(
    '../data/shapefiles/FRA_adm/FRA_adm1.shp')
bretagne = counties_france[counties_france.NAME_1 == 'Bretagne']
uzbekistan = country[country.SOVEREIGNT == 'Uzbekistan']
kenya = country[country.SOVEREIGNT == 'Kenya']
bavaria = salem.read_shapefile('../data/shapefiles/bavaria/regbez_ex.shp')


# %%


# get the map from a predefined grid
grid = salem.mercator_grid(transverse=False, center_ll=(15., 28.),
                           extent=(12e6, 8e6))
smap = salem.Map(grid)

# Add the background (other resolutions include: 'mr', 'hr')
# smap.set_rgb(natural_earth='lr')
#smap.set_points(11, 48)
smap.set_shapefile(bavaria, facecolor='deepskyblue', linewidth=0)
smap.set_shapefile(bretagne, facecolor='blue', linewidth=0)
smap.set_shapefile(uzbekistan, facecolor='turquoise', linewidth=0)
smap.set_shapefile(kenya, facecolor='green', linewidth=0)
# done!
smap.set_scale_bar(location=(0.87, 0.04), add_bbox=True)
smap.visualize()
plt.savefig('crop_world.pdf')
plt.show()


# %%

kenya_merged['Year'] = 2019
# %%
# https://www.net-analysis.com/blog/cartopylayout.html


kenya_merged.to_file('Kenya/Kenya_labels_PlantVillage.shp')


# %%
kenya_merged.columns
# %%


gpd_filtered = GeodataFrameFilter(kenya_merged, 0, True)
gpd_filtered = gpd_filtered.filter()
gpd_filtered.describe()
# %%

# %%
kenya_merged.crs
# %%
durnast.crs
# %%
len(bavaria)
# %%
