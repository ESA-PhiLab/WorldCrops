# %%
import rasterio
import shapely.geometry
from sentinelhub import WmsRequest
from sentinelhub.constants import MimeType
from sentinelhub import SHConfig
from sentinelhub.time_utils import parse_time
from sentinelhub import FisRequest, BBox, Geometry, CRS, WcsRequest, CustomUrlParam, \
    DataCollection, HistogramType
from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataCollection, bbox_to_dimensions, DownloadRequest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import seaborn as sns
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import geopandas as gpd
import geojson
from shapely.geometry import Point
from credentials import *
# %%

# helper functions


def plot_image(image, factor=1, vmin=0, vmax=1):
    """
    Utility function for plotting RGB images.
    """
    fig = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))

    if np.issubdtype(image.dtype, np.floating):
        plt.imshow(np.minimum(image/10000 * factor, 1), vmin=vmin, vmax=vmax)
    else:
        plt.imshow(image, vmin=vmin, vmax=vmax)


# %%
yields = pd.read_csv('../data/cropdata/Bavaria/yields/yields2018.csv', sep=",",
                     encoding="ISO-8859-1", engine='python')
yields = yields[['Name', 'Latitude', 'Longitude', 'Elevation(m)', 'Ertr.masse (Nass)(tonne/ha)', 'Ertr.masse (Tr.)(tonne/ha)',
                 'Ertr.vol (Tr.)(L/ha)', 'ErtragNass', 'ErtragTr', 'Feuchtigkeit(%)', 'Jahr', 'TAG']]

f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14,
    ax15), (ax16, ax17, ax18), (ax19, ax20, ax21)) = plt.subplots(7, 3, figsize=(20, 40))
ax_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11,
           ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21]
size = 40
_tmp2 = yields
# Muehlacker rausgenommen obwohl WW damit bild quadratisch wird
gerste = ['Muehlacker', 'Grafenfeld', 'Krohberg', 'Radarstation', 'Sieblerfeld', 'Striegelfeld', 'Viehhausen1', 'Viehhausen11', 'Viehhausen3', 'Viehhausen5'
          ]
_tmp2 = _tmp2[~_tmp2.Name.isin(gerste)]

names = _tmp2['Name'].unique().tolist()
# print(len(names),names)
for i in range(len(names)):
    field = _tmp2[_tmp2.Name == names[i]]
    # print(i,names[i])
    geometry = [Point(xy) for xy in zip(field.Longitude, field.Latitude)]
    crs = {'init': 'epsg:4326'}
    gdf = gpd.GeoDataFrame(field, crs=crs, geometry=geometry)

    minx, miny, maxx, maxy = gdf.total_bounds
    # print(ax_list[i])
    ax_list[i].set_xlim(minx, maxx)
    ax_list[i].set_ylim(miny, maxy)
    ax_list[i].axes.get_xaxis().set_visible(False)
    ax_list[i].axes.get_yaxis().set_visible(False)
    txt = names[i]
    ax_list[i].set_title(txt, fontsize=size)
    ax_list[i].scatter(y=field.Latitude, x=field.Longitude, alpha=1, cmap=plt.get_cmap(
        "jet_r"), c=field['Ertr.masse (Nass)(tonne/ha)'], s=2.2)

# plt.xlabel("Test")
#plt.ylabel("common Y")

#cb_ax = f.add_axes([0.92, 0.05, 0.02, 0.9])
# cb_ax.tick_params(labelsize=40)

f.tight_layout()
# f.subplots_adjust(right=0.9)

plt.show()
# %%
# read test area
area = gpd.read_file(
    "../data/cropdata/Bavaria/Test_area.shp")
wheat_area = area[area.NC_ant == '115']

# read TUM field data
tum = gpd.read_file(
    '../data/cropdata/Bavaria/yields/FeldstueckeTUM/Feldstuecke_WGS84.shp')


fields_with_yields = ['Baumacker', 'D8', 'Dichtlacker', 'Heindlacker', 'Heng', 'Holzacker', 'Neulandsiedlung',
                      'Itzling2', 'Itzling5', 'Itzling6', 'Schluetterfabrik', 'Thalhausen138', 'Thalhausen141', 'Voettingerfeld']

# get the polygon of all considered fields.
filtered_fields = tum[tum.Name_new.isin(fields_with_yields)]
#filtered_fields = pd.concat([wheat_area, filtered_fields], ignore_index=True)

# %%
# one multipoly for the area with fields
tmp = filtered_fields.geometry.unary_union
gdf2 = gpd.GeoDataFrame(geometry=[tmp], crs=filtered_fields.crs)
allfields = shapely.geometry.box(*gdf2.total_bounds)
resolution = 10
field = BBox((allfields), CRS.WGS84)
field_size = bbox_to_dimensions(field, resolution=resolution)
time_interval = ('2018-02-01', '2018-05-01')

# %%
betsiboka_coords_wgs84 = [46.16, -16.15, 46.51, -15.58]
betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)


request = WmsRequest(
    data_collection=DataCollection.SENTINEL2_L1C,
    layer='AGRICULTURE_L2A_GAN',
    bbox=field,
    time=time_interval,
    width=512,
    image_format=MimeType.TIFF,
    config=config
)

data = request.get_data(redownload=True)

# %%
len(data[:])
# %%
plot_image(data[8]
           [:, :, [5, 4, 3]], 2.5, vmax=0.4)
# %%
# download all data


def download_field_data(name, polygon, timeinterval):
    resolution = 10
    field = BBox((polygon), CRS.WGS84)

    request = WmsRequest(
        data_collection=DataCollection.SENTINEL2_L1C,
        data_folder='GAN_data/'+name,
        layer='AGRICULTURE_L2A_GAN',
        bbox=field,
        custom_url_params={CustomUrlParam.GEOMETRY: polygon},
        time=time_interval,
        width=512,
        maxcc=0.1,
        image_format=MimeType.TIFF,
        config=config
    )
    dem_request = WmsRequest(
        data_collection=DataCollection.DEM,
        data_folder='GAN_data/'+name,
        layer='DEM',
        bbox=field,
        width=512,
        image_format=MimeType.TIFF,
        custom_url_params={CustomUrlParam.GEOMETRY: polygon},
        config=config
    )

    #data = request.get_data(redownload=True)
    #dem = dem_request.get_data(redownload=True)
    dem_request.save_data()
    request.save_data()
    # return data


time_interval = ('2018-02-01', '2018-09-01')
#filtered_fields = filtered_fields.head(1)

for idx, field in filtered_fields.iterrows():
    name = field.Name_new
    polygon = field.geometry
    download_field_data(name, polygon, time_interval)
    # imagedata.download_list[0]


# %%
len(imagedata)
# %%
plot_image(imagedata[3]
           [:, :, [5, 4, 3]], 2.5, vmax=0.4)

# %%
filtered_fields.plot()
# %%
imagedata[3].shape
# %%
os.listdir('GAN_data/')

# %%
with rasterio.open('GAN_data/Schluetterfabrik/ca01850b0bf761cc421b2d00b36d2061/response.tiff', 'r') as ds:
    arr = ds.read()  # read all raster values


# %%

# %%
arr = np.moveaxis(arr, 0, -1)
arr = np.moveaxis(arr, 1, 0)
# %%
plot_image(arr[:, :, [5, 4, 3]])
# %%


# %%
