# %%


from sentinelhub import SHConfig
from sentinelhub.time_utils import parse_time
from sentinelhub import FisRequest, BBox, Geometry, CRS, WcsRequest, CustomUrlParam, \
    DataCollection, HistogramType
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
filtered_fields = tum[tum.Name_new.isin(fields_with_yields)]
filtered_fields = pd.concat([wheat_area, filtered_fields], ignore_index=True)


# %%
filtered_fields.iloc[1].geometry
# %%
len(filtered_fields)
# %%
sahara_bbox = BBox((1.0, 26.0, 1.3, 25.7), CRS.WGS84)

# download images
config = SHConfig()
config.instance_id = '91e07825-ab2e-43d7-9b58-696f7d56974f'
config.sh_client_id = '21a60596-ad3f-40b3-b3cb-8f6ce4d0793a'
config.sh_client_secret = 'YMWx-%%o[ZibQ|&wB)DrjDsk@]DXMouZ:y[AAA<*'

if config.instance_id == '':
    print("Warning! To use FIS functionality, please configure the `instance_id`.")

# Configure your layer in the dashboard (configuration utility)
SHUB_LAYER_NAME1 = 'AGRICULTURE_L1C'
SHUB_LAYER_NAME2 = 'AGRICULTURE_L2A'
time_interval = ('2018-02-01', '2018-05-01')

wcs_request = WcsRequest(
    data_collection=DataCollection.SENTINEL2_L2A,
    layer=SHUB_LAYER_NAME2,
    bbox=BBox((filtered_fields.iloc[1].geometry), CRS.WGS84),
    time=time_interval,
    resx='10m',
    resy='10m',
    custom_url_params={CustomUrlParam.SHOWLOGO: False},
    config=config
)

images = wcs_request.get_data()

fig, axs = plt.subplots((len(images) + 2) // 3, 3, figsize=(10, 20))
for idx, (image, time) in enumerate(zip(images, wcs_request.get_dates())):
    axs.flat[idx].imshow(image)
    axs.flat[idx].set_title(time.date().isoformat())

fig.tight_layout()


# %%
