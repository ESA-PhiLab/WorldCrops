# %%
from uuid import uuid4
from geodataframefilter import *
from http.client import HTTPConnection
import logging
from shapely.ops import transform
from shapely.geometry import *
from sentinelhub import SentinelHubStatistical, DataCollection, CRS, BBox, bbox_to_dimensions, Geometry, SHConfig, parse_time, parse_time_interval, SentinelHubStatisticalDownloadClient
from credentials import *
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from collections import defaultdict
import datetime as dt
import json
from shapely.geometry import Polygon
from shapely import wkt

from sentinelhub import FisRequest, BBox, Geometry, CRS, WcsRequest, CustomUrlParam, DataCollection, HistogramType
%matplotlib inline


#HTTPConnection.debuglevel = 1

# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)
#equests_log = logging.getLogger("requests.packages.urllib3")
# requests_log.setLevel(logging.DEBUG)
#requests_log.propagate = True
# %%
config
# %%
area = 'POLYGON ((11.64184785604579 48.39218695695232, 11.64177309600206 48.39223816388215, 11.64162112423547 48.39317133965693, 11.64153349151354 48.39370983478911, 11.6439369994422 48.39370219203855, 11.6440564025146 48.39294139749138, 11.64420179710519 48.39201548241762, 11.64184785604579 48.39218695695232))'
area = wkt.loads(area)
time_interval = ("2018-01-01", "2018-12-30")

# %%
polygons_gdf = gpd.read_file('statapi_test.geojson')
polygons_gdf.crs
# %%
# Load data for central asia

print(area)

# %%

# %%
# helper
ndvi_evalscript = """
//VERSION=3

function setup() {
  return {
    input: [{
      bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "CLM", "CLP", "dataMask"],
      units: "REFLECTANCE"
    }],
    output: [
      {
        id: "bands",
        bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12","NDVI"],
        sampleType: "FLOAT32"
      },
      {
        id: "dataMask",
        bands: 1
      }]
  }
}

function evaluatePixel(samples) {
    // Normalised Difference Vegetation Index and variation
    let NDVI = index(samples.B08, samples.B04);

    // cloud probability normalized to interval [0, 1]
    let CLP = samples.CLP / 255.0;
     // masking cloudy pixels
    let combinedMask = samples.dataMask

    if (samples.CLM > 0) {
        combinedMask = 0;
    }

    return {
        bands: [samples.B01, samples.B02, samples.B03, samples.B04, samples.B05, samples.B06,
                samples.B07, samples.B08, samples.B8A, samples.B09, samples.B11, samples.B12, NDVI],
        masks: [samples.CLM],
        dataMask: [combinedMask]
    };
}

"""


def add_cloud_info(dataframe):
    ''' Add column with cloud infos for every observation'''

    df = dataframe.copy()
    # channel 0 stands for clouds // (CLM: fraction of cloudy pixels per each observation)
    clouds = df[df.channel == 0][['date', 'max']]
    clouds.rename(columns={'max': 'clouds'}, inplace=True)
    clouds.set_index('date', inplace=True)
    df.set_index('date', inplace=True)
    #df[df.channel == 0].to_excel('clouds.xlsx')

    newdf = df.merge(clouds, left_index=True, right_index=True, how='outer')
    newdf.reset_index(inplace=True)
    newdf.rename({'index': 'date'}, inplace=True)
    newdf = newdf[newdf['channel'] != 0]
    newdf.sort_values(by=['channel', 'date'], inplace=True)
    newdf.reset_index(drop=True, inplace=True)

    return newdf


def stats_to_df(stats_data):
    """ Transform Statistical API response into a pandas.DataFrame
    """
    df_data = []

    for single_data in stats_data['data']:
        df_entry = {}
        is_valid_entry = True

        df_entry['interval_from'] = parse_time(
            single_data['interval']['from']).date()
        df_entry['interval_to'] = parse_time(
            single_data['interval']['to']).date()

        for output_name, output_data in single_data['outputs'].items():
            for band_name, band_values in output_data['bands'].items():

                band_stats = band_values['stats']
                if band_stats['sampleCount'] == band_stats['noDataCount']:
                    is_valid_entry = False
                    break

                for stat_name, value in band_stats.items():
                    col_name = f'{output_name}_{band_name}_{stat_name}'
                    if stat_name == 'percentiles':
                        for perc, perc_val in value.items():
                            perc_col_name = f'{col_name}_{perc}'
                            df_entry[perc_col_name] = perc_val
                    else:
                        df_entry[col_name] = value

        if is_valid_entry:
            df_data.append(df_entry)

    return pd.DataFrame(df_data)


def fis_data_to_dataframe(fis_data):
    """ from Sentinel Hub examples; 
    Creates a DataFrame from list of FIS responses
    """
    COLUMNS = ['channel', 'date', 'min', 'max', 'mean', 'stDev']
    data = []

    for fis_response in fis_data:
        for channel, channel_stats in fis_response.items():
            for stat in channel_stats:
                row = [int(channel[1:]), parse_time(
                    stat['date'], force_datetime=True)]

                for column in COLUMNS[2:]:
                    row.append(stat['basicStats'][column])

                data.append(row)

    return pd.DataFrame(data, columns=COLUMNS).sort_values(['channel', 'date'])


# %%
_data = gpd.read_file(
    "../data/cropdata/Bavaria/Test_area.shp")
#_data.to_crs(epsg=3857, inplace=True)

_data.to_crs(epsg=4326, inplace=True)
_data['year'] = _data['year'].astype(str)
timespan = {"2018": ["2018-01-01", "2018-12-30"]}
year_list = list(timespan.keys())

gpd_filtered = GeodataFrameFilter(_data, 0, True)
gpd_filtered = gpd_filtered.filter()

#print('Length:', len(gpd_filtered))

# filter geodataframe with years
_tmp = pd.DataFrame()
for year in year_list:
    _tmp = pd.concat(
        [_tmp, gpd_filtered.loc[(gpd_filtered.year == year)]], axis=0)
gpd_filtered = _tmp

gpd_filtered['id'] = gpd_filtered.index.to_series().map(
    lambda x: uuid4().hex)

# %%
#gpd_filtered.to_crs(epsg=3857, inplace=True)
gpd_filtered.set_crs(epsg=4326, inplace=True)
gpd_filtered.to_crs(epsg=3857, inplace=True)

# %%
gpd_filtered = gpd_filtered.head(1)
# %%%%
gpd_filtered.geometry[0]
# %%%

for (idx, row) in gpd_filtered.iterrows():

    print(row.geometry)
    fis_request_L2A = FisRequest(
        data_collection=DataCollection.SENTINEL2_L2A,
        layer=SHUB_LAYER_NAME2,
        geometry_list=[Geometry(row.geometry, crs=CRS(gpd_filtered.crs))],
        time=time_interval,
        resolution='10m',
        data_folder='data/jsondata',
        config=config,
        maxcc=1
    )
    fis_data2 = fis_request_L2A.get_data(redownload=True)

df2 = fis_data_to_dataframe(fis_data2)
df2 = add_cloud_info(df2)
# %%
df2
# %%
df2 = df2[df2.clouds != 1]
df2[df2.channel == 2]['mean'].plot()

# %%
df2.head()
# %%
len(df2[df2.channel == 0])
# %%
centralasia = gpd.read_file(
    "../data/cropdata/Bavaria/Test_area.shp")

centralasia.to_crs(epsg=3857, inplace=True)
print("GPD INFO:", centralasia.describe())

gpd_filtered = centralasia.head(1)
area = gpd_filtered.geometry[0]

aggregation = SentinelHubStatistical.aggregation(
    evalscript=ndvi_evalscript,
    time_interval=time_interval,
    aggregation_interval='P1D',
    resolution=(10, 10),
    #size=(631, 1047)
)

input_data = SentinelHubStatistical.input_data(
    DataCollection.SENTINEL2_L2A,
    maxcc=1
)


histogram_calculations = {
    "ndvi": {
        "histograms": {
            "default": {
                "nBins": 20,
                "lowEdge": -1.0,
                "highEdge": 1.0
            }
        }
    }
}

ndvi_requests = []

request = SentinelHubStatistical(
    aggregation=aggregation,
    input_data=[input_data],
    geometry=Geometry(area, crs=CRS(gpd_filtered.crs)),
    calculations=histogram_calculations,
    config=config
)
stats = request.get_data(redownload=True)[0]


# %%
ndvi_df = [stats_to_df(polygon_stats) for polygon_stats in [stats]]
ndvi_df = pd.concat(ndvi_df)
# %%
fig, ax = plt.subplots(figsize=(15, 8))

series = ndvi_df

series.plot(
    ax=ax,
    x='interval_from',
    y='bands_NDVI_mean')

ax.fill_between(
    series.interval_from.values,
    series['bands_NDVI_mean'] - series['bands_NDVI_stDev'],
    series['bands_NDVI_mean'] + series['bands_NDVI_stDev'],
    alpha=0.3
)
# %%
ndvi_df.tail()
# %%
fig, ax = plt.subplots(figsize=(15, 8))

series = df2[df2.channel == 1]

series.plot(
    ax=ax,
    x='date',
    y='mean')

ax.fill_between(
    series.date.values,
    series['mean'] - series['stDev'],
    series['mean'] + series['stDev'],
    alpha=0.3
)
# %%
ndvi_df.head()
# %%
ndvi_df.set_index('interval_from', inplace=True)
ndvi_df.resample('W').mean()
# %%
ndvi_df['interval_from'] = pd.to_datetime(
    ndvi_df['interval_from'], format="%Y-%m-%d")
# %%
ndvi_df.reset_index(inplace=True)
# %%
ndvi_df.head()
# %%
test = df2[df2.channel == 1]

# %%
test[test['mean'] == 0]
# %%
ndvi_df.land_type
# %%
t
