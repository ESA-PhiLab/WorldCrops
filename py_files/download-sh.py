# %%
import pandas as pd
from uuid import uuid4
import geopandas as gpd

from sentinelhub.time_utils import parse_time
from sentinelhub import FisRequest, BBox, Geometry, CRS, WcsRequest, CustomUrlParam, DataCollection, HistogramType

from geodataframefilter import *
from credentials import *

# %%


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


# %%
time_interval = ('2016-02-01', '2016-05-01')
centralasia = gpd.read_file(
    "/Volumes/Untitled 1/CropTypes2.0/data/cropdata/CentralAsia/CAWa_CropType_samples.shp")

print("GPD INFO:", centralasia.describe())

# filter out small fields < 3 hectare and Multipolygons
gpd_filtered = GeodataFrameFilter(centralasia, 30000, True)
gpd_filtered = gpd_filtered.filter()

# print(gpd_filtered.describe())

gpd_filtered = gpd_filtered.head(2)

# %%
L1C_df = pd.DataFrame()
L2A_df = pd.DataFrame()
gpd_filtered['id'] = gpd_filtered.index.to_series().map(lambda x: uuid4().hex)

for (idx, row) in gpd_filtered.iterrows():

    fis_request_L1C = FisRequest(
        data_collection=DataCollection.SENTINEL2_L1C,
        layer='AGRICULTURE_L1C',
        geometry_list=[Geometry((row.geometry), CRS.WGS84)],
        time=time_interval,
        resolution='10m',
        data_folder='./data',
        config=config
    )

    fis_request_L2A = FisRequest(
        data_collection=DataCollection.SENTINEL2_L1C,
        layer='AGRICULTURE_L2A',
        geometry_list=[Geometry((row.geometry), CRS.WGS84)],
        time=time_interval,
        resolution='10m',
        data_folder='./data',
        config=config
    )

    # Takes about 30s, to avoid redownloading we are saving results
    fis_data = fis_request_L1C.get_data(save_data=True)
    fis_data2 = fis_request_L2A.get_data(save_data=True)

    df = fis_data_to_dataframe(fis_data)
    df2 = fis_data_to_dataframe(fis_data2)

    field_df = add_cloud_info(df)
    field_df['id'] = row.id
    field_df2 = add_cloud_info(df2)
    field_df2['id'] = row.id

    L1C_df = pd.concat([L1C_df, field_df], axis=0)
    L2A_df = pd.concat([L2A_df, field_df2], axis=0)


# save geodataframe with id
# channel 0: clouds, channel 1: NDVI, channel 2: NDWI
# channel 4-15 Bands
gpd_filtered.to_file("CentralAsia/CAWa_CropType_samples_applied.shp")
# save merged dataframe with same id
L1C_df.reset_index(drop=True, inplace=True)
L1C_df.to_excel('CentralAsia/CAWa_CropType_samples_id_S2_L1C.xlsx')

L2A_df.reset_index(drop=True, inplace=True)
L2A_df.to_excel('CentralAsia/CAWa_CropType_samples_id_S2_L2A.xlsx')


# %%


# %%
# %%

# %%

# %%

# %%
