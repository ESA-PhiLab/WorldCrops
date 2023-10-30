# %%
from uuid import uuid4

import credentials
import geopandas as gpd
import pandas as pd
from geodataframefilter import GeodataFrameFilter
from sentinelhub import CRS, DataCollection, FisRequest, Geometry
from sentinelhub.time_utils import parse_time

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
                row = [
                    int(channel[1:]),
                    parse_time(stat['date'], force_datetime=True)
                ]

                for column in COLUMNS[2:]:
                    row.append(stat['basicStats'][column])

                data.append(row)

    return pd.DataFrame(data, columns=COLUMNS).sort_values(['channel', 'date'])


def add_cloud_info(dataframe):
    ''' Add column with cloud infos for every observation'''

    df = dataframe.copy()
    # channel 0 stands for clouds //
    # (CLM: fraction of cloudy pixels per each observation)
    clouds = df[df.channel == 0][['date', 'max']]
    clouds.rename(columns={'max': 'clouds'}, inplace=True)
    clouds.set_index('date', inplace=True)
    df.set_index('date', inplace=True)
    # df[df.channel == 0].to_excel('clouds.xlsx')

    newdf = df.merge(clouds, left_index=True, right_index=True, how='outer')
    newdf.reset_index(inplace=True)
    newdf.rename({'index': 'date'}, inplace=True)
    newdf = newdf[newdf['channel'] != 0]
    newdf.sort_values(by=['channel', 'date'], inplace=True)
    newdf.reset_index(drop=True, inplace=True)

    return newdf


# %%

# Load data for central asia
centralasia = gpd.read_file("../data/cropdata/Bavaria/Test_area.shp")

print("GPD INFO:", centralasia.describe())
centralasia.to_crs(epsg=4326, inplace=True)
# filter out small fields < 1 hectare (10000) and Multipolygons
gpd_filtered = GeodataFrameFilter(centralasia, 0, True)
gpd_filtered = gpd_filtered.filter()
# %%
gpd_filtered.crs = "EPSG:3856"
print(gpd_filtered.crs)
gpd_filtered = gpd_filtered.head(1)
# %%

# Load data for kenya
kenya1 = gpd.read_file(
    '/Volumes/Untitled 1/CropTypes2.0/data/cropdata/Kenya/ref_african_crops_kenya_01_labels/ref_african_crops_kenya_01_labels_00/labels.geojson'
)
kenya2 = gpd.read_file(
    '../data/cropdata/Kenya/ref_african_crops_kenya_01_labels/ref_african_crops_kenya_01_labels_01/labels.geojson'
)
kenya3 = gpd.read_file(
    '../data/cropdata/Kenya/ref_african_crops_kenya_01_labels/ref_african_crops_kenya_01_labels_02/labels.geojson'
)
kenya_merged = pd.concat([kenya1, kenya2, kenya3], axis=0)

# %%
_tmp = pd.DataFrame()
year_list = ['2018']
timespan = {"2018": ["2018-01-01", "2018-12-30"]}

# %%
gpd_filtered['year'] = gpd_filtered['year'].astype(str)

for year in year_list:
    _tmp = pd.concat([_tmp, gpd_filtered[gpd_filtered.year == year]], axis=0)
gpd_filtered = _tmp
print(gpd_filtered.describe())

# %%
gpd_filtered.head()
# define y column
y = 'NC_ant'
# %%

L1C_df = pd.DataFrame()
L2A_df = pd.DataFrame()
time_interval = tuple()
gpd_filtered['id'] = gpd_filtered.index.to_series().map(lambda x: uuid4().hex)

gpd_filtered = gpd_filtered.head(2)

for (idx, row) in gpd_filtered.iterrows():

    if str(row.year) in year_list:
        time_interval = (str(timespan[str(row.year)][0]),
                         str(timespan[str(row.year)][1]))

    else:
        continue

    fis_request_L1C = FisRequest(
        data_collection=DataCollection.SENTINEL2_L1C,
        layer=credentials.SHUB_LAYER_NAME1,
        geometry_list=[Geometry(row.geometry, crs=CRS(gpd_filtered.crs))],
        time=time_interval,
        resolution='10m',
        data_folder='data/jsondata',
        config=credentials.config)

    print(row.geometry)
    fis_request_L2A = FisRequest(
        data_collection=DataCollection.SENTINEL2_L2A,
        layer=credentials.SHUB_LAYER_NAME2,
        geometry_list=[Geometry(row.geometry, crs=CRS(gpd_filtered.crs))],
        time=time_interval,
        resolution='10m',
        data_folder='data/jsondata',
        config=credentials.config)

    # channel 0: clouds, channel 1: dataMask, channel 2: NDVI, channel 3: NDWI
    fis_data = fis_request_L1C.get_data(redownload=True)
    fis_data2 = fis_request_L2A.get_data(redownload=True)

    df = fis_data_to_dataframe(fis_data)
    df2 = fis_data_to_dataframe(fis_data2)

    field_df = add_cloud_info(df)
    field_df['id'] = row.id
    field_df['year'] = row.year
    field_df['CropType'] = row[y]
    field_df2 = add_cloud_info(df2)
    field_df2['id'] = row.id
    field_df2['year'] = row.year
    field_df2['CropType'] = row[y]

    L1C_df = pd.concat([L1C_df, field_df], axis=0)
    L2A_df = pd.concat([L2A_df, field_df2], axis=0)

# save geodataframe with id
# channel 0 is in clouds columns
# channel 1: dataMask, channel 2: NDVI, channel 3: NDWI
# Rest channels: Bands
gpd_filtered.to_file("data/CAWa_CropType_filtered.shp")
# save merged dataframe with same id
L1C_df.reset_index(drop=True, inplace=True)
L1C_df.to_excel('data/CAWa_CropType_filtered_S2_L1C.xlsx')

L2A_df.reset_index(drop=True, inplace=True)
L2A_df.to_excel('data/CAWa_CropType_filtered_S2_L2A.xlsx')


# %%
