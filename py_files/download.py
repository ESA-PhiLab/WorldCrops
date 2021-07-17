# %%
import sys
import argparse
from pathlib import Path
import geopandas as gpd
import pandas as pd
from uuid import uuid4
import json
from sentinelhub.time_utils import parse_time
from sentinelhub import FisRequest, BBox, Geometry, CRS, DataCollection

from geodataframefilter import *
import credentials
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
    # df[df.channel == 0].to_excel('clouds.xlsx')

    newdf = df.merge(clouds, left_index=True, right_index=True, how='outer')
    newdf.reset_index(inplace=True)
    newdf.rename({'index': 'date'}, inplace=True)
    newdf = newdf[newdf['channel'] != 0]
    newdf.sort_values(by=['channel', 'date'], inplace=True)
    newdf.reset_index(drop=True, inplace=True)

    return newdf


def download_from_sh(args):

    input_path = args.file
    output_path = args.output
    # t1 = args.t1
    # t2 = args.t2
    multipoly = args.multipoly
    area = args.area
    # proc = args.proc
    # index = args.index
    timespan = json.loads(args.timespan)
    year_list = list(timespan.keys())

    # print(timespan)
    # check if input file exists
    if not Path(input_path).is_file():
        print('The file does not exist')
        sys.exit()

    # create output directory
    try:
        os.mkdir(output_path)
    except OSError as e:
        if e.errno != 17:
            print("Error:", e)

    L1C_df = pd.DataFrame()
    L2A_df = pd.DataFrame()
    time_interval = tuple()
    _data = gpd.read_file(input_path)

    # make sure data is in the correct format
    _data.to_crs(epsg=4326, inplace=True)
    _data['year'] = _data['year'].astype(str)

    gpd_filtered = GeodataFrameFilter(_data, area, timespan)
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

    #gpd_filtered = gpd_filtered.head(2)

    if 'year' not in gpd_filtered.columns:
        print('Year column in geodataframe does not exists..Please add column year')
        sys.exit()

    for (idx, row) in gpd_filtered.iterrows():

        if str(row.year) in year_list:
            time_interval = (
                str(timespan[str(row.year)][0]), str(timespan[str(row.year)][1]))
        else:
            continue

        fis_request_L1C = FisRequest(
            data_collection=DataCollection.SENTINEL2_L1C,
            layer=credentials.SHUB_LAYER_NAME1,
            geometry_list=[Geometry((row.geometry), CRS.WGS84)],
            time=time_interval,
            resolution='10m',
            data_folder=output_path + '/jsondata',
            config=config
        )

        fis_request_L2A = FisRequest(
            data_collection=DataCollection.SENTINEL2_L2A,
            layer=credentials.SHUB_LAYER_NAME2,
            geometry_list=[Geometry((row.geometry), CRS.WGS84)],
            time=time_interval,
            resolution='10m',
            data_folder=output_path + '/jsondata',
            config=config
        )

        # channel 0: clouds, channel 1: dataMask, channel 2: NDVI, channel 3: NDWI
        fis_data = fis_request_L1C.get_data(redownload=True)
        fis_data2 = fis_request_L2A.get_data(redownload=True)

        df = fis_data_to_dataframe(fis_data)
        df2 = fis_data_to_dataframe(fis_data2)

        # print(df.head())

        field_df = add_cloud_info(df)
        field_df['id'] = row.id
        field_df['year'] = row.year
        field_df2 = add_cloud_info(df2)
        field_df2['id'] = row.id
        field_df2['year'] = row.year

        L1C_df = pd.concat([L1C_df, field_df], axis=0)
        L2A_df = pd.concat([L2A_df, field_df2], axis=0)
        print('.', sep=' ', end='', flush=True)

    # save geodataframe with id
    # channel 0 is in clouds columns (1 stands for cloud/ 0 no clouds)
    # channel 1: dataMask, channel 2: NDVI, channel 3: NDWI
    # Rest channels: Bands
    gpd_filtered.to_file(
        output_path + "/labels_new.shp")
    # save merged dataframe with same id
    L1C_df.reset_index(drop=True, inplace=True)
    L1C_df.to_excel(
        output_path + '/timeseriesL1C.xlsx')

    L2A_df.reset_index(drop=True, inplace=True)
    L2A_df.to_excel(
        output_path + '/timeseries_L2A.xlsx')


def main(args_list=None):
    # parser for input parameters
    parser = argparse.ArgumentParser(
        description='Download Sentinel-2 NDVI, NDWI, Raw bands from Sentinel Hub. All products are based on L1C and L2A.')

    parser.add_argument(
        '-f', '--file', help='shapefile input file', type=str, required=True)
    parser.add_argument(
        '-o', '--output', help='output directory name e.g. -o test', type=str, required=True)
    parser.add_argument(
        '-t', '--timespan', help='JSON with year and timespan for every year e.g. "{"2016": ["01-01-2016", "01-12-2016"]}"', type=str, default='{"2016": ["01-01-2016", "01-2-2016"]}', required=True)

    parser.add_argument(
        '-mp', '--multipoly', help='Filter out multipolygons in shapefile', type=str, default=True)
    parser.add_argument(
        '-area', help='Filter out small polygons (square meter)', type=str, default=0)
    # parser.add_argument(
    #    '-t1', help='startDate (string): e.g. MM-DD-YYYY', type=str, default='01-01-2016')
    # parser.add_argument(
    #    '-t2', help='stopDate (string): e.g. 12-30', type=str, default='12-30')

    # parser.add_argument(
    #    '-proc', help='Processing Level Options: L1C or L2A', type=str, default='both')
    # parser.add_argument(
    #    '-index', help='Options: NDVI, NDWI, Raw bands', type=str, default='all')

    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    exec(open("credentials.py").read())
    print("Sentinel Hub Configuration", config)

    download_from_sh(args)


if __name__ == "__main__":
    main()
