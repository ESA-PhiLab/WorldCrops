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
from download_helper import *
from sentinelhub import SentinelHubStatistical, DataCollection, CRS, BBox, bbox_to_dimensions, Geometry, SHConfig, parse_time, parse_time_interval, SentinelHubStatisticalDownloadClient

# %%


def download_via_shs(args):
    input_path = args.file
    output_path = args.output
    target = args.target

    #print(target, type(target))
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

    gpd_filtered = GeodataFrameFilter(_data, area, True)
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

    gpd_filtered.set_crs(epsg=4326, inplace=True)
    gpd_filtered.to_crs(epsg=3857, inplace=True)

    gpd_filtered = gpd_filtered.head(2)

    if 'year' not in gpd_filtered.columns:
        print('Year column in geodataframe does not exists..Please add column year')
        sys.exit()

    input_data_L2A = SentinelHubStatistical.input_data(
        DataCollection.SENTINEL2_L2A,
        maxcc=1
    )

    input_data_L1C = SentinelHubStatistical.input_data(
        DataCollection.SENTINEL2_L1C,
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

    # download and save S2 L2A from Sentinel Hub
    all_requests = []

    for (idx, row) in gpd_filtered.iterrows():

        if str(row.year) in year_list:
            time_interval = (
                str(timespan[str(row.year)][0]), str(timespan[str(row.year)][1]))
        else:
            continue

        aggregation = SentinelHubStatistical.aggregation(
            evalscript=evalscript,
            time_interval=time_interval,
            aggregation_interval='P1D',
            resolution=(10, 10)
        )

        request = SentinelHubStatistical(
            aggregation=aggregation,
            input_data=[input_data_L2A],
            geometry=Geometry((row.geometry), crs=CRS(gpd_filtered.crs)),
            calculations=histogram_calculations,
            config=config
        )
        #stats = request.get_data(redownload=True)[0]
        all_requests.append(request)

    download_requests = [_request.download_list[0]
                         for _request in all_requests]
    client = SentinelHubStatisticalDownloadClient(config=config)
    stats = client.download(download_requests)

    _dfs = [stats_to_df(polygon_stats) for polygon_stats in stats]

    # set target column for y values
    if target != 0:
        for df, crop_type, year, id in zip(_dfs, gpd_filtered[target].values, gpd_filtered.year.values, gpd_filtered.id.values):
            df['crop_type'] = crop_type
            df['year'] = year
            df['id'] = id

    L2A_df = pd.concat(_dfs)

    gpd_filtered.to_file(
        output_path + "/labels_new.shp")
    # save merged dataframe with same id
    L2A_df.reset_index(drop=True, inplace=True)
    L2A_df.to_excel(
        output_path + '/timeseries_L2A.xlsx')

    # download and save S2 L1C
    all_requests_L1C = []
    for (idx, row) in gpd_filtered.iterrows():

        if str(row.year) in year_list:
            time_interval = (
                str(timespan[str(row.year)][0]), str(timespan[str(row.year)][1]))
        else:
            continue

        aggregation = SentinelHubStatistical.aggregation(
            evalscript=evalscript,
            time_interval=time_interval,
            aggregation_interval='P1D',
            resolution=(10, 10)
        )

        request = SentinelHubStatistical(
            aggregation=aggregation,
            input_data=[input_data_L1C],
            geometry=Geometry((row.geometry), crs=CRS(gpd_filtered.crs)),
            calculations=histogram_calculations,
            config=config
        )
        #stats = request.get_data(redownload=True)[0]
        all_requests_L1C.append(request)

    download_requests = [_request.download_list[0]
                         for _request in all_requests_L1C]
    client = SentinelHubStatisticalDownloadClient(config=config)
    stats = client.download(download_requests)

    _dfs = [stats_to_df(polygon_stats) for polygon_stats in stats]

    # set target column for y values
    if target != 0:
        for df, crop_type, year, id in zip(_dfs, gpd_filtered[target].values, gpd_filtered.year.values, gpd_filtered.id.values):
            df['crop_type'] = crop_type
            df['year'] = year
            df['id'] = id

    L1C_df = pd.concat(_dfs)

    gpd_filtered.to_file(
        output_path + "/labels_new.shp")
    # save merged dataframe with same id
    L1C_df.reset_index(drop=True, inplace=True)
    L1C_df.to_excel(
        output_path + '/timeseries_L1C.xlsx')


def download_via_fis(args):

    input_path = args.file
    output_path = args.output
    target = args.target
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

    gpd_filtered = GeodataFrameFilter(_data, area, True)
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

    gpd_filtered.set_crs(epsg=4326, inplace=True)
    gpd_filtered.to_crs(epsg=3857, inplace=True)

    gpd_filtered = gpd_filtered.head(2)

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
            geometry_list=[
                Geometry((row.geometry), crs=CRS(gpd_filtered.crs))],
            time=time_interval,
            resolution='10m',
            data_folder=output_path + '/jsondata',
            config=config,
            maxcc=1
        )

        fis_request_L2A = FisRequest(
            data_collection=DataCollection.SENTINEL2_L2A,
            layer=credentials.SHUB_LAYER_NAME2,
            geometry_list=[
                Geometry((row.geometry), crs=CRS(gpd_filtered.crs))],
            time=time_interval,
            resolution='10m',
            data_folder=output_path + '/jsondata',
            config=config,
            maxcc=1
        )

        # channel 0: clouds, channel 1: NDVI, channel 2: NDWI
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

        if target != 0:
            field_df['crop_type'] = row[target]
            field_df2['crop_type'] = row[target]

        L1C_df = pd.concat([L1C_df, field_df], axis=0)
        L2A_df = pd.concat([L2A_df, field_df2], axis=0)
        print('.', sep=' ', end='', flush=True)

    # save geodataframe with id
    # channel 0 is in clouds columns (1 stands for cloud/ 0 no clouds)
    # channel 1: NDVI, channel 2: NDWI
    # Rest channels: Bands
    gpd_filtered.to_file(
        output_path + "/labels_new.shp")
    # save merged dataframe with same id
    L1C_df.reset_index(drop=True, inplace=True)
    L1C_df.to_excel(
        output_path + '/timeseries_L1C.xlsx')

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
        '-a', '--api', help='Choose Sentinel hub API: FisRequst or SentinelHubStatistical', type=str, default='FisRequest', required=True)

    parser.add_argument(
        '-y', '--target', help='add a target variable to satellite data', type=str, default=0)
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

    if args.api == 'FisRequest':
        download_via_fis(args)
    elif args.api == 'SentinelHubStatistical':
        download_via_shs(args)
    else:
        print('Please verify API name...should be FisRequest or SentinelHubstatistical.')


if __name__ == "__main__":
    main()
