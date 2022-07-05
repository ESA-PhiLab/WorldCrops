import pandas as pd
import contextily as ctx
import matplotlib.pyplot as plt
import breizhcrops
import torch
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd
from breizhcrops import BreizhCrops
from breizhcrops.datasets.breizhcrops import BANDS as allbands
import os
from radiant_mlhub import Dataset, client
import tarfile
from pathlib import Path
from credentials import *


# download French data set with BreizhCrops
# https://colab.research.google.com/drive/1i0M_X5-ytFhF0NO-FjhKiqnraclSEIb0#scrollTo=4CVAlHY6OhAK

def raw_transform(input_timeseries):
    return input_timeseries


def france_download():
    regions = ["belle-ile", "frh01", "frh02", "rfh03", "frh04"]
    regions = [regions[0]]

    try:
        for _region in regions:
            BreizhCrops(region=_region, transform=raw_transform, level="L1C")
            BreizhCrops(region=_region, transform=raw_transform, level="L2A")
            #region_2A = BreizhCrops(region=_region, transform=raw_transform, level="L2A")

            # region_1C.download_geodataframe()
            # region_1C.download_h5_database()
        print('France downloaded')
    except Exception as e:
        pass


def kenya_download():
    dataset = Dataset.fetch('ref_african_crops_kenya_01')

    #print(f'ID: {dataset.id}')
    print(f'Title: {dataset.title}')
    # print('Collections:')
    for collection in dataset.collections:
        print(f'* {collection.id}')

    output_path = Path("./").resolve()
    for collection in dataset.collections:
        archive_path = output_path / f'{collection.id}.tar.gz'

        if archive_path.exists():
            print(f'Archive {archive_path} exists. Skipping.')
        else:
            print(f'Downloading {archive_path}...')
            collection.download(output_dir=output_path)

        print(f'Extracting {archive_path}...')
        with tarfile.open(archive_path) as tfile:
            tfile.extractall(path=output_path)

    print('Kenya downloaded')


def uganda_download():
    print('Uganda downloaded')


def ghana_download():
    print('Ghana downloaded')


_root = os.getcwd()
_root, filename = _root.rsplit('/', 1)
print('Root:', _root)

# create directory structure
dic_struct = [
    '/data/cropdata/France',
    '/data/cropdata/Kenya'
]

for path in dic_struct:
    print(path)
    try:
        os.mkdir(_root+path)
    except OSError as e:
        if e.errno != 17:
            print("Error:", e)


for path in dic_struct:
    os.chdir(_root + path)
    if path == dic_struct[0]:
        if len(os.listdir(_root + path)) == 0:
            print("Directory is empty")
            france_download()
        else:
            print("Directory is not empty....Skipping France download")

        os.chdir(_root)
    elif path == dic_struct[1]:
        if len(os.listdir(_root + path)) == 0:
            print("Directory is empty")

            kenya_download()

        else:
            _labels = Path(
                _root + path + '/ref_african_crops_kenya_01_labels.tar.gz')
            if not _labels.is_file():
                kenya_download()
            else:
                print("Directory is not empty....Skipping Kenya download")

        os.chdir(_root)
    else:
        print('No path in dictionary found')
