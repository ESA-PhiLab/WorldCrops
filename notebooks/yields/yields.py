# %%
# Demo for visualisation of crop type and yield data
import json
import os
from datetime import datetime
from functools import partial

import geojson
import geopandas as gpd
#!pip install matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
#3D stuff
from IPython.core.display import HTML, display
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import NullFormatter, StrMethodFormatter
from shapely.geometry import Point  # Point class
from shapely.geometry import \
    shape  # shape() is a function to convert geo objects through the interface
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

sns.set_theme()
sns.set_style("darkgrid")

import json
import warnings

import numpy as np
import rasterio as rio
from pyproj import Transformer

warnings.filterwarnings("ignore")

tqdm.pandas()

# %%
fields =  ['Baumacker', 'D8', 'Dichtlacker', 'Heindlacker', 'Heng', 
                            'Holzacker', 'Neulandsiedlung','Itzling2', 'Itzling5', 
                            'Itzling6', 'Schluetterfabrik', 'Thalhausen138', 
                            'Thalhausen141', 'Voettingerfeld']

#fields =  ['Dichtlacker', 'Heindlacker', 'Heng', 
#                            'Holzacker', 'Neulandsiedlung','Itzling5', 
#                            'Itzling6', 'Schluetterfabrik', 'Thalhausen138', 'Voettingerfeld']

test_fields = ['Baumacker', 'Itzling2', 'Thalhausen141' ]

field_summary = pd.read_excel("../data/cropdata/Bavaria/yields/fields_summary.xlsx")
yields_2018 = pd.read_csv("../data/cropdata/Bavaria/yields/yields2018.csv")
yields_df = yields_2018.copy()
yields_df2 = yields_2018.copy()

bands = ["B04", "B05", "B06", "B07", "B08", "B8A", \
                "B09", "B11", "B12"]
angles = ['solar_zenith', 'observer_zenith', 'relative_azimuth']
other_features = ["et0", "rain", "cum_rain"]
feature_cols = bands + other_features
target_col = "Ertr.masse (Nass)(tonne/ha)"

# %%
fields = yields_df.Name.unique().tolist()

# %%
conversion = 1

    
def getYieldwithoutBorders(group):
    #print(group['Name'].values[0])
    _fieldname = group['Name'].values[0]
    geo_df = gpd.GeoDataFrame.from_file('../data/cropdata/Bavaria/yields/FeldstueckeTUM/Feldstuecke_WGS84.shp')
    geo_df = geo_df[geo_df.Name_new == _fieldname ]
    geo_df2 = geo_df.buffer(-0.00004,resolution=1)
    #put Lon, Lat from dataframe to GeoDataFrame
    geometry = [Point(xy) for xy in zip(group.Longitude, group.Latitude)]
    crs = {'init': 'epsg:4326'}
    #schneide group mit felddaten
    gdf = gpd.GeoDataFrame(group, crs=crs, geometry=geometry)
    mask = gdf.geometry.within(geo_df2.geometry.unary_union)
    newdata = gdf.loc[mask]
    #ertrag cut als einzelwert fürs feld schreiben
    group['Ertrag_wBorders'] = newdata['Ertr.masse (Nass)(tonne/ha)'].sum()*conversion / newdata['Ertr.masse (Nass)(tonne/ha)'].shape[0]
    return group

yields_df = yields_df.groupby(['Name']).apply(getYieldwithoutBorders)
yields_df.reset_index(drop=True,inplace=True)

# %%
# helper functions
# from https://github.com/ADA-research/AutoML4HybridEarthScienceModels

def extract_date_from_url(url):
    idx = url.find("TIME=")
    return url[idx+5:idx+15]

def map_to_degrees(x):
    if x<0:
        x = 360+x
    else:
        x = x
    return x

def filter_by_2std(mean, std, target, data):
    condition = mean + 2*std
    condition2 = mean - 2*std
    return data[(data[target] < condition) & (data[target] > condition2)]

def drop_unnamed_columns(df):
    """
    When saving/loading .csv files, the index is often saved as an unnamed column.
    This function removes any unnamed columns.
    
    Args:
        df (pd DataFrame): input DataFrame
    """
    
    return df.loc[:, ~df.columns.str.contains('^Unnamed')]

def create_pixelwise_S2_data(yields_df, fields, path):
    s2_cols = ["CLM", "dataMask", "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", \
               "B8A", "B09", "B11", "B12", "solar_azimuth", "solar_zenith", \
               "observer_azimuth", "observer_zenith", "unknown"]
    
    data = []
    for field in tqdm(fields):
        yield_data = yields_df[yields_df["Name"]==field][["Latitude", "Longitude", "Ertr.masse (Nass)(tonne/ha)", "ErtragNass"]]
        
        for img_dir in os.listdir(os.path.join(path, field)):
            # Read satellite image with rasterio                                             
            src = rio.open(os.path.join(path, field, img_dir, "response.tiff"), mode="r+")
            # Extract image time from json request                                             
            msg = json.loads(open(os.path.join(path, field, img_dir, "request.json")).read())
            img_date = extract_date_from_url(msg["url"])
            
            # Get reflectance values per pixel
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", authority="EPSG")
            yield_data["x"], yield_data["y"] = transformer.transform(yield_data["Latitude"], yield_data["Longitude"])
            
            s2_data = list(rio.sample.sample_gen(src, yield_data[["x", "y"]].values))
            try:
                temp_df = pd.DataFrame(s2_data, columns=s2_cols).drop_duplicates().join(yield_data.reset_index())
                
                temp_df["relative_azimuth"] = (temp_df["solar_azimuth"] - temp_df["observer_azimuth"])\
                                            .apply(map_to_degrees)
                
                temp_df["date"] = img_date
                temp_df["Name"] = field
                data.append(temp_df)
            except Exception as e:
                print(e)
                print("Failed to extract reflectance values from: {}".format(os.path.join(path, field, img_dir, "response.tiff")))
            
    data = pd.concat(data)                                           
    return data

def resample_and_merge_data(sat_df, et0_df, frequency="W"):
    """
    Creates a weekly or monthly resampled dataset from satellite data and rain/et0 data
    
    Args:
        sat_df (pd DataFrame): S2A reflectance data
        et0_df (pd DataFrame): rain/et0 data
        frequency (str): "W" for weekly or "M" for monthly
    """
    
    sat_df["date"] = pd.to_datetime(sat_df["date"])
    et0_df["date"] = pd.to_datetime(et0_df["date"])
    
    # Filter by cloud mask
    sat_df = sat_df[sat_df["CLM"]==0]
    
    # Resample reflectance data to frequency
    sat_df = sat_df.groupby("index").resample(frequency, on="date").mean().interpolate()
    sat_df = sat_df.reset_index("date")
    
    et0_df = et0_df[["date", "et0", "rain", "cum_rain"]].drop_duplicates()
    
    # Resample et0 data to frequency, starting at the same date as sat_df
    # Maybe it would be better to use a sum/mean over time for et0 and rain instead of resampling
    et0_df = et0_df.resample(frequency, on="date", origin=sat_df["date"].min()).mean().interpolate()
    et0_df = et0_df.reset_index("date")
    
    df = sat_df.merge(et0_df, left_on="date", right_on="date")
    df = drop_unnamed_columns(df)
    
    return df

def invert_rtm(rtm_df, model, hyperparams, feature_cols, target_col="lai", do_cv=True):
        
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model(**hyperparams))])
    # Normally you would fit hyperparameters separately, 
    # for now just show cv score here to get an idea of inversion performance
    if do_cv:
        results = cross_validate(pipeline, X=rtm_df[feature_cols], y=rtm_df[target_col], cv=5,
                                scoring=('r2', 'neg_mean_squared_error'),
                                return_train_score=True)
        
        display("Inversion for {}".format(target_col))
        display("Mean train R2: {}, individual folds: {}".format(np.mean(results["train_r2"]), results["train_r2"]))
        display("Mean test R2: {}, individual folds: {}\n".format(np.mean(results["test_r2"]), results["test_r2"]))
        
    pipeline.fit(rtm_df[feature_cols], rtm_df[target_col])
    
    return pipeline

def create_dataset(bands, yields_df, fields, should_create_files=True, include_rtm=False, frequency="W"):
    data_path = "."
    if should_create_files:
        # To create locally:
        
        sat_images_path = "../data/cropdata/Bavaria/yields/sat_images_10m/"

        #yields_df = pd.read_csv(os.path.join(data_path, "../datayields2018.csv"))
        fields_of_interest =  fields
        
        sat_df = create_pixelwise_S2_data(yields_df, fields_of_interest, sat_images_path)
        # S2 values are scaled by a factor 10000
        sat_df[bands] = sat_df[bands] / 10000
        et0_df = pd.read_excel(os.path.join("../data/cropdata/Bavaria/yields/satellite_data_orginal.xlsx"))

        df = resample_and_merge_data(sat_df, et0_df, frequency)

    else:
        # To simply load files that were already created:
        filename = "reflectance_per_pixel_weekly_10m_rtm.csv" \
                    if frequency=="W" else "reflectance_per_pixel_monthly_10m_rtm.csv"
        df = pd.read_csv(os.path.join(data_path, filename))

    if include_rtm:
        # For now use a similar simple model setup for RTM inversion
        rf = RandomForestRegressor
        hyperparams = {
            "n_jobs":-1, 
            "n_estimators":300,
            "max_depth":100,
            "max_features":'sqrt',
            "random_state":984
            }

        include_angles = True

        angles = ['solar_zenith', 'observer_zenith', 'relative_azimuth']
        features = bands+angles if include_angles else bands

        lai_model = invert_rtm(rtm_df, rf, hyperparams, feature_cols = features, target_col="lai")
        cm_model = invert_rtm(rtm_df, rf, hyperparams, feature_cols = features, target_col="cm")
        cab_model = invert_rtm(rtm_df, rf, hyperparams, feature_cols = features, target_col="cab")


        df["lai"] = lai_model.predict(df[features])
        df["cm"] = cm_model.predict(df[features])
        df["cab"] = cab_model.predict(df[features])

    return df

def flatten_time_series(df, feature_cols, target_col):
    """
    Flattens a dataset for use in a supervised model. Not suitable for recurrent models.
    
    Args:
        df (pd DataFrame):
        feature_cols (list of str): Feature column names
        target_col (str):

    return
        df (pd DataFrame):
        feature_cols (list of str): New feature column names with 
                                    suffix for each timestep, 
                                    e.g. _t-5 for 5 weeks/months before last timestep
    """

    out_df = []
    for field_index in df["index"].unique():
        sub_df = df[df["index"]==field_index]
        n_timesteps = len(sub_df)
        cols = list(np.array([[col+"_t-{}".format(i) for col in feature_cols] for i in reversed(range(n_timesteps))]).flatten())
        ts_df = pd.DataFrame(sub_df[feature_cols].values.flatten()).T
        ts_df.columns = cols
        ts_df[target_col] = sub_df.iloc[0][target_col]
        out_df.append(ts_df)
    return pd.concat(out_df).interpolate(), cols






# %%
df = create_dataset(bands=bands, yields_df=yields_df, fields=fields)
out_df, feature_cols = flatten_time_series(df, feature_cols, "Ertr.masse (Nass)(tonne/ha)")

mean = out_df['Ertr.masse (Nass)(tonne/ha)'].mean()
std = out_df['Ertr.masse (Nass)(tonne/ha)'].std()

out_df = filter_by_2std(mean, std,'Ertr.masse (Nass)(tonne/ha)', out_df )




# %%
#Train RF with pixels and apply
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

rf = RandomForestRegressor(n_jobs=4, n_estimators=100)

target_col = "Ertr.masse (Nass)(tonne/ha)"
# feature_cols = list(range(0, 326))

results = cross_validate(rf, X=out_df[feature_cols], y=out_df[target_col], cv=cv,
                         scoring=('r2', 'neg_mean_squared_error'),
                         return_train_score=True)

display("Mean train R2: {}, individual folds: {}".format(np.mean(results["train_r2"]), results["train_r2"]))
display("Mean test R2: {}, individual folds: {}".format(np.mean(results["test_r2"]), results["test_r2"]))



