# functions for yield prediction demo
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from pyproj import Transformer
from shapely.geometry import Point  # Point class
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

tqdm.pandas()

# Convert DataFrame to GeoDataFrame
def create_geometry(df):
    geometry = [Point(xy) for xy in zip(df.Longitude, df.Latitude)]
    return gpd.GeoDataFrame(df, crs={'init': 'epsg:4326'}, geometry=geometry)


def getYieldwithoutBorders(group, geo_df, conversion):
    _fieldname = group['Name'].values[0]
    field_geo = geo_df[geo_df.Name_new ==
                       _fieldname].buffer(-0.00004, resolution=1).geometry.unary_union

    mask = group.geometry.within(field_geo)
    newdata = group.loc[mask]

    if not newdata.empty:
        sum_yield = newdata['Ertr.masse (Nass)(tonne/ha)'].sum()
        avg_yield = sum_yield * conversion / newdata.shape[0]
    else:
        avg_yield = 0

    group['Ertrag_wBorders'] = avg_yield
    return group


def extract_date_from_url(url):
    return url[url.find("TIME=") + 5:url.find("TIME=") + 15]


def map_to_degrees(x):
    return x + 360 if x < 0 else x


def filter_by_2std(mean, std, target, data):
    upper_limit = mean + 2 * std
    lower_limit = mean - 2 * std
    return data[(data[target] < upper_limit) & (data[target] > lower_limit)]


def drop_unnamed_columns(df):
    return df.loc[:, ~df.columns.str.contains('^Unnamed')]


# https://github.com/ADA-research/AutoML4HybridEarthScienceModels
def create_pixelwise_S2_data(yields_df, fields, path):
    s2_cols = ["CLM", "dataMask", "B01", "B02", "B03", "B04", "B05", "B06", "B07",
               "B08", "B8A", "B09", "B11", "B12", "solar_azimuth", "solar_zenith",
               "observer_azimuth", "observer_zenith", "unknown"]

    data = []
    for field in tqdm(fields):
        yield_data = yields_df[yields_df["Name"] == field][
            ["Latitude", "Longitude", "Ertr.masse (Nass)(tonne/ha)", "ErtragNass"]]

        for img_dir in os.listdir(os.path.join(path, field)):
            try:
                with rio.open(os.path.join(path, field, img_dir, "response.tiff"), mode="r+") as src, \
                        open(os.path.join(path, field, img_dir, "request.json")) as req_file:
                    msg = json.load(req_file)
                    img_date = extract_date_from_url(msg["url"])

                    transformer = Transformer.from_crs(
                        "EPSG:4326", "EPSG:3857", authority="EPSG")
                    yield_data["x"], yield_data["y"] = transformer.transform(
                        yield_data["Latitude"], yield_data["Longitude"])

                    s2_data = list(rio.sample.sample_gen(
                        src, yield_data[["x", "y"]].values))
                    temp_df = pd.DataFrame(s2_data, columns=s2_cols).drop_duplicates().join(
                        yield_data.reset_index())
                    temp_df["relative_azimuth"] = (
                        temp_df["solar_azimuth"] - temp_df["observer_azimuth"]).apply(map_to_degrees)
                    temp_df["date"] = img_date
                    temp_df["Name"] = field
                    data.append(temp_df)
            except Exception as e:
                print(
                    f"Error processing {os.path.join(path, field, img_dir, 'response.tiff')}: {e}")

    return pd.concat(data)


def resample_and_merge_data(sat_df, et0_df, frequency="W"):
    sat_df["date"] = pd.to_datetime(sat_df["date"])
    et0_df["date"] = pd.to_datetime(et0_df["date"])

    sat_df = sat_df[sat_df["CLM"] == 0]
    sat_df = sat_df.groupby("index").resample(
        frequency, on="date").mean().interpolate().reset_index("date")
    et0_df = et0_df[["date", "et0", "rain", "cum_rain"]].drop_duplicates()
    et0_df = et0_df.resample(frequency, on="date", origin=sat_df["date"].min(
    )).mean().interpolate().reset_index("date")

    return drop_unnamed_columns(sat_df.merge(et0_df, on="date"))


def invert_rtm(rtm_df, model, hyperparams, feature_cols, target_col="lai", do_cv=True):
    pipeline = Pipeline([('scaler', StandardScaler()),
                        ('model', model(**hyperparams))])
    if do_cv:
        results = cross_validate(pipeline, X=rtm_df[feature_cols], y=rtm_df[target_col],
                                 cv=5, scoring=('r2', 'neg_mean_squared_error'), return_train_score=True)
        print(f"Inversion for {target_col}")
        print(
            f"Mean train R2: {np.mean(results['train_r2'])}, individual folds: {results['train_r2']}")
        print(
            f"Mean test R2: {np.mean(results['test_r2'])}, individual folds: {results['test_r2']}\n")

    pipeline.fit(rtm_df[feature_cols], rtm_df[target_col])
    return pipeline


def create_dataset(bands,
                   yields_df,
                   fields,
                   should_create_files=True,
                   include_rtm=False,
                   frequency="W"):
    data_path = "."
    if should_create_files:
        # To create locally:

        sat_images_path = "../data/cropdata/Bavaria/yields/sat_images_10m/"

        # yields_df = pd.read_csv(os.path.join(data_path, "../datayields2018.csv"))
        fields_of_interest = fields

        sat_df = create_pixelwise_S2_data(yields_df, fields_of_interest,
                                          sat_images_path)
        # S2 values are scaled by a factor 10000
        sat_df[bands] = sat_df[bands] / 10000
        et0_df = pd.read_excel(
            os.path.join(
                "../data/cropdata/Bavaria/yields/satellite_data_orginal.xlsx"
            ))

        df = resample_and_merge_data(sat_df, et0_df, frequency)

    else:
        # To simply load files that were already created:
        filename = "reflectance_per_pixel_weekly_10m_rtm.csv" \
            if frequency == "W" else "reflectance_per_pixel_monthly_10m_rtm.csv"
        df = pd.read_csv(os.path.join(data_path, filename))

    if include_rtm:
        # For now use a similar simple model setup for RTM inversion
        rf = RandomForestRegressor
        hyperparams = {
            "n_jobs": -1,
            "n_estimators": 300,
            "max_depth": 100,
            "max_features": 'sqrt',
            "random_state": 984
        }

        include_angles = True

        angles = ['solar_zenith', 'observer_zenith', 'relative_azimuth']
        features = bands + angles if include_angles else bands

        lai_model = invert_rtm(rtm_df,
                               rf,
                               hyperparams,
                               feature_cols=features,
                               target_col="lai")
        cm_model = invert_rtm(rtm_df,
                              rf,
                              hyperparams,
                              feature_cols=features,
                              target_col="cm")
        cab_model = invert_rtm(rtm_df,
                               rf,
                               hyperparams,
                               feature_cols=features,
                               target_col="cab")

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
        sub_df = df[df["index"] == field_index]
        n_timesteps = len(sub_df)
        cols = list(
            np.array([[col + "_t-{}".format(i) for col in feature_cols]
                      for i in reversed(range(n_timesteps))]).flatten())
        ts_df = pd.DataFrame(sub_df[feature_cols].values.flatten()).T
        ts_df.columns = cols
        ts_df[target_col] = sub_df.iloc[0][target_col]
        out_df.append(ts_df)
    return pd.concat(out_df).interpolate(), cols
