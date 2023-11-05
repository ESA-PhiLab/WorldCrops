# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from credentials import config
from geodataframefilter import GeodataFrameFilter
from sentinelhub import (CRS, DataCollection, Geometry, SentinelHubStatistical,
                         SentinelHubStatisticalDownloadClient, parse_time)

# %%
# Load data for central asia
centralasia = gpd.read_file("../data/cropdata/Bavaria/Test_area.shp")

centralasia.to_crs(epsg=4326, inplace=True)
print("GPD INFO:", centralasia.describe())

# filter out small fields < 1 hectare and Multipolygons
gpd_filtered = GeodataFrameFilter(centralasia, 0, True)
gpd_filtered = gpd_filtered.filter()

gpd_filtered.set_crs(epsg=4326, inplace=True)
gpd_filtered.to_crs(epsg=3857, inplace=True)
gpd_filtered = gpd_filtered.head(2)
print(gpd_filtered)

yearly_time_interval = '2018-01-01', '2018-12-30'

# %%
len(gpd_filtered)
# %%

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

aggregation = SentinelHubStatistical.aggregation(
    evalscript=ndvi_evalscript,
    time_interval=yearly_time_interval,
    aggregation_interval='P1D',
    resolution=(10, 10))

input_data = SentinelHubStatistical.input_data(DataCollection.SENTINEL2_L2A,
                                               maxcc=1)

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

for geo_shape in gpd_filtered.geometry.values:
    request = SentinelHubStatistical(aggregation=aggregation,
                                     input_data=[input_data],
                                     geometry=Geometry(
                                         (geo_shape),
                                         crs=CRS(gpd_filtered.crs)),
                                     calculations=histogram_calculations,
                                     config=config)
    # rgb_stats = request.get_data(redownload=True)[0]
    ndvi_requests.append(request)

# %%
download_requests = [
    ndvi_request.download_list[0] for ndvi_request in ndvi_requests
]
client = SentinelHubStatisticalDownloadClient(config=config)
ndvi_stats = client.download(download_requests)

len(ndvi_stats)
# %%
ndvi_stats

# %%


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


ndvi_dfs = [stats_to_df(polygon_stats) for polygon_stats in ndvi_stats]

# %%
for df, land_type in zip(ndvi_dfs, gpd_filtered['NC_ant'].values,
                         gpd_filtered.year.values):
    df['land_type'] = land_type
ndvi_df = pd.concat(ndvi_dfs)
# %%
gpd_filtered.year.values
# gpd_filtered.year.values

# %%
ndvi_df.head()
# %%
fig, ax = plt.subplots(figsize=(15, 8))

for idx, land_type in enumerate(gpd_filtered['NC_ant'].values):
    series = ndvi_df[ndvi_df['land_type'] == land_type]

    series.plot(ax=ax,
                x='interval_from',
                y='bands_NDVI_mean',
                color=f'C{idx}',
                label=land_type)

    ax.fill_between(series.interval_from.values,
                    series['bands_NDVI_mean'] - series['bands_NDVI_stDev'],
                    series['bands_NDVI_mean'] + series['bands_NDVI_stDev'],
                    color=f'C{idx}',
                    alpha=0.3)

# %%
gpd_filtered.head()
# %%
ndvi_df['indices_NDVI_mean'].plot()
# %%
ndvi_df[ndvi_df['land_type'] == '592'].head()

# %%
# %%
fig, axs = plt.subplots(figsize=(12, 4))
ndvi_df['indices_NDVI_mean'].plot(ax=axs)
plt.savefig('sentinelhubstatistics3.png')

# %%
