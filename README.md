
![ScreenShot](/docs/crops.png) 





# Overview
This is an aggregation of different crop types and yield data in several regions of the world. The variance in the data should drive the development of new methods and different experiments. The merging of data represents only one part of the project. For each field polygon, a time series of mean values and other statistical properties was created. For example, a field (polygon) is represented by a Sentinel-2 time series of means per field.
The labels for the crop types come from different sources and have been summarised.


| Attempt | Fields  | Year | Level| Crops | Ref.|
| :-----: | :-: | :-: |  :-: | :-: |:-: |
| Bretagne | < 610T | 2017 | S2 L1C/L2A | barley, wheat, rapeseed, corn, sunflower, orchards, nuts, meadows| [[1]](#http://breizhcrops.org) |
| Bavaria | 2400 | 2016-2018 | S2 L1C/L2A| maize, wheet, rapeseed, barley, potato, sugar beet| |
| Kenya| ≈ 3000 | 2019 | S2 L1C/L2A | maize, cassava, common bean, soybean | [[2]](#https://mlhub.earth) |
| Uzbekistan | < 8196| 2016-2018 | S2 L1C/L2A| cotton, wheat, rice, maize, orchards, vineyards, alfalfa| [[3]](#https://www.nature.com/articles/s41597-020-00591-2)|

 
[<img src="https://github.com/ESA-PhiLab/WorldCrops/blob/main/docs/Data_sphere_small.mp4" width="20%">]()


## Environment setup for Python3

Create a new virtual environment with

    python3 -m venv env
    
activate it

    source env/bin/activate
    
install requirements

    pip3 install -r requirements.txt 


## Add additional data

The script 'download.py' uses Sentinel Hub to download Sentinel data. More regions are welcome to be added. Configure your layer with Sentinel Hub and add your API keys to credentials.py. The download supports the FisRequest and SentinelHubStatistical from Sentinel Hub. 

	python download.py -f ../data/cropdata/CentralAsia/CAWa_CropType_samples.shp -t '{"2018": ["01-01-2018", "30-12-2018"],"2017": ["01-01-2017", "30-12-2017"],"2016": ["01-01-2016", "30-12-2016"]}' -o centralasia


## Install

	cd py_files/

	python install_dataset.py

The script installs all crop type data. If the data is downloaded again, the data for France or Kenya should be deleted. The data for Bavaria and Uzbekistan have already been added.
The Kenya data comes from Radiant Earth which is why an api key is needed. All keys are summarized under 'credentials.py'. Only the MLHUB_API_KEY for https://mlhub.earth/ is needed! 
The French dataset is downloaded via https://breizhcrops.org/, which also offers some additional great features.


## References

[1] Breizhcrops, https://breizhcrops.org <br/>
[2] Radiant Earth Foundation, https://www.radiant.earth/ <br/>
[3] Remelgado, R., Zaitov, S., Kenjabaev, S. et al. A crop type dataset for consistent land cover classification in Central Asia. Sci Data 7, 250 (2020). https://doi.org/10.1038/s41597-020-00591-2 <br/>
[4] Chair of Plant Nutrition, TUM, https://www.pe.wzw.tum.de/
