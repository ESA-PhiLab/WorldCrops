
![ScreenShot](/docs/crops.png) 

<video src=https://raw.githubusercontent.com/ESA-PhiLab/WorldCrops/main/docs/embeddings.mp4 width=100/>



## Environment setup for Python3

Create a new virtual environment with

    python3 -m venv env
    
activate it

    source env/bin/activate
    
install requirements

    pip3 install -r requirements.txt 
    cd src/
    python setup.py develop 
    (optional) pip install .
    import selfsupervised


## Add additional data

The script 'download.py' uses Sentinel Hub to download Sentinel data. More regions are welcome to be added. Configure your layer with Sentinel Hub and add your API keys to credentials.py. The download supports the FisRequest and SentinelHubStatistical from Sentinel Hub. 

	python download.py -f ../data/cropdata/CentralAsia/CAWa_CropType_samples.shp -t '{"2018": ["01-01-2018", "30-12-2018"],"2017": ["01-01-2017", "30-12-2017"],"2016": ["01-01-2016", "30-12-2016"]}' -o centralasia 


    
## new data sources

To include new data, a custom data set or data module must be defined for pytorch lightning.
(https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html)

## References

[1] Breizhcrops, https://breizhcrops.org <br/>
[2] Radiant Earth Foundation, https://www.radiant.earth/ <br/>
[3] Remelgado, R., Zaitov, S., Kenjabaev, S. et al. A crop type dataset for consistent land cover classification in Central Asia. Sci Data 7, 250 (2020). https://doi.org/10.1038/s41597-020-00591-2 <br/>
[4] Chair of Plant Nutrition, TUM, https://www.pe.wzw.tum.de/
