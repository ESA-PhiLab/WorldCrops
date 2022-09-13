

<div align="center">
<img src="https://user-images.githubusercontent.com/11621580/177815096-f5936f2c-7942-4ebe-971a-38afbc2b5471.png" width="250" height="300" /> <img src="https://media.giphy.com/media/dFCkbzISh2IZWLdj7S/giphy.gif" width="250" height="300" /> <img src="https://user-images.githubusercontent.com/11621580/177831311-5aa2a302-3b85-4d8a-af2f-fe01a9531bf0.gif" width="300" height="300">
</div>

## Environment setup for Python3

Create a new virtual environment with

    python3 -m venv env
    
activate it

    source env/bin/activate
    
install requirements

    pip3 install -r requirements.txt 
    
change directory

    cd src/
    
install package  

    python setup.py develop 
    (optional) pip install .
 
 Python example:
    
    ```python
    #import installed package
    import selfsupervised as ssl
    #examples for a datamodule
    ssl.data
    ```


## Add additional data

The script 'download.py' uses Sentinel Hub to download Sentinel data. More regions are welcome to be added. Configure your layer with Sentinel Hub and add your API keys to credentials.py. The download supports the FisRequest and SentinelHubStatistical from Sentinel Hub. 

	python download.py -f ../data/cropdata/CentralAsia/CAWa_CropType_samples.shp -t '{"2018": ["01-01-2018", "30-12-2018"],"2017": ["01-01-2017", "30-12-2017"],"2016": ["01-01-2016", "30-12-2016"]}' -o centralasia 


    
## New data modules for SSL

To include new data, a custom data set or data module must be defined for pytorch lightning.
(https://pytorch-lightning.readthedocs.io/)

## References

[1] Breizhcrops, https://breizhcrops.org <br/>
[2] Radiant Earth Foundation, https://www.radiant.earth/ <br/>
[3] Remelgado, R., Zaitov, S., Kenjabaev, S. et al. A crop type dataset for consistent land cover classification in Central Asia. Sci Data 7, 250 (2020). https://doi.org/10.1038/s41597-020-00591-2 <br/>
[4] Chair of Plant Nutrition, TUM, https://www.pe.wzw.tum.de/ <br/>
[5] Lightly, https://www.lightly.ai/
