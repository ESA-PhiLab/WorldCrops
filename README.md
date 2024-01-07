

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

    python3 setup.py install
 
 Python example:
    
    ```python
    #import installed package
    import selfsupervised as ssl
    ```


## Data
The notebooks contain examples for loading yield data from the combine harvester and crop types. The script install_cropdata.sh in scripts/data/ installs several crop type data sources for domain adaptation experiments in different regions. 
For future experiments with crop types, there are now very interesting processed data sources such as EuroCrops[6].

## Paper SELF-SUPERVISED LEARNING
The paper used crop types data (time series) from Sentinel-2 for Bavaria to train a model using data from 2016 and 2017 and apply it to 2018. Experiments were also run with 5 and 10 percent data from 2018. 2018 has deviating climate conditions compared to 2016 and 2017. Two example python files for reproducing the results can be found under src/experiments/paper.

## Yields
The yield data at field level were part of the work Prediction of multi-year winter wheat yields and were documented and collected in detailed investigations[4]. The example script notebooks/demo_yields_data shows how to load yield data. This includes data from a combine harvester as well as time series for weather and Sentinel-2 data.    

## Citation
These data sets include yields and crop types and were introduced in following publications:

	@article{Data1,
	title = {Prediction of multi-year winter wheat yields at the field level with satellite and climatological data},
	journal = {Computers and Electronics in Agriculture},
	volume = {194},
	pages = {106777},
	year = {2022},
	issn = {0168-1699},
	doi = {https://doi.org/10.1016/j.compag.2022.106777},
	url = {https://www.sciencedirect.com/science/article/pii/S0168169922000941},
	author = {Michael Marszalek and Marco Körner and Urs Schmidhalter},
	}

	@article{Data2,
	author = {Marszalek, Michael and Saux, B. and Mathieu, P.-P and Nowakowski, Artur and Springer, Daniel},
	year = {2022},
	month = {05},
	pages = {1327-1333},
	title = {SELF-SUPERVISED LEARNING – A WAY TO MINIMIZE TIME AND EFFORT FOR PRECISION AGRICULTURE?},
	volume = {XLIII-B3-2022},
	journal = {ISPRS - International Archives of the Photogrammetry Remote Sensing and Spatial Information Sciences},
	doi = {10.5194/isprs-archives-XLIII-B3-2022-1327-2022}
	}

## References

[1] Breizhcrops, https://breizhcrops.org <br/>
[2] Radiant Earth Foundation, https://www.radiant.earth/ <br/>
[3] Remelgado, R., Zaitov, S., Kenjabaev, S. et al. A crop type dataset for consistent land cover classification in Central Asia. Sci Data 7, 250 (2020). https://doi.org/10.1038/s41597-020-00591-2 <br/>
[4] Chair of Plant Nutrition, TUM, https://www.pe.wzw.tum.de/ <br/>
[5] Lightly, https://www.lightly.ai/
[6] EuroCrops, https://www.eurocrops.tum.de/index.html
