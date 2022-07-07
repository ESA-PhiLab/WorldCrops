## Environment setup for Python3

Create a new virtual environment with

    python3 -m venv env
    
activate it

    source env/bin/activate
    
install requirements

    pip3 install -r requirements.txt 


## Selfsupervised learning packages

    cd src/
    python setup.py develop 
    (optional) pip install .

Run experiments using "import selfsupervised".
The root directory "config" holds yaml files with the corresponding configuration for 
pretraining or finetuning.
The "scripts" directory contains sh scripts that run the experiments defined in src/experiments
The package selfsupervised consists of the models (e.g. Transformer or SimSiam) 
In selfsupervised.data the data modules and custom data sets are defined.
Datamodules are used by pytorch lightning. "A datamodule is a shareable, reusable class that encapsulates all the steps needed to process data"
(https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html)


