import os

from sentinelhub import SHConfig

################################################################
# Radiant Earth for dataset installation
################################################################

os.environ['MLHUB_API_KEY'] = ''

################################################################
# Sentinel Hub
################################################################

config = SHConfig()
config.instance_id = ''
config.sh_client_id = ''
config.sh_client_secret = ''

if config.instance_id == '':
    print(
        "Warning! To use FIS functionality, please configure the `instance_id`."
    )

# Configure your layer in the dashboard (configuration utility)
SHUB_LAYER_NAME1 = 'AGRICULTURE_L1C'
SHUB_LAYER_NAME2 = 'AGRICULTURE_L2A'

################################################################
