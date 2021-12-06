from sentinelhub import SHConfig
import os

################################################################
# Radiant Earth for dataset installation
################################################################

os.environ['MLHUB_API_KEY'] = '6fd273e083563324c317c41f4c3b6f63c46318f353317719716c8bdc0587c6c7'

################################################################
# Sentinel Hub
################################################################


config = SHConfig()
config.instance_id = '5e98cacf-5c35-4e3b-9674-52c8241a01f1'
config.sh_client_id = '93c443b0-b60d-4ba3-b8e2-f05b9d5c47ac'
config.sh_client_secret = ')!sd9B)JKEF?eemyWf*8U|93iPXo5F:#mmbw/YWM'


if config.instance_id == '':
    print("Warning! To use FIS functionality, please configure the `instance_id`.")


# Configure your layer in the dashboard (configuration utility)
SHUB_LAYER_NAME1 = 'AGRICULTURE_L1C'
SHUB_LAYER_NAME2 = 'AGRICULTURE_L2A'

################################################################
