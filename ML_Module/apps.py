from django.apps import AppConfig
from utils.WassabiClient import WasabiClient, wassabi_client, \
    set_wassabi_client


class MlModuleConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ML_Module'

    def ready(self):
        global wassabi_client  # Use the global client variable
        if wassabi_client is None:  # Initialize only once
            set_wassabi_client(WasabiClient())  # Create a new instance of the Client class
