import yaml

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

def update_nested_config(config, key, value):
    """ Overwrite nestd configuration file with dot-separated key-value pair
    """
    sub_keys = key.split('.')
    sub_dict = config
    for sub_key in sub_keys[:-1]:
        sub_dict = sub_dict[sub_key] # Will throw an error if the command line argument is wrong
    sub_dict[sub_keys[-1]] = yaml.safe_load(value)

    return config

def update_config_command_line(config, config_updates=None, config_updater=update_nested_config):
    """ Allow override of configuration values using command line arguments
            e.g. python driver.py pipeline_id pipeline_config_id key.to.set=value
    """
    if config_updates:
        for config_update in config_updates:
            try:
                key, value = config_update.split('=', 1) # Split at first instance of equals
                logger.info(f"Updating config with {key}={value}")
                config = config_updater(config, key, value)
            except Exception as e:
                logger.warning(f"Could not override {config_update}, skipping")
                logger.warning(f"\t{type(e).__name__}, Message: {e}")

    return config