import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from .load_air_data import load_air_data
from .load_covid_data import load_covid_data
from .load_patient_visits import load_patient_visits
from .load_weather_data import load_weather_data
from .load_zcta_data import load_zcta_data, load_svi_data, load_coi_data

def data_load(config):
    dataframes = {}
    logger.info(f"Looking for data in {config['data_dir']}")

    logger.info(f"Loading patients from {config['patient_df']}")
    dataframes['patient_df'] = load_patient_visits(os.path.join(config['data_dir'], config['patient_df']))

    logger.info(f"Loading air data from {config['air_df']}")
    dataframes['air_df'] = load_air_data(os.path.join(config['data_dir'], config['air_df']))

    logger.info(f"Loading weather data from {config['weather_df']}")
    dataframes['weather_df'] = load_weather_data(os.path.join(config['data_dir'], config['weather_df']))

    logger.info(f"Loading COVID-19 data from {config['covid_df']}")
    dataframes['covid_df'] = load_covid_data(os.path.join(config['data_dir'], config['covid_df']))

    logger.info(f"Loading ZCTA population data from {config['zcta_df']}")
    dataframes['zcta_df'] = load_zcta_data(os.path.join(config['data_dir'], config['zcta_df']))

    logger.info(f"Loading Social Vulnerability data from {config['svi_df']}")
    dataframes['svi_df'] = load_svi_data(os.path.join(config['data_dir'], config['svi_df']))

    logger.info(f"Loading Child Opportunity Index from {config['coi_df']}")
    dataframes['coi_df'] = load_coi_data(os.path.join(config['data_dir'], config['coi_df']))

    return dataframes