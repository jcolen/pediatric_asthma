from ..utils.registration import ModuleRegistry
import os
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')

data_registry = ModuleRegistry()

def register(id, **kwargs):
    return data_registry.register(id, **kwargs)

def make(id, **kwargs):
    return data_registry.make(id, **kwargs)

def spec(id):
    return data_registry.spec(id)

def list_registered_modules():
    return list(data_registry.module_specs.keys())

register(
    id="airquality_svi_daily",
    entry_point="airquality.datamodules.base:AirQualityDataModule",
    default_config_path=os.path.join(config_path, 'default.yaml'),
    exog=['NO2', 'SO2', 'CO', 'PM25', 'PM10', 
          'SVI_overall_perc', 'AverageDryBulbTemperature', 'AverageRelativeHumidity', 'TotalPrecipitation', 'log_covid',]
)

register(
    id="airquality_coi_daily",
    entry_point="airquality.datamodules.base:AirQualityDataModule",
    default_config_path=os.path.join(config_path, 'default.yaml'),
    exog=['NO2', 'SO2', 'CO', 'PM25', 'PM10', 
          'COI_overall_perc', 'AverageDryBulbTemperature', 'AverageRelativeHumidity', 'TotalPrecipitation', 'log_covid',]
)

register(
    id="pytorch_svi_daily",
    entry_point="airquality.datamodules.pytorch:TorchAQModule",
    default_config_path=os.path.join(config_path, 'default.yaml'),
    exog=['NO2', 'SO2', 'CO', 'PM25', 'PM10', 
          'SVI_overall_perc', 'AverageDryBulbTemperature', 'AverageRelativeHumidity', 'TotalPrecipitation', 'log_covid',],
    loader_kwargs={
        'batch_size': 256,
        'num_workers': 2,
        'pin_memory': True,
    },
)

register(
    id="pytorch_coi_daily",
    entry_point="airquality.datamodules.pytorch:TorchAQModule",
    default_config_path=os.path.join(config_path, 'default.yaml'),
    exog=['NO2', 'SO2', 'CO', 'PM25', 'PM10', 
          'COI_overall_perc', 'AverageDryBulbTemperature', 'AverageRelativeHumidity', 'TotalPrecipitation', 'log_covid',],
    loader_kwargs={
        'batch_size': 256,
        'num_workers': 2,
        'pin_memory': True,
    },
)