from ..utils.registration import ModuleRegistry
import os
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')

model_registry = ModuleRegistry()

def register(id, **kwargs):
    return model_registry.register(id, **kwargs)

def make(id, **kwargs):
    return model_registry.make(id, **kwargs)

def spec(id):
    return model_registry.spec(id)

def list_registered_modules():
    return list(model_registry.module_specs.keys())

register(
    id="glm",
    entry_point="airquality.models.stationary:StationaryModel",
    default_config_path=os.path.join(config_path, 'glm.yaml')
)

register(
    id="glm_reg",
    entry_point="airquality.models.stationary:StationaryModel",
    default_config_path=os.path.join(config_path, 'glm.yaml'),
    backend=dict(
        id='glm',
        scale='dev',
        regularized=True,
        regularized_kwargs=dict(
            method='elastic_net',
            L1_wt=0.5,
            alpha=0.01,
            refit=True,
        )
    )
)

register(
    id="glm_single_variable",
    entry_point="airquality.models.stationary:StationaryModel",
    default_config_path=os.path.join(config_path, 'glm.yaml'),
    optimizer=dict(id='single_variable')
)

register(
    id="glm_stlsq",
    entry_point="airquality.models.stationary:StationaryModel",
    default_config_path=os.path.join(config_path, 'glm_stlsq.yaml')
)

register(
    id="glm_stlsq_reg",
    entry_point="airquality.models.stationary:StationaryModel",
    default_config_path=os.path.join(config_path, 'glm_stlsq.yaml'),
    backend=dict(
        id='glm',
        scale='dev',
        regularized=True,
        regularized_kwargs=dict(
            method='elastic_net',
            L1_wt=0.25,
            alpha=1e-4,
            refit=True,
        )
    )
)

register(
    id="poisson",
    entry_point="airquality.models.stationary:StationaryModel",
    default_config_path=os.path.join(config_path, 'poisson.yaml')
)

register(
    id="poisson_stlsq",
    entry_point="airquality.models.stationary:StationaryModel",
    default_config_path=os.path.join(config_path, 'poisson_stlsq.yaml')
)