from airquality.utils.registration import ModuleRegistry

backend_registry = ModuleRegistry()

def register(id, **kwargs):
    return backend_registry.register(id, **kwargs)

def make(id, **kwargs):
    return backend_registry.make(id, **kwargs)

def spec(id):
    return backend_registry.spec(id)

def list_registered_modules():
    return list(backend_registry.module_specs.keys())

register(
    id="glm",
    entry_point="airquality.models.backends.glm:GeneralizedLinearModel",
)

register(
    id="gam",
    entry_point="airquality.models.backends.gam:GeneralizedAdditiveModel",
)

register(
    id="poisson",
    entry_point="airquality.models.backends.poisson:Poisson",
)