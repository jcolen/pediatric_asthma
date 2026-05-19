from airquality.utils.registration import ModuleRegistry

optimizer_registry = ModuleRegistry()

def register(id, **kwargs):
    return optimizer_registry.register(id, **kwargs)

def make(id, **kwargs):
    return optimizer_registry.make(id, **kwargs)

def spec(id):
    return optimizer_registry.spec(id)

def list_registered_modules():
    return list(optimizer_registry.module_specs.keys())

register(
    id="base",
    entry_point="airquality.models.optimizers.base:BaseOptimizer",
)

register(
    id="stlsq",
    entry_point="airquality.models.optimizers.stlsq:STLSQ",
)

register(
    id="ensemble",
    entry_point="airquality.models.optimizers.ensemble:EnsembleOptimizer"
)

register(
    id="single_variable",
    entry_point="airquality.models.optimizers.variable_selection:SingleVariableSelection",
    p_threshold=0.05,
)