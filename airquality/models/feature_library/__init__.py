import pysindy as ps
import numpy as np

def make_library(num_exog_features,
                 num_indicator_features,
                 polynomial_degree=1,
                 include_bias=True,
                 interaction_only=False,
                 include_interaction=True):
    # Build composite library
    libraries = []
    if polynomial_degree > 0:
        libraries.append(ps.PolynomialLibrary(degree=polynomial_degree, include_bias=include_bias, include_interaction=include_interaction, interaction_only=interaction_only))
    libraries.append(ps.IdentityLibrary()) # For indicator variables

    total_features = num_exog_features + num_indicator_features
    inputs_per_library = np.tile(np.arange(total_features, dtype=int), len(libraries))
    inputs_per_library = inputs_per_library.reshape([len(libraries), total_features])

    inputs_per_library[:-1, num_exog_features:] = 0
    inputs_per_library[-1:, :num_exog_features] = num_exog_features

    feature_library = ps.GeneralizedLibrary(
        libraries=libraries,
        inputs_per_library=inputs_per_library
    )
    return feature_library

from airquality.utils.registration import ModuleRegistry

library_registry = ModuleRegistry()

def register(id, **kwargs):
    return library_registry.register(id, **kwargs)

def make(id, **kwargs):
    return library_registry.make(id, **kwargs)

def spec(id):
    return library_registry.spec(id)

def list_registered_modules():
    return list(library_registry.module_specs.keys())

register(
    id="identity_library",
    entry_point="airquality.models.feature_library:make_library",
    kwargs=dict(
        polynomial_degree=1,
        include_bias=False,
    )
)

register(
    id="identity_library_with_constant",
    entry_point="airquality.models.feature_library:make_library",
    kwargs=dict(
        polynomial_degree=1,
        include_bias=True,
    )
)

register(
    id="polynomial_library",
    entry_point="airquality.models.feature_library:make_library",
)