import importlib
import logging
import yaml

logger = logging.getLogger("Registry")

def load(name):
    mod_name, attr_name = name.split(":")
    logger.info(f"Attempting to load {mod_name} with {attr_name}")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

def load_config(path):
    if path is None:
        return {}
    with open(path, 'r') as file:
        return yaml.safe_load(file)

class ModuleSpec(object):
    def __init__(self, id, entry_point=None, default_config_path=None, **kwargs):
        self.id = id
        self.entry_point = entry_point
        self.default_config = load_config(default_config_path)
        self.default_config.update(kwargs)

    def make(self, **kwargs):
        """Instantiates an instance of data module with appropriate kwargs"""
        if self.entry_point is None:
            logger.error(
                "Attempting to make deprecated module {}. \
                            (HINT: is there a newer registered version \
                            of this module?)".format(
                    self.id
                )
            )
            raise RuntimeError

        config = self.default_config.copy()
        config.update(kwargs)
        if callable(self.entry_point):
            gen = self.entry_point(**config)
        else:
            cls = load(self.entry_point)
            gen = cls(**config)

        return gen

class ModuleRegistry(object):
    def __init__(self):
        self.module_specs = {}

    def make(self, path, **kwargs):
        logger.info("Making new module: %s", path)
        module_spec = self.spec(path)
        module = module_spec.make(**kwargs)

        return module

    def all(self):
        return self.module_specs.values()

    def spec(self, path):
        if ":" in path:
            mod_name, _sep, id = path.partition(":")
            try:
                importlib.import_module(mod_name)
            except ImportError:
                logger.error(
                    "A module ({}) was specified for the module but was not found, \
                                make sure the package is installed with `pip install` before \
                                calling `module.make()`".format(
                        mod_name
                    )
                )
                raise

        else:
            id = path

        try:
            return self.module_specs[id]
        except KeyError:
            logger.error("No registered module with id: {}".format(id))
            raise

    def register(self, id, **kwargs):
        if id in self.module_specs:
            logger.error("Cannot re-register id: {}".format(id))
            raise RuntimeError
        self.module_specs[id] = ModuleSpec(id, **kwargs)


# Global  registry
module_registry = ModuleRegistry()

def register(id, **kwargs):
    return module_registry.register(id, **kwargs)

def make(id, **kwargs):
    return module_registry.make(id, **kwargs)

def spec(id):
    return module_registry.spec(id)

def list_registered_modules():
    return list(module_registry.module_specs.keys())