import importlib
from typing import Any


def is_simple_literal(value) -> bool:
    return value is None or isinstance(value, (bool, int, float, complex, str))


def get_type(module_name: str, class_name: str) -> type:
    module = importlib.import_module(module_name)
    result = getattr(module, class_name)
    if not isinstance(result, type):
        raise TypeError(result)
    return result


class Configurable:

    @classmethod
    def build(cls, config: dict[str, Any]) -> 'Configurable':
        subclass = get_type(config['__module__'], config['__class__'])
        if subclass is not cls:
            assert issubclass(subclass, cls)
            return subclass.build(config)
        instance = cls()
        instance.configure(config)
        return instance

    def get_configuration(self) -> dict[str, Any]:
        config = dict(__module__=type(self).__module__, __class__=type(self).__name__)
        for property_name in dir(self):
            if property_name.startswith('_'):
                continue
            property_value = getattr(self, property_name)
            if is_simple_literal(property_value):
                config[property_name] = property_value
            elif isinstance(property_value, Configurable):
                config[property_name] = property_value.get_configuration()
        return config

    def configure(self, config: dict[str, Any]) -> None:
        assert config['__module__'] == type(self).__module__
        assert config['__class__'] == type(self).__name__
        for property_name in dir(self):
            if property_name not in config:
                continue
            property_default = getattr(self, property_name)
            property_override = config[property_name]
            if isinstance(property_default, Configurable):
                property_default.configure(property_override)
            elif is_simple_literal(property_default) and is_simple_literal(property_override):
                try:
                    setattr(self, property_name, property_override)
                except AttributeError:
                    pass
