import datetime
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


def python_type_from_config(config: dict[str, Any]) -> type:
    if isinstance(config, type):
        return config
    assert config['__module__'] == type.__module__
    assert config['__class__'] == type.__name__
    return get_type(config['module'], config['name'])


def python_type_to_config(type_obj: type) -> dict[str, Any]:
    assert isinstance(type_obj, type)
    return dict(
        __module__=type.__module__,
        __class__=type.__name__,
        module=type_obj.__module__,
        name=type_obj.__name__
    )


DEFAULT_DATETIME_FORMAT = '%Y%m%d%H%M%S.%f'


def datetime_from_config(config: dict[str, Any]) -> datetime.datetime:
    assert config['__module__'] == datetime.datetime.__module__
    assert config['__class__'] == datetime.datetime.__name__
    return datetime.datetime.strptime(config['value'], config.get('format', DEFAULT_DATETIME_FORMAT))


def datetime_to_config(dt: datetime.datetime, fmt: str = None) -> dict[str, Any]:
    assert isinstance(dt, datetime.datetime)
    fmt = fmt or DEFAULT_DATETIME_FORMAT
    result = dict(
        __module__=datetime.datetime.__module__,
        __class__=datetime.datetime.__name__,
        value=dt.strftime(fmt)
    )
    if fmt != DEFAULT_DATETIME_FORMAT:
        result['format'] = fmt
    return result


class Configurable:

    # TODO: Automatically handle subclasses that are also dataclasses.

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
            elif isinstance(property_value, type):
                config[property_name] = python_type_to_config(property_value)
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
            elif (isinstance(property_default, type) and
                  isinstance(property_override, dict) and
                  property_override.keys() == {'__module__', '__class__', 'module', 'name'}):
                property_override = python_type_from_config(property_override)
                try:
                    setattr(self, property_name, property_override)
                except AttributeError:
                    pass
