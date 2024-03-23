from typing import Any, Optional

from .framework import LCSAlgorithm, ClassifierSet


_ALGORITHMS: dict[str, type[LCSAlgorithm]] = {}
_ALGORITHM_PREFERRED_NAMES: dict[type[LCSAlgorithm], str] = {}


def register_algorithm(algorithm_type: type[LCSAlgorithm], *names: str) -> None:
    assert issubclass(algorithm_type, LCSAlgorithm)
    assert names
    for name in names:
        assert name and isinstance(name, str)
        assert name not in _ALGORITHMS
        _ALGORITHMS[name] = algorithm_type
        if algorithm_type not in _ALGORITHM_PREFERRED_NAMES:
            _ALGORITHM_PREFERRED_NAMES[algorithm_type] = name


def get_algorithm(name: str) -> Optional[type[LCSAlgorithm]]:
    assert name and isinstance(name, str)
    return _ALGORITHMS.get(name, None)


def list_algorithms() -> list[str]:
    return list(_ALGORITHM_PREFERRED_NAMES.values())


def build_algorithm(config: dict[str, Any]) -> LCSAlgorithm:
    algorithm_name = config['__class__']
    algorithm_type = get_algorithm(algorithm_name)
    assert algorithm_type is not None
    return algorithm_type.build(config)


def build_model(config: dict[str, Any]) -> ClassifierSet:
    algorithm = build_algorithm(config['algorithm'])
    return algorithm.build_model(config)
