import os

try:
    import yaml
except ImportError as exc:
    raise RuntimeError(
        "PyYAML is required to load config.yaml. "
        "Install it with: pip install pyyaml"
    ) from exc

_CONFIG_CACHE = None


def _config_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))


def load_config():
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        path = _config_path()
        if not os.path.exists(path):
            _CONFIG_CACHE = {}
        else:
            with open(path, "r", encoding="utf-8") as handle:
                _CONFIG_CACHE = yaml.safe_load(handle) or {}
    return _CONFIG_CACHE


def get_value(*path, default=None):
    data = load_config()
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data
