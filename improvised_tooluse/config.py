"""Access to config.yaml, which lives outside the package.

The entry points load it explicitly and every module reads it from here:

    from improvised_tooluse.config import load_config
    load_config(args.config)          # main.py, main_real_world.py

Nothing reads the file at import time -- values are looked up with `get()` when
they are actually needed, so the entry point's `load_config` always wins over
the fallback. If no one loaded a config, the first lookup falls back to
$IMPROVISED_TOOLUSE_CONFIG, then ./config.yaml, then the repo-root config.yaml.
"""

import os
from pathlib import Path

import yaml

CONFIG_ENV_VAR = "IMPROVISED_TOOLUSE_CONFIG"
# Repo root -- the package sits one level below it.
REPO_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

_config = None
_config_path = None


def default_config_path() -> Path:
    """Where to read the config from when no path is given."""
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()
    cwd_path = Path.cwd() / "config.yaml"
    if cwd_path.is_file():
        return cwd_path
    return REPO_CONFIG_PATH


def load_config(path=None) -> dict:
    """Read a config.yaml and make it the configuration of this process."""
    global _config, _config_path
    path = Path(path).expanduser() if path is not None else default_config_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"config file not found: {path}. Pass --config /path/to/config.yaml "
            f"or set ${CONFIG_ENV_VAR}."
        )
    with open(path) as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must hold a YAML mapping, got {type(loaded).__name__}")
    _config, _config_path = loaded, path
    return _config


def get_config() -> dict:
    """The loaded config, loading the default one on first use."""
    if _config is None:
        load_config()
    return _config


def config_path():
    """Path the current config was read from, or None before it is loaded."""
    return _config_path


def get(*keys):
    """A nested entry, e.g. get("servers", "sam3", "host")."""
    node = get_config()
    for depth, key in enumerate(keys, start=1):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"'{'.'.join(keys[:depth])}' is missing from {config_path()}")
        node = node[key]
    return node
