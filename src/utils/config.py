from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: str, base_config_path: str | None = None) -> Dict[str, Any]:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if base_config_path is not None:
        base_path = Path(base_config_path)
        if not base_path.exists():
            raise FileNotFoundError(f"Base config file not found: {base_path}")
        with base_path.open("r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}
        cfg = _deep_update(base_cfg, cfg)

    required_keys = ["system", "data", "model", "training", "logging"]
    for key in required_keys:
        if key not in cfg:
            raise KeyError(f"Missing required config section: {key}")
    return cfg
