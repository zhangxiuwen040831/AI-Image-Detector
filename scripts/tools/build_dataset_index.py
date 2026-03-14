import argparse
import platform
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.index_builder import build_dataset_index


def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "dataset_config.yaml"))
    args = parser.parse_args()
    cfg = load_cfg(Path(args.config))
    data_cfg = cfg["data"]

    if data_cfg.get("runtime_dataset_root"):
        dataset_root = Path(data_cfg["runtime_dataset_root"]).resolve()
    elif platform.system().lower().startswith("win"):
        dataset_root = Path(data_cfg["local_dataset_root"]).resolve()
    else:
        dataset_root = Path(data_cfg["server_dataset_root"]).resolve()

    output = (ROOT / data_cfg.get("dataset_index_name", "dataset_index.json")).resolve()
    res = build_dataset_index(
        dataset_root=dataset_root,
        output_path=output,
        max_workers=int(data_cfg.get("index_workers", 16)),
    )
    print(res["stats"])


if __name__ == "__main__":
    main()
