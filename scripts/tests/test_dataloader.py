import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloader_builder import build_train_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "dataset_config.yaml"))
    parser.add_argument("--max-images", type=int, default=1000)
    args = parser.parse_args()

    loader = build_train_loader(args.config)
    seen = 0
    start = time.time()
    source_counts = {"artifact": 0, "cifake": 0}
    label_counts = {0: 0, 1: 0}

    for batch in loader:
        images, labels, metas = batch
        bs = images.size(0)
        seen += bs
        label_cpu = labels.detach().cpu().tolist()
        for lab in label_cpu:
            label_counts[int(lab)] += 1
        for m in metas["source"]:
            if m in source_counts:
                source_counts[m] += 1
        if seen >= args.max_images:
            break

    elapsed = max(time.time() - start, 1e-6)
    print(
        {
            "images_checked": seen,
            "elapsed_sec": round(elapsed, 4),
            "throughput_img_s": round(seen / elapsed, 2),
            "source_counts": source_counts,
            "label_counts": label_counts,
            "dtype": str(images.dtype),
            "shape": tuple(images.shape),
            "device": str(torch.device("cpu")),
        }
    )


if __name__ == "__main__":
    main()
