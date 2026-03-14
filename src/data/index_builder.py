import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ARTIFACT_REAL_SUBTYPES = [
    "afhq",
    "coco",
    "imagenet",
    "ffhq",
    "metfaces",
    "landscape",
]

ARTIFACT_FAKE_SUBTYPES = [
    "big_gan",
    "celebahq",
    "cips",
    "cycle_gan",
    "ddpm",
    "denoising_diffusion_gan",
    "diffusion_gan",
    "face_synthetics",
    "gansformer",
    "gau_gan",
    "generative_inpainting",
    "glide",
    "lama",
    "latent_diffusion",
    "lsun",
    "mat",
    "palette",
    "pro_gan",
    "projected_gan",
    "sfhq",
    "stable_diffusion",
    "star_gan",
    "stylegan1",
    "stylegan2",
    "stylegan3",
    "taming_transformer",
    "vq_diffusion",
]


def _scan_single_subtype(
    subtype_dir: Path,
    label: int,
    source: str,
    subtype: str,
) -> Tuple[List[Dict], int, int]:
    records: List[Dict] = []
    real_count = 0
    fake_count = 0
    if not subtype_dir.exists():
        return records, real_count, fake_count

    for path in subtype_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            records.append(
                {
                    "path": str(path.resolve()),
                    "label": label,
                    "source": source,
                    "subtype": subtype,
                }
            )
            if label == 0:
                real_count += 1
            else:
                fake_count += 1
    return records, real_count, fake_count


def build_dataset_index(
    dataset_root: Path,
    output_path: Path,
    max_workers: int = 16,
) -> Dict:
    dataset_root = dataset_root.resolve()
    output_path = output_path.resolve()
    artifact_root = dataset_root / "artifact-dataset"
    cifake_root = dataset_root / "cifake"

    scan_jobs = []
    for subtype in ARTIFACT_REAL_SUBTYPES:
        scan_jobs.append((artifact_root / subtype, 0, "artifact", subtype))
    for subtype in ARTIFACT_FAKE_SUBTYPES:
        scan_jobs.append((artifact_root / subtype, 1, "artifact", subtype))
    scan_jobs.append((cifake_root / "real", 0, "cifake", "real"))
    scan_jobs.append((cifake_root / "fake", 1, "cifake", "fake"))

    all_records: List[Dict] = []
    total_real = 0
    total_fake = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_scan_single_subtype, path, label, source, subtype)
            for path, label, source, subtype in scan_jobs
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning subtypes"):
            records, real_count, fake_count = future.result()
            all_records.extend(records)
            total_real += real_count
            total_fake += fake_count

    output = {
        "dataset_root": str(dataset_root),
        "records": all_records,
        "stats": {
            "total_images": len(all_records),
            "real_count": total_real,
            "fake_count": total_fake,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    return output
