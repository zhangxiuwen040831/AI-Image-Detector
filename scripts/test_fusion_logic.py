import csv
from pathlib import Path

def check_paths():
    base_dir = Path("data/artifact-dataset")
    if not base_dir.exists():
        print("Artifact dataset not found.")
        return

    # Check COCO
    coco_meta = base_dir / "coco" / "metadata.csv"
    if coco_meta.exists():
        with open(coco_meta, "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            img_rel = row["image_path"]
            
            # Option 1: Relative to metadata file
            p1 = base_dir / "coco" / img_rel
            # Option 2: Relative to dataset root
            p2 = base_dir / img_rel
            
            print(f"COCO Image: {img_rel}")
            print(f"Path 1 (relative to meta): {p1} -> {p1.exists()}")
            print(f"Path 2 (relative to root): {p2} -> {p2.exists()}")

    # Check BigGAN
    bg_meta = base_dir / "big_gan" / "metadata.csv"
    if bg_meta.exists():
        with open(bg_meta, "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            img_rel = row["image_path"]
            
            p1 = base_dir / "big_gan" / img_rel
            p2 = base_dir / img_rel
            
            print(f"BigGAN Image: {img_rel}")
            print(f"Path 1 (relative to meta): {p1} -> {p1.exists()}")
            print(f"Path 2 (relative to root): {p2} -> {p2.exists()}")

if __name__ == "__main__":
    check_paths()
