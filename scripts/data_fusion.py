import os
import shutil
import csv
import argparse
import random
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import numpy as np

class DataAnalyzer:
    def __init__(self, source_dirs):
        self.source_dirs = [Path(d) for d in source_dirs]
        self.stats = {}

    def analyze(self):
        print("Analyzing datasets...")
        for source in self.source_dirs:
            print(f"Scanning {source}...")
            if (source / "metadata.csv").exists():
                # Artifact dataset structure (per folder)
                self._analyze_artifact_folder(source)
            elif (source / "train").exists():
                # CIFAKE structure
                self._analyze_cifake_folder(source)
            elif any((source / sub).is_dir() for sub in ["coco", "big_gan", "afhq"]):
                 # Artifact root folder
                 for sub in source.iterdir():
                     if sub.is_dir() and (sub / "metadata.csv").exists():
                         self._analyze_artifact_folder(sub)
            else:
                print(f"Unknown structure for {source}")

        return self.stats

    def _analyze_artifact_folder(self, folder):
        try:
            with open(folder / "metadata.csv", "r") as f:
                reader = csv.DictReader(f)
                count = 0
                resolutions = []
                targets = set()
                for row in reader:
                    count += 1
                    targets.add(row.get("target", "?"))
                    # Sample resolution every 1000 images
                    if count % 1000 == 1:
                        img_path = folder / row["image_path"]
                        if img_path.exists():
                            with Image.open(img_path) as img:
                                resolutions.append(img.size)
                
                avg_res = np.mean(resolutions, axis=0) if resolutions else (0, 0)
                self.stats[folder.name] = {
                    "count": count,
                    "type": "artifact_subset",
                    "targets": list(targets),
                    "avg_resolution": f"{int(avg_res[0])}x{int(avg_res[1])}"
                }
        except Exception as e:
            print(f"Error analyzing {folder}: {e}")

    def _analyze_cifake_folder(self, folder):
        count = 0
        resolutions = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    count += 1
                    if count % 1000 == 1:
                         with Image.open(Path(root) / file) as img:
                                resolutions.append(img.size)
        
        avg_res = np.mean(resolutions, axis=0) if resolutions else (0, 0)
        self.stats[folder.name] = {
            "count": count,
            "type": "cifake_structure",
            "avg_resolution": f"{int(avg_res[0])}x{int(avg_res[1])}"
        }

class DataMerger:
    def __init__(self, source_dirs, target_dir, ratio=0.8):
        self.source_dirs = [Path(d) for d in source_dirs]
        self.target_dir = Path(target_dir)
        self.ratio = ratio
        self.manifest_file = self.target_dir / "fusion_manifest.json"
        self.manifest = {"files": [], "stats": {"added": 0, "skipped": 0, "conflicts": 0}}
        self.target_size = (224, 224)

    def merge(self):
        self._prepare_target_dirs()
        
        for source in self.source_dirs:
            if (source / "train").exists():
                self._merge_cifake(source)
            elif any((source / sub).is_dir() for sub in ["coco", "big_gan", "afhq"]):
                 for sub in source.iterdir():
                     if sub.is_dir() and (sub / "metadata.csv").exists():
                         self._merge_artifact_subfolder(sub)
            elif (source / "metadata.csv").exists():
                 self._merge_artifact_subfolder(source)

        self._save_manifest()
        return self.manifest["stats"]

    def _prepare_target_dirs(self):
        for split in ["train", "test"]:
            for cls in ["real", "fake"]:
                (self.target_dir / split / cls).mkdir(parents=True, exist_ok=True)

    def _process_image(self, src_path, split, label):
        if not src_path.exists():
            return

        dest_folder = self.target_dir / split / label
        dest_name = src_path.name
        dest_path = dest_folder / dest_name

        # Conflict resolution
        if dest_path.exists():
            self.manifest["stats"]["conflicts"] += 1
            base, ext = os.path.splitext(dest_name)
            dest_name = f"{base}_{random.randint(1000,9999)}{ext}"
            dest_path = dest_folder / dest_name

        try:
            # Data Alignment: Resize/Convert
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img = img.resize(self.target_size, Image.Resampling.LANCZOS)
                img.save(dest_path, quality=95)
            
            self.manifest["files"].append(str(dest_path.relative_to(self.target_dir)))
            self.manifest["stats"]["added"] += 1
        except Exception as e:
            print(f"Error processing {src_path}: {e}")
            self.manifest["stats"]["skipped"] += 1

    def _merge_cifake(self, source):
        print(f"Merging CIFAKE from {source}...")
        # Map: train/REAL -> train/real, etc.
        # CIFAKE usually has 'train' and 'test' folders, inside them 'REAL' and 'FAKE'
        for split in ["train", "test"]:
            for label_upper in ["REAL", "FAKE"]:
                src_folder = source / split / label_upper
                if not src_folder.exists():
                    # Try lowercase
                    src_folder = source / split / label_upper.lower()
                
                if src_folder.exists():
                    label_lower = "real" if "real" in label_upper.lower() else "fake"
                    # CIFAKE is already split, so respect its split
                    for img_file in tqdm(list(src_folder.glob("*.*")), desc=f"CIFAKE {split}/{label_lower}"):
                        self._process_image(img_file, split, label_lower)

    def _merge_artifact_subfolder(self, source):
        # Sampling strategy:
        # - Real: Limit total real from artifact to avoid overwhelming cifake
        # - Fake: Limit total fake, ensure diversity across generators
        
        # Hardcoded limits for demonstration/balance
        # Real sources: coco, imagenet, lsun, ffhq, celebahq, afhq, landscape, metfaces
        # Fake sources: others
        
        folder_name = source.name.lower()
        is_real_source = folder_name in ["coco", "imagenet", "lsun", "ffhq", "celebahq", "afhq", "landscape", "metfaces"]
        
        # Limit per folder to ensure diversity (e.g. don't take 1M stylegan2)
        LIMIT_PER_FOLDER = 5000 if not is_real_source else 10000 
        
        print(f"Merging Artifact subfolder {source.name} (Limit: {LIMIT_PER_FOLDER})...")
        try:
            with open(source / "metadata.csv", "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                # Shuffle for random sampling
                random.shuffle(rows)
                
                # Apply limit
                rows = rows[:LIMIT_PER_FOLDER]
                
                split_idx = int(len(rows) * self.ratio)
                
                for i, row in enumerate(tqdm(rows, desc=f"Artifact {source.name}")):
                    img_path = source / row["image_path"]
                    target = row.get("target", "1")
                    
                    # 0 is real, others are fake. 
                    # Note: cycle_gan and pro_gan have mix of 0 and 6.
                    label = "real" if str(target) == "0" else "fake"
                    split = "train" if i < split_idx else "test"
                    
                    self._process_image(img_path, split, label)
        except Exception as e:
            print(f"Error merging {source}: {e}")

    def _save_manifest(self):
        with open(self.manifest_file, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def rollback(self):
        if not self.manifest_file.exists():
            print("No manifest found. Cannot rollback.")
            return

        with open(self.manifest_file, "r") as f:
            data = json.load(f)
        
        print(f"Rolling back {len(data['files'])} files...")
        for file_rel in tqdm(data["files"]):
            file_path = self.target_dir / file_rel
            if file_path.exists():
                file_path.unlink()
        
        self.manifest_file.unlink()
        print("Rollback complete.")

def main():
    parser = argparse.ArgumentParser(description="Data Fusion Tool")
    parser.add_argument("--action", choices=["analyze", "merge", "rollback"], required=True)
    parser.add_argument("--source_dirs", nargs="+", help="List of source directories")
    parser.add_argument("--target_dir", help="Target directory for merge")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train/Test split ratio")
    
    args = parser.parse_args()

    if args.action == "analyze":
        analyzer = DataAnalyzer(args.source_dirs)
        stats = analyzer.analyze()
        print(json.dumps(stats, indent=2))
    
    elif args.action == "merge":
        merger = DataMerger(args.source_dirs, args.target_dir, args.ratio)
        stats = merger.merge()
        print("Merge Complete:", stats)
        
    elif args.action == "rollback":
        merger = DataMerger([], args.target_dir) # Sources not needed for rollback
        merger.rollback()

if __name__ == "__main__":
    main()
