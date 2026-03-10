import os
import shutil
import hashlib
import random
from pathlib import Path
from tqdm import tqdm
import csv
import json

# Configuration
SOURCE_ARTIFACT = Path("/root/autodl-tmp/ai-image-detector/data/artifact-dataset")
SOURCE_CIFAKE = Path("/root/autodl-tmp/ai-image-detector/data/cifake")
TARGET_ROOT = Path("/root/autodl-tmp/ai-image-detector/data/mix_data_v2")
MANIFEST_FILE = Path("/root/autodl-tmp/ai-image-detector/data/fusion_manifest.jsonl")

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_hash_path(src_str):
    """Generate a 2-level subdirectory path based on md5 of src_str."""
    h = hashlib.md5(src_str.encode()).hexdigest()
    return Path(h[0:2]) / Path(h[2:4])

def get_files_recursively_generator(folder):
    """Recursively yield all image files."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    if not folder.exists():
        return
    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            if Path(filename).suffix.lower() in extensions:
                yield Path(root) / filename

def analyze_cifake_stream(source_dir, writer):
    """
    Stream CIFAKE images to manifest writer.
    """
    print(f"Analyzing CIFAKE at {source_dir}...", flush=True)
    count = 0
    for split in ["train", "test"]:
        for label_name in ["REAL", "FAKE"]:
            folder = source_dir / split / label_name
            if not folder.exists():
                folder = source_dir / split / label_name.lower()
            
            if folder.exists():
                target_label = "real" if "real" in label_name.lower() else "fake"
                for p in get_files_recursively_generator(folder):
                    entry = {
                        "src": str(p),
                        "split": split,
                        "label": target_label,
                        "source": "cifake"
                    }
                    writer.write(json.dumps(entry) + "\n")
                    count += 1
    print(f"Found {count} images in CIFAKE.", flush=True)
    return count

def analyze_artifact_stream(source_dir, writer):
    """
    Stream Artifact images to manifest writer.
    """
    print(f"Analyzing Artifact at {source_dir}...", flush=True)
    
    if not source_dir.exists():
        print(f"Artifact source directory not found: {source_dir}", flush=True)
        return 0

    count = 0
    # List all subdirectories
    try:
        subdirs = [d for d in source_dir.iterdir() if d.is_dir()]
        print(f"Found {len(subdirs)} subdirectories in Artifact.", flush=True)
    except Exception as e:
        print(f"Error accessing artifact directory {source_dir}: {e}", flush=True)
        return 0
    
    for sub in subdirs:
        print(f"Processing {sub.name}...", flush=True)
        meta_file = sub / "metadata.csv"
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    reader = csv.DictReader(f)
                    
                    for row in reader:
                        img_rel = row.get("image_path")
                        if not img_rel: continue
                        
                        img_path = sub / img_rel
                        if not img_path.exists():
                            img_path = source_dir / img_rel
                        
                        if img_path.exists():
                            target = row.get("target", "1")
                            label = "real" if str(target) == "0" else "fake"
                            # Stream split decision
                            split = "train" if random.random() < 0.8 else "test"
                            
                            entry = {
                                "src": str(img_path),
                                "split": split,
                                "label": label,
                                "source": f"artifact_{sub.name}"
                            }
                            writer.write(json.dumps(entry) + "\n")
                            count += 1
                        
            except Exception as e:
                print(f"Error reading metadata for {sub.name}: {e}", flush=True)
    
    print(f"Found {count} images in Artifact.", flush=True)
    return count

def main():
    # 1. Prepare Target Directory
    for split in ["train", "test"]:
        for label in ["real", "fake"]:
            (TARGET_ROOT / split / label).mkdir(parents=True, exist_ok=True)

    # 2. Analyze Sources and create Manifest
    print(f"Creating manifest at {MANIFEST_FILE}...", flush=True)
    total_source_count = 0
    if not MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "w") as f:
            total_source_count += analyze_cifake_stream(SOURCE_CIFAKE, f)
            total_source_count += analyze_artifact_stream(SOURCE_ARTIFACT, f)
    else:
        print("Manifest already exists. Counting lines...", flush=True)
        with open(MANIFEST_FILE, "r") as f:
            for _ in f:
                total_source_count += 1
    
    print(f"Total images found: {total_source_count}", flush=True)
    
    # 3. Execute Copy from Manifest
    copied_count = 0
    stats = {"real": 0, "fake": 0, "train": 0, "test": 0}
    
    print("Starting copy process (Subdirectory Mode)...", flush=True)
    
    # Cache created subdirectories to avoid redundant mkdir calls
    created_subdirs = set()

    with open(MANIFEST_FILE, "r") as f:
        for i, line in enumerate(f):
            if i % 10000 == 0:
                print(f"Progress: {i}/{total_source_count} ({i/total_source_count:.1%})", flush=True)
                
            try:
                item = json.loads(line)
                src_str = item["src"]
                src = Path(src_str)
                split = item["split"]
                label = item["label"]
                
                # Subdirectory hashing
                rel_hash_path = get_hash_path(src_str)
                dest_dir = TARGET_ROOT / split / label / rel_hash_path
                
                # Efficient mkdir
                if dest_dir not in created_subdirs:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    created_subdirs.add(dest_dir)

                dest_name = src.name
                dest_path = dest_dir / dest_name
                
                # Resumable Check
                if dest_path.exists() or os.path.islink(dest_path):
                     copied_count += 1
                     stats[label] += 1
                     stats[split] += 1
                     continue

                # Use symlink
                try:
                    os.symlink(src, dest_path)
                except OSError as e:
                    if e.errno == 17: # File exists
                         pass 
                    else:
                        print(f"ERROR: Symlink failed for {src} to {dest_path}: {e}", flush=True)
                        # Do not raise, just continue to skip this problematic file
                        continue
                copied_count += 1
                stats[label] += 1
                stats[split] += 1
            except Exception as e:
                print(f"CRITICAL ERROR at line {i}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                if "Disk quota exceeded" in str(e) or "No space left on device" in str(e):
                    print("Stopping due to storage limits.", flush=True)
                    break

    # 4. Validation
    print("Validating merge...", flush=True)
    target_count = 0
    for root, _, files in os.walk(TARGET_ROOT):
        target_count += len([f for f in files if not f.startswith('.')])
        
    print(f"\n=== Merge Report ===", flush=True)
    print(f"Source Total: {total_source_count}", flush=True)
    print(f"Target Total: {target_count}", flush=True)
    print(f"Stats: {stats}", flush=True)
    
    if target_count == total_source_count:
        print("SUCCESS: Count matches.", flush=True)
    else:
        print(f"WARNING: Count mismatch! Diff: {total_source_count - target_count}", flush=True)

    # Generate MD5 Report (Sample)
    report_path = TARGET_ROOT / "merge_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Merge Report\n")
        f.write(f"Total Source: {total_source_count}\n")
        f.write(f"Total Target: {target_count}\n")
        f.write(f"Details: {stats}\n")
        f.write(f"Validation: {'PASS' if target_count == total_source_count else 'FAIL'}\n")

if __name__ == "__main__":
    main()
