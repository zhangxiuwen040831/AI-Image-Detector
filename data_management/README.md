# Dataset Management & Upload Guide

## 1. Environment Setup
Install required packages:
```bash
pip install -r data_management/requirements.txt
```

## 2. Download Datasets
Use the provided script to download datasets.

**Download CIFAKE (Small, ~100MB):**
```bash
python data_management/download_datasets.py --dataset cifake --output_dir ./data
```

**Download GenImage Subset (Large, filtered by keyword):**
```bash
# Downloads only files containing "Midjourney" in their path
python data_management/download_datasets.py --dataset genimage --subset Midjourney --output_dir ./data
```

## 3. Verify Data Integrity
Before uploading, verify that all images are readable:
```bash
python data_management/verify_data.py --data_dir ./data/genimage
```

## 4. Upload to Cloud GPU (SSH)
Assuming you have SSH access to your cloud instance.

**Option A: Using `rsync` (Recommended for speed and resumability)**
```bash
# Replace user@cloud-ip with your actual SSH login
rsync -avz --progress -e "ssh -p 22" ./data/genimage user@cloud-ip:/path/to/remote/project/data/
```

**Option B: Using `scp` (Simpler but slower for many files)**
First, zip the folder to avoid overhead of transferring thousands of small files:
```bash
# On local machine
tar -czf genimage.tar.gz -C ./data genimage

# Upload
scp genimage.tar.gz user@cloud-ip:/path/to/remote/project/data/

# On remote machine (SSH)
cd /path/to/remote/project/data/
tar -xzf genimage.tar.gz
```

## 5. Directory Structure for Training
Ensure your remote dataset structure matches what `src/datasets/genimage.py` expects:

```
data/
  genimage/
    train/
      real/
      fake/
    val/
      real/
      fake/
```

If the downloaded dataset has a different structure, you may need to reorganize it using `mv` commands or a simple Python script.
