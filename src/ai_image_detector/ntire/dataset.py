from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_COLUMN_CANDIDATES = [
    "image_name",
    "image",
    "file_name",
    "filename",
    "img_name",
    "name",
    "path",
]

LABEL_COLUMN_CANDIDATES = [
    "label",
    "target",
    "class",
    "is_generated",
    "y",
]

OPTIONAL_METADATA_KEYS = [
    "distortion",
    "source",
    "domain",
    "provider",
    "dataset",
    "generator",
]

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


@dataclass
class SampleRecord:
    image_path: Path
    label: int
    shard_name: str
    metadata: Dict[str, Any]


def _normalize_column_names(columns: Iterable[str]) -> Dict[str, str]:
    return {str(c).strip().lower(): str(c) for c in columns}


def _pick_column(
    columns: Iterable[str],
    candidates: Sequence[str],
    fallback_index: int,
) -> str:
    normalized = _normalize_column_names(columns)
    for c in candidates:
        if c in normalized:
            return normalized[c]
    column_list = list(columns)
    if not column_list:
        raise ValueError("labels.csv has no columns")
    fallback_index = min(fallback_index, len(column_list) - 1)
    return column_list[fallback_index]


def _safe_label(v: Any) -> int:
    if isinstance(v, str):
        v = v.strip()
    try:
        x = int(float(v))
    except Exception:
        x = 1 if str(v).lower() in {"fake", "generated", "ai", "yes", "true"} else 0
    return 1 if x > 0 else 0


def _resolve_image_path(images_dir: Path, image_ref: str) -> Path:
    image_ref = str(image_ref).strip()
    p = Path(image_ref)
    if p.is_absolute() and p.exists():
        return p
    candidate = images_dir / image_ref
    if candidate.exists():
        return candidate
    if p.suffix:
        return candidate
    for ext in IMAGE_EXTENSIONS:
        with_ext = images_dir / f"{image_ref}{ext}"
        if with_ext.exists():
            return with_ext
    return candidate


class NTIRETrainDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        shard_ids: Optional[Sequence[int]] = None,
        subset_csv: Optional[str | Path] = None,
        transform: Optional[Any] = None,
        strict: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.strict = strict
        self.records: List[SampleRecord] = []
        self.subset_csv = Path(subset_csv) if subset_csv is not None else None
        self._load_all_shards(shard_ids)
        self._apply_subset()

    def _find_shard_dirs(self, shard_ids: Optional[Sequence[int]]) -> List[Path]:
        if shard_ids is None:
            shard_dirs = sorted(
                [p for p in self.root_dir.glob("shard_*") if p.is_dir()],
                key=lambda p: p.name,
            )
            return shard_dirs
        selected = []
        for sid in shard_ids:
            p = self.root_dir / f"shard_{sid}"
            if p.is_dir():
                selected.append(p)
        return sorted(selected, key=lambda p: p.name)

    def _read_labels_csv(self, csv_path: Path) -> pd.DataFrame:
        separators = [",", ";", "\t"]
        for sep in separators:
            try:
                df = pd.read_csv(csv_path, sep=sep)
                if len(df.columns) > 1:
                    if len(df.columns) == 1 and str(df.columns[0]).startswith("Unnamed"):
                        df = pd.read_csv(csv_path, sep=sep, index_col=0).reset_index()
                    return df
            except Exception:
                continue
        try:
            df = pd.read_csv(csv_path)
            if len(df.columns) == 1 and str(df.columns[0]).startswith("Unnamed"):
                df = pd.read_csv(csv_path, index_col=0).reset_index()
            return df
        except Exception:
            df = pd.read_csv(csv_path, header=None)
            if df.shape[1] >= 2:
                df = df.rename(columns={0: "image_name", 1: "label"})
            return df

    def _extract_optional_metadata(self, row: pd.Series) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        for col in row.index:
            key = str(col).strip().lower()
            if any(k in key for k in OPTIONAL_METADATA_KEYS):
                metadata[key] = row[col]
        return metadata

    def _load_all_shards(self, shard_ids: Optional[Sequence[int]]) -> None:
        shard_dirs = self._find_shard_dirs(shard_ids)
        if not shard_dirs:
            raise FileNotFoundError(f"No shard directories found in {self.root_dir}")
        for shard_dir in shard_dirs:
            csv_path = shard_dir / "labels.csv"
            images_dir = shard_dir / "images"
            if not csv_path.exists() or not images_dir.exists():
                if self.strict:
                    raise FileNotFoundError(f"Missing labels.csv or images in {shard_dir}")
                continue
            df = self._read_labels_csv(csv_path)
            image_col = _pick_column(df.columns, IMAGE_COLUMN_CANDIDATES, 0)
            label_col = _pick_column(df.columns, LABEL_COLUMN_CANDIDATES, min(1, len(df.columns) - 1))
            for _, row in df.iterrows():
                image_path = _resolve_image_path(images_dir, row[image_col])
                if self.strict and (not image_path.exists()):
                    raise FileNotFoundError(f"Image missing: {image_path}")
                label = _safe_label(row[label_col])
                metadata = self._extract_optional_metadata(row)
                metadata["shard_name"] = shard_dir.name
                metadata["image_path"] = str(image_path)
                metadata["label"] = label
                if "source" not in metadata:
                    parent_name = image_path.parent.name.lower() if image_path.parent else ""
                    if parent_name not in {"images", ""}:
                        metadata["source"] = parent_name
                self.records.append(
                    SampleRecord(
                        image_path=image_path,
                        label=label,
                        shard_name=shard_dir.name,
                        metadata=metadata,
                    )
                )
        if not self.records:
            raise RuntimeError(f"No valid records loaded from {self.root_dir}")

    def _apply_subset(self) -> None:
        if self.subset_csv is None:
            return
        if not self.subset_csv.exists():
            raise FileNotFoundError(f"subset_csv not found: {self.subset_csv}")
        df = pd.read_csv(self.subset_csv)
        if "image_path" in df.columns:
            allowed = set(df["image_path"].astype(str).tolist())
            self.records = [r for r in self.records if str(r.image_path) in allowed]
        else:
            name_col = None
            for c in df.columns:
                if "image" in str(c).lower() or "name" in str(c).lower():
                    name_col = c
                    break
            if name_col is None:
                raise ValueError("subset_csv must contain image_path or image name column")
            allowed_names = {Path(x).name for x in df[name_col].astype(str).tolist()}
            self.records = [r for r in self.records if r.image_path.name in allowed_names]
        if not self.records:
            raise RuntimeError("subset filter removed all samples")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        image_np = np.array(image)
        if self.transform is not None:
            if hasattr(self.transform, "__class__") and "albumentations" in str(type(self.transform)).lower():
                image_tensor = self.transform(image=image_np)["image"]
            else:
                image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        label_tensor = torch.tensor(record.label, dtype=torch.float32)
        return image_tensor, label_tensor, dict(record.metadata)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.records:
            row = dict(r.metadata)
            row["shard_name"] = r.shard_name
            row["image_path"] = str(r.image_path)
            row["label"] = r.label
            rows.append(row)
        return pd.DataFrame(rows)


def print_dataset_sanity(dataset: NTIRETrainDataset, max_rows: int = 5) -> None:
    df = dataset.to_dataframe()
    label_counts = df["label"].value_counts(dropna=False).to_dict()
    shard_counts = df["shard_name"].value_counts(dropna=False).to_dict()
    print(f"Total samples: {len(dataset)}")
    print(f"Label counts: {label_counts}")
    print(f"Shard counts: {shard_counts}")
    optional_cols = [c for c in df.columns if any(k in str(c).lower() for k in OPTIONAL_METADATA_KEYS)]
    if optional_cols:
        print(f"Optional metadata columns: {optional_cols}")
    print(df.head(max_rows).to_string(index=False))


def build_balanced_sample_weights(
    dataset: NTIRETrainDataset,
    secondary_key_priority: Sequence[str] = ("distortion", "source", "shard_name"),
) -> torch.DoubleTensor:
    df = dataset.to_dataframe().copy()
    label_counts = df["label"].value_counts().to_dict()
    label_weight = {k: 1.0 / max(v, 1) for k, v in label_counts.items()}
    secondary_key = None
    for key in secondary_key_priority:
        matches = [c for c in df.columns if key in str(c).lower()]
        if matches:
            secondary_key = matches[0]
            break
    if secondary_key is None:
        secondary_key = "shard_name"
    secondary_counts = df[secondary_key].value_counts().to_dict()
    secondary_weight = {k: 1.0 / max(v, 1) for k, v in secondary_counts.items()}
    raw = []
    for _, row in df.iterrows():
        w = label_weight[row["label"]] * secondary_weight[row[secondary_key]]
        raw.append(w)
    weights = np.array(raw, dtype=np.float64)
    weights = weights / max(weights.sum(), 1e-12)
    return torch.from_numpy(weights)


def build_train_val_indices(
    dataset: NTIRETrainDataset,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], str]:
    df = dataset.to_dataframe().copy()
    rng = random.Random(seed)
    source_cols = [c for c in df.columns if "source" in str(c).lower()]
    if source_cols and df[source_cols[0]].nunique(dropna=True) > 1:
        key = source_cols[0]
        groups = list(df[key].dropna().unique())
        rng.shuffle(groups)
        val_groups: set[str] = set()
        target = max(int(len(df) * val_ratio), 1)
        count = 0
        for g in groups:
            val_groups.add(g)
            count += int((df[key] == g).sum())
            if count >= target:
                break
        val_idx = df.index[df[key].isin(val_groups)].tolist()
        train_idx = df.index[~df[key].isin(val_groups)].tolist()
        return train_idx, val_idx, f"source-held-out ({key})"
    if df["shard_name"].nunique() > 1:
        shards = list(df["shard_name"].unique())
        rng.shuffle(shards)
        val_shards: set[str] = set()
        target = max(int(len(df) * val_ratio), 1)
        count = 0
        for s in shards:
            val_shards.add(s)
            count += int((df["shard_name"] == s).sum())
            if count >= target:
                break
        val_idx = df.index[df["shard_name"].isin(val_shards)].tolist()
        train_idx = df.index[~df["shard_name"].isin(val_shards)].tolist()
        return train_idx, val_idx, "shard-held-out"
    label0 = df.index[df["label"] == 0].tolist()
    label1 = df.index[df["label"] == 1].tolist()
    rng.shuffle(label0)
    rng.shuffle(label1)
    n0 = max(1, int(len(label0) * val_ratio))
    n1 = max(1, int(len(label1) * val_ratio))
    val_idx = label0[:n0] + label1[:n1]
    val_set = set(val_idx)
    train_idx = [i for i in df.index.tolist() if i not in val_set]
    return train_idx, val_idx, "stratified-random"
