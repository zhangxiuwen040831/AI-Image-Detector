from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


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


def load_hard_negative_names(path: Optional[str | Path]) -> set[str]:
    if path is None:
        return set()
    file_path = Path(path)
    if not file_path.exists():
        return set()
    names = set()
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if (not line) or line.startswith("#"):
            continue
        names.add(Path(line).name.lower())
    return names


def _extract_sample_name(row: pd.Series) -> str:
    for key in ("image_name", "image", "file_name", "filename", "img_name", "name", "image_path", "path"):
        if key in row.index and pd.notna(row[key]):
            value = Path(str(row[key])).name.strip()
            if value:
                return value
    return ""


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


def normalize_transform_probs(
    clean_prob: float,
    mild_prob: float,
    hard_prob: float,
) -> Tuple[float, float, float]:
    probs = np.asarray([clean_prob, mild_prob, hard_prob], dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if total <= 1e-12:
        return 1.0, 0.0, 0.0
    probs = probs / total
    return float(probs[0]), float(probs[1]), float(probs[2])


def normalize_real_transform_probs(
    clean_prob: float,
    mild_prob: float,
    hard_prob: float,
) -> Tuple[float, float, float]:
    return normalize_transform_probs(
        clean_prob=clean_prob,
        mild_prob=mild_prob,
        hard_prob=hard_prob,
    )


def compute_base_hard_real_score(
    base_logit: torch.Tensor,
    semantic_logit: Optional[torch.Tensor] = None,
    frequency_logit: Optional[torch.Tensor] = None,
    min_probability: float = 0.0,
) -> torch.Tensor:
    base_logit = base_logit.view(-1)
    score = torch.sigmoid(base_logit)
    if semantic_logit is not None and frequency_logit is not None:
        semantic_logit = semantic_logit.view(-1)
        frequency_logit = frequency_logit.view(-1)
        score = score * (1.0 + torch.relu(frequency_logit - semantic_logit))
    if float(min_probability) > 0.0:
        score = score * (torch.sigmoid(base_logit) >= float(min_probability)).float()
    return score


def compute_fragile_aigc_score(
    base_logit: torch.Tensor,
    semantic_logit: Optional[torch.Tensor] = None,
    frequency_logit: Optional[torch.Tensor] = None,
    max_probability: float = 0.80,
) -> torch.Tensor:
    base_logit = base_logit.view(-1)
    base_prob = torch.sigmoid(base_logit)
    uncertainty = 4.0 * base_prob * (1.0 - base_prob)
    score = uncertainty
    if semantic_logit is not None and frequency_logit is not None:
        semantic_logit = semantic_logit.view(-1)
        frequency_logit = frequency_logit.view(-1)
        agreement = torch.sigmoid(0.5 * (semantic_logit + frequency_logit))
        score = score * (0.5 + agreement)
    if float(max_probability) < 1.0:
        score = score * (base_prob <= float(max_probability)).float()
    return score


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
                metadata.setdefault("image_name", image_path.name)
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


class HardRealBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        primary_indices: Sequence[int],
        buffer_indices: Sequence[int],
        batch_size: int,
        buffer_ratio: float,
        seed: int,
    ) -> None:
        self.primary_indices = [int(idx) for idx in primary_indices]
        self.buffer_indices = sorted(set(int(idx) for idx in buffer_indices))
        self.batch_size = int(batch_size)
        self.buffer_ratio = max(0.0, min(float(buffer_ratio), 0.5))
        self.seed = int(seed)

    def __len__(self) -> int:
        return max(len(self.primary_indices) // max(self.batch_size, 1), 1)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        primary = self.primary_indices[:]
        rng.shuffle(primary)
        buffer_count = 0
        if self.buffer_indices:
            buffer_count = max(1, int(round(self.batch_size * self.buffer_ratio)))
            buffer_count = min(buffer_count, self.batch_size - 1)
        primary_count = max(self.batch_size - buffer_count, 1)
        cursor = 0
        for _ in range(len(self)):
            batch: List[int] = []
            for _ in range(primary_count):
                if cursor >= len(primary):
                    rng.shuffle(primary)
                    cursor = 0
                batch.append(primary[cursor])
                cursor += 1
            for _ in range(buffer_count):
                batch.append(rng.choice(self.buffer_indices))
            rng.shuffle(batch)
            yield batch


class CurriculumBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        primary_indices: Sequence[int],
        hard_real_indices: Optional[Sequence[int]],
        anchor_hard_real_indices: Optional[Sequence[int]],
        fragile_aigc_indices: Optional[Sequence[int]],
        batch_size: int,
        hard_real_ratio: float,
        anchor_hard_real_ratio: float,
        fragile_aigc_ratio: float,
        seed: int,
    ) -> None:
        self.primary_indices = [int(idx) for idx in primary_indices]
        self.hard_real_indices = sorted(set(int(idx) for idx in (hard_real_indices or [])))
        self.anchor_hard_real_indices = sorted(set(int(idx) for idx in (anchor_hard_real_indices or [])))
        self.fragile_aigc_indices = sorted(set(int(idx) for idx in (fragile_aigc_indices or [])))
        self.batch_size = max(int(batch_size), 1)
        self.hard_real_ratio = max(0.0, min(float(hard_real_ratio), 0.5))
        self.anchor_hard_real_ratio = max(0.0, min(float(anchor_hard_real_ratio), 0.5))
        self.fragile_aigc_ratio = max(0.0, min(float(fragile_aigc_ratio), 0.5))
        self.seed = int(seed)

    def __len__(self) -> int:
        return max(len(self.primary_indices) // self.batch_size, 1)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        primary = self.primary_indices[:]
        rng.shuffle(primary)
        hard_real_count = 0
        anchor_hard_real_count = 0
        fragile_aigc_count = 0
        if self.hard_real_indices:
            hard_real_count = max(1, int(round(self.batch_size * self.hard_real_ratio)))
        if self.anchor_hard_real_indices:
            anchor_hard_real_count = max(1, int(round(self.batch_size * self.anchor_hard_real_ratio)))
        if self.fragile_aigc_indices:
            fragile_aigc_count = max(1, int(round(self.batch_size * self.fragile_aigc_ratio)))
        max_buffer = max(self.batch_size - 1, 0)
        total_buffer = hard_real_count + anchor_hard_real_count + fragile_aigc_count
        if total_buffer > max_buffer and total_buffer > 0:
            scale = max_buffer / float(total_buffer)
            hard_real_count = int(round(hard_real_count * scale))
            anchor_hard_real_count = int(round(anchor_hard_real_count * scale))
            fragile_aigc_count = int(round(fragile_aigc_count * scale))
            if self.hard_real_indices and hard_real_count == 0:
                hard_real_count = 1
            if (
                self.anchor_hard_real_indices
                and anchor_hard_real_count == 0
                and (hard_real_count + fragile_aigc_count) < max_buffer
            ):
                anchor_hard_real_count = 1
            if (
                self.fragile_aigc_indices
                and fragile_aigc_count == 0
                and (hard_real_count + anchor_hard_real_count) < max_buffer
            ):
                fragile_aigc_count = 1
        primary_count = max(self.batch_size - hard_real_count - anchor_hard_real_count - fragile_aigc_count, 1)
        cursor = 0
        for _ in range(len(self)):
            batch: List[int] = []
            for _ in range(primary_count):
                if cursor >= len(primary):
                    rng.shuffle(primary)
                    cursor = 0
                batch.append(primary[cursor])
                cursor += 1
            for _ in range(hard_real_count):
                batch.append(rng.choice(self.hard_real_indices))
            for _ in range(anchor_hard_real_count):
                batch.append(rng.choice(self.anchor_hard_real_indices))
            for _ in range(fragile_aigc_count):
                batch.append(rng.choice(self.fragile_aigc_indices))
            rng.shuffle(batch)
            yield batch


class BufferedTransformDataset(Dataset):
    def __init__(
        self,
        base_dataset: NTIRETrainDataset,
        transform: Optional[Any],
        secondary_transform: Optional[Any] = None,
        real_clean_transform: Optional[Any] = None,
        real_mild_transform: Optional[Any] = None,
        real_hard_transform: Optional[Any] = None,
        secondary_real_clean_transform: Optional[Any] = None,
        secondary_real_mild_transform: Optional[Any] = None,
        secondary_real_hard_transform: Optional[Any] = None,
        real_clean_prob: float = 0.4,
        real_mild_prob: float = 0.4,
        real_hard_prob: float = 0.2,
        hard_real_indices: Optional[Sequence[int]] = None,
        hard_real_clean_prob: float = 0.7,
        hard_real_mild_prob: float = 0.2,
        hard_real_hard_prob: float = 0.1,
        anchor_hard_real_indices: Optional[Sequence[int]] = None,
        anchor_hard_real_clean_prob: float = 0.8,
        anchor_hard_real_mild_prob: float = 0.15,
        anchor_hard_real_hard_prob: float = 0.05,
        aigc_clean_transform: Optional[Any] = None,
        aigc_mild_transform: Optional[Any] = None,
        aigc_hard_transform: Optional[Any] = None,
        secondary_aigc_clean_transform: Optional[Any] = None,
        secondary_aigc_mild_transform: Optional[Any] = None,
        secondary_aigc_hard_transform: Optional[Any] = None,
        aigc_clean_prob: float = 0.5,
        aigc_mild_prob: float = 0.3,
        aigc_hard_prob: float = 0.2,
        fragile_aigc_indices: Optional[Sequence[int]] = None,
        fragile_aigc_clean_prob: float = 0.6,
        fragile_aigc_mild_prob: float = 0.25,
        fragile_aigc_hard_prob: float = 0.15,
    ) -> None:
        self.base_dataset = base_dataset
        self.transform = transform
        self.secondary_transform = secondary_transform
        self.real_clean_transform = real_clean_transform
        self.real_mild_transform = real_mild_transform
        self.real_hard_transform = real_hard_transform
        self.secondary_real_clean_transform = secondary_real_clean_transform
        self.secondary_real_mild_transform = secondary_real_mild_transform
        self.secondary_real_hard_transform = secondary_real_hard_transform
        self.aigc_clean_transform = aigc_clean_transform
        self.aigc_mild_transform = aigc_mild_transform
        self.aigc_hard_transform = aigc_hard_transform
        self.secondary_aigc_clean_transform = secondary_aigc_clean_transform
        self.secondary_aigc_mild_transform = secondary_aigc_mild_transform
        self.secondary_aigc_hard_transform = secondary_aigc_hard_transform
        (
            self.real_clean_prob,
            self.real_mild_prob,
            self.real_hard_prob,
        ) = normalize_transform_probs(
            clean_prob=real_clean_prob,
            mild_prob=real_mild_prob,
            hard_prob=real_hard_prob,
        )
        (
            self.hard_real_clean_prob,
            self.hard_real_mild_prob,
            self.hard_real_hard_prob,
        ) = normalize_transform_probs(
            clean_prob=hard_real_clean_prob,
            mild_prob=hard_real_mild_prob,
            hard_prob=hard_real_hard_prob,
        )
        (
            self.anchor_hard_real_clean_prob,
            self.anchor_hard_real_mild_prob,
            self.anchor_hard_real_hard_prob,
        ) = normalize_transform_probs(
            clean_prob=anchor_hard_real_clean_prob,
            mild_prob=anchor_hard_real_mild_prob,
            hard_prob=anchor_hard_real_hard_prob,
        )
        (
            self.aigc_clean_prob,
            self.aigc_mild_prob,
            self.aigc_hard_prob,
        ) = normalize_transform_probs(
            clean_prob=aigc_clean_prob,
            mild_prob=aigc_mild_prob,
            hard_prob=aigc_hard_prob,
        )
        (
            self.fragile_aigc_clean_prob,
            self.fragile_aigc_mild_prob,
            self.fragile_aigc_hard_prob,
        ) = normalize_transform_probs(
            clean_prob=fragile_aigc_clean_prob,
            mild_prob=fragile_aigc_mild_prob,
            hard_prob=fragile_aigc_hard_prob,
        )
        self.hard_real_indices = {int(idx) for idx in (hard_real_indices or [])}
        self.anchor_hard_real_indices = {int(idx) for idx in (anchor_hard_real_indices or [])}
        self.fragile_aigc_indices = {int(idx) for idx in (fragile_aigc_indices or [])}

    def __len__(self) -> int:
        return len(self.base_dataset)

    def set_hard_real_indices(self, indices: Sequence[int]) -> None:
        self.hard_real_indices = {int(idx) for idx in indices}

    def set_anchor_hard_real_indices(self, indices: Sequence[int]) -> None:
        self.anchor_hard_real_indices = {int(idx) for idx in indices}

    def set_fragile_aigc_indices(self, indices: Sequence[int]) -> None:
        self.fragile_aigc_indices = {int(idx) for idx in indices}

    @staticmethod
    def _sample_mode(clean_prob: float, mild_prob: float) -> str:
        r = random.random()
        if r < clean_prob:
            return "clean"
        if r < clean_prob + mild_prob:
            return "mild"
        return "hard"

    @staticmethod
    def _resolve_real_transform(
        mode: str,
        fallback_transform: Optional[Any],
        clean_transform: Optional[Any],
        mild_transform: Optional[Any],
        hard_transform: Optional[Any],
    ) -> Optional[Any]:
        candidates = {
            "clean": clean_transform,
            "mild": mild_transform,
            "hard": hard_transform,
        }
        if candidates.get(mode) is not None:
            return candidates[mode]
        for candidate_mode in ("clean", "mild", "hard"):
            candidate = candidates.get(candidate_mode)
            if candidate is not None:
                return candidate
        return fallback_transform

    @staticmethod
    def _apply_transform(arr: np.ndarray, transform: Optional[Any]) -> torch.Tensor:
        if transform is None:
            return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        transformed = transform(image=arr) if callable(transform) else None
        if isinstance(transformed, dict) and "image" in transformed:
            return transformed["image"]
        return transform(arr)

    def __getitem__(self, index: int):
        record = self.base_dataset.records[index]
        label = torch.tensor(record.label, dtype=torch.float32)
        metadata = dict(record.metadata)
        metadata["dataset_index"] = int(index)
        metadata["image_path"] = str(record.image_path)
        metadata["image_name"] = str(metadata.get("image_name", record.image_path.name))
        metadata["label"] = int(record.label)
        metadata["real_transform_policy"] = "not_applicable"
        metadata["real_transform_profile"] = "not_applicable"
        metadata["hard_real_buffer_hit"] = bool(index in self.hard_real_indices)
        metadata["anchor_hard_real_buffer_hit"] = bool(index in self.anchor_hard_real_indices)
        metadata["aigc_transform_policy"] = "not_applicable"
        metadata["aigc_transform_profile"] = "not_applicable"
        metadata["fragile_aigc_buffer_hit"] = bool(index in self.fragile_aigc_indices)

        arr = np.array(Image.open(record.image_path).convert("RGB"))
        primary_transform = self.transform
        secondary_transform = self.secondary_transform
        if record.label == 0:
            hard_real = bool(index in self.hard_real_indices)
            anchor_hard_real = bool(index in self.anchor_hard_real_indices)
            metadata["hard_real_buffer_hit"] = hard_real or anchor_hard_real
            metadata["anchor_hard_real_buffer_hit"] = anchor_hard_real
            if anchor_hard_real:
                metadata["real_transform_policy"] = "anchor_hard_real_buffer"
                real_mode = self._sample_mode(
                    clean_prob=self.anchor_hard_real_clean_prob,
                    mild_prob=self.anchor_hard_real_mild_prob,
                )
            elif hard_real:
                metadata["real_transform_policy"] = "hard_real_buffer"
                real_mode = self._sample_mode(
                    clean_prob=self.hard_real_clean_prob,
                    mild_prob=self.hard_real_mild_prob,
                )
            else:
                metadata["real_transform_policy"] = "standard_real"
                real_mode = self._sample_mode(
                    clean_prob=self.real_clean_prob,
                    mild_prob=self.real_mild_prob,
                )
            metadata["real_transform_profile"] = real_mode
            primary_transform = self._resolve_real_transform(
                real_mode,
                primary_transform,
                self.real_clean_transform,
                self.real_mild_transform,
                self.real_hard_transform,
            )
            secondary_transform = self._resolve_real_transform(
                real_mode,
                secondary_transform,
                self.secondary_real_clean_transform,
                self.secondary_real_mild_transform,
                self.secondary_real_hard_transform,
            )
        else:
            fragile_aigc = bool(index in self.fragile_aigc_indices)
            metadata["fragile_aigc_buffer_hit"] = fragile_aigc
            metadata["aigc_transform_policy"] = "fragile_aigc_buffer" if fragile_aigc else "standard_aigc"
            if fragile_aigc:
                aigc_mode = self._sample_mode(
                    clean_prob=self.fragile_aigc_clean_prob,
                    mild_prob=self.fragile_aigc_mild_prob,
                )
            else:
                aigc_mode = self._sample_mode(
                    clean_prob=self.aigc_clean_prob,
                    mild_prob=self.aigc_mild_prob,
                )
            metadata["aigc_transform_profile"] = aigc_mode
            primary_transform = self._resolve_real_transform(
                aigc_mode,
                primary_transform,
                self.aigc_clean_transform,
                self.aigc_mild_transform,
                self.aigc_hard_transform,
            )
            secondary_transform = self._resolve_real_transform(
                aigc_mode,
                secondary_transform,
                self.secondary_aigc_clean_transform,
                self.secondary_aigc_mild_transform,
                self.secondary_aigc_hard_transform,
            )
        image = self._apply_transform(arr, primary_transform)
        if secondary_transform is None:
            return image, label, metadata
        image2 = self._apply_transform(arr, secondary_transform)
        return image, label, metadata, image2


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
    hard_negative_names: Optional[Sequence[str]] = None,
    hard_negative_indices: Optional[Sequence[int]] = None,
    hard_negative_boost: float = 3.0,
) -> torch.DoubleTensor:
    df = dataset.to_dataframe().copy()
    df["_dataset_index"] = np.arange(len(df), dtype=np.int64)
    return build_balanced_sample_weights_from_dataframe(
        df,
        secondary_key_priority=secondary_key_priority,
        hard_negative_names=hard_negative_names,
        hard_negative_indices=hard_negative_indices,
        hard_negative_boost=hard_negative_boost,
    )


def build_balanced_sample_weights_from_dataframe(
    df: pd.DataFrame,
    secondary_key_priority: Sequence[str] = ("distortion", "source", "shard_name"),
    hard_negative_names: Optional[Sequence[str]] = None,
    hard_negative_indices: Optional[Sequence[int]] = None,
    hard_negative_boost: float = 3.0,
) -> torch.DoubleTensor:
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
    hard_negative_lookup = {str(name).lower() for name in (hard_negative_names or [])}
    hard_negative_index_lookup = {int(idx) for idx in (hard_negative_indices or [])}
    raw = []
    for _, row in df.iterrows():
        w = label_weight[row["label"]] * secondary_weight[row[secondary_key]]
        sample_name = _extract_sample_name(row).lower()
        sample_index = int(row["_dataset_index"]) if "_dataset_index" in row.index else None
        if sample_name in hard_negative_lookup:
            w *= float(hard_negative_boost)
        if sample_index is not None and sample_index in hard_negative_index_lookup:
            w *= float(hard_negative_boost)
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
