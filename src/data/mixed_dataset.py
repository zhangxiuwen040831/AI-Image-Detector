import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset


class MixedAIGCDetectionDataset(Dataset):
    def __init__(self, index_path: str, transform: Callable | None = None) -> None:
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(f"dataset index not found: {self.index_path}")
        self.transform = transform
        self.records: List[Tuple[str, int, str, str]] = []
        self._source_to_indices: Dict[str, List[int]] = {"artifact": [], "cifake": []}
        self._load_index()

    def _load_index(self) -> None:
        with self.index_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        raw_records = obj.get("records", [])
        for i, item in enumerate(raw_records):
            path = item["path"]
            label = int(item["label"])
            source = item["source"]
            subtype = item["subtype"]
            self.records.append((path, label, source, subtype))
            if source in self._source_to_indices:
                self._source_to_indices[source].append(i)

    @property
    def source_to_indices(self) -> Dict[str, List[int]]:
        return self._source_to_indices

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        path, label, source, subtype = self.records[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        meta = {"source": source, "subtype": subtype, "path": path}
        return image, label, meta
