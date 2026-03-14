from typing import Tuple

import torch
from torch.utils.data import Sampler


class SourceBalancedSampler(Sampler[int]):
    def __init__(
        self,
        dataset,
        artifact_cifake_ratio: Tuple[int, int] = (3, 1),
        real_fake_ratio: Tuple[int, int] = (1, 1),
        num_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.artifact_cifake_ratio = artifact_cifake_ratio
        self.real_fake_ratio = real_fake_ratio
        self.num_samples = num_samples or len(dataset)
        self.seed = int(seed)
        self.epoch = 0
        self.real_indices, self.artifact_indices, self.cifake_indices, self.fake_other_indices = self._build_group_indices()
        if len(self.real_indices) == 0:
            raise RuntimeError("Real samples are required for real/fake balance")
        if len(self.artifact_indices) == 0 or len(self.cifake_indices) == 0:
            raise RuntimeError("Both artifact and cifake samples are required for balanced sampling")

    def _build_group_indices(self):
        real_indices = []
        fake_artifact_indices = []
        fake_cifake_indices = []
        fake_other_indices = []
        records = getattr(self.dataset, "records", [])
        for idx, record in enumerate(records):
            if len(record) < 4:
                continue
            _, label, source, _ = record
            if int(label) == 0:
                real_indices.append(idx)
                continue
            if str(source) == "artifact":
                fake_artifact_indices.append(idx)
            elif str(source) == "cifake":
                fake_cifake_indices.append(idx)
            else:
                fake_other_indices.append(idx)
        if len(records) == 0 and hasattr(self.dataset, "source_to_indices"):
            source_map = self.dataset.source_to_indices
            fake_artifact_indices = list(source_map.get("artifact", []))
            fake_cifake_indices = list(source_map.get("cifake", []))
        return real_indices, fake_artifact_indices, fake_cifake_indices, fake_other_indices

    @staticmethod
    def _draw_from_pool(pool, count, generator):
        if count <= 0:
            return []
        if len(pool) == 0:
            return []
        sampled_pos = torch.randint(0, len(pool), (count,), generator=generator).tolist()
        return [pool[i] for i in sampled_pos]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        real_ratio_sum = max(float(self.real_fake_ratio[0] + self.real_fake_ratio[1]), 1.0)
        real_target = int(round(self.num_samples * float(self.real_fake_ratio[0]) / real_ratio_sum))
        fake_target = self.num_samples - real_target
        fake_ratio_sum = max(float(self.artifact_cifake_ratio[0] + self.artifact_cifake_ratio[1]), 1.0)
        artifact_target = int(round(fake_target * float(self.artifact_cifake_ratio[0]) / fake_ratio_sum))
        cifake_target = fake_target - artifact_target

        sampled_real = self._draw_from_pool(self.real_indices, real_target, generator)
        sampled_artifact = self._draw_from_pool(self.artifact_indices, artifact_target, generator)
        sampled_cifake = self._draw_from_pool(self.cifake_indices, cifake_target, generator)

        sampled = sampled_real + sampled_artifact + sampled_cifake
        missing = self.num_samples - len(sampled)
        if missing > 0:
            fallback_fake_pool = self.artifact_indices + self.cifake_indices + self.fake_other_indices
            sampled.extend(self._draw_from_pool(fallback_fake_pool + self.real_indices, missing, generator))
        shuffle_order = torch.randperm(len(sampled), generator=generator).tolist()
        sampled = [sampled[i] for i in shuffle_order]
        return iter(sampled[: self.num_samples])

    def __len__(self) -> int:
        return self.num_samples

    def inspect_batch_composition(self, indices):
        real_count = 0
        fake_count = 0
        artifact_count = 0
        cifake_count = 0
        records = getattr(self.dataset, "records", [])
        for idx in indices:
            if len(records) == 0:
                break
            _, label, source, _ = records[idx]
            if int(label) == 0:
                real_count += 1
            else:
                fake_count += 1
                if str(source) == "artifact":
                    artifact_count += 1
                elif str(source) == "cifake":
                    cifake_count += 1
        total = max(len(indices), 1)
        fake_total = max(fake_count, 1)
        return {
            "real_ratio": real_count / total,
            "fake_ratio": fake_count / total,
            "artifact_ratio_in_fake": artifact_count / fake_total,
            "cifake_ratio_in_fake": cifake_count / fake_total,
        }
