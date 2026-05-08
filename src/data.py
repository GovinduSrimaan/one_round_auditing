"""data.py — dataset helpers.

Changes vs original
--------------------
* Removed all ``torchdata.datapipes`` / ``torchdata.dataloader2`` imports.
  These sub-packages were removed in torchdata ≥ 0.7 and are unavailable in
  current Colab / PyTorch 2.3+ environments.
* ``Dataset.build_datapipe()`` and ``Dataset.build_map_datapipe()`` now return
  plain ``torch.utils.data.Subset``-style wrappers so callers that previously
  consumed a datapipe still receive something iterable.
* All other logic (canary types, poison types, splits) is unchanged.
"""

import enum
import hashlib
import math
import pathlib
import typing

import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.v2

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class CanaryType(enum.Enum):
    CLEAN = "clean"
    LABEL_NOISE = "label_noise"
    BLANK_IMAGES = "blank_images"
    OOD = "ood"
    SSL_WORST_CASE = "ssl"
    DUPLICATES_MISLABEL_HALF = "duplicates_mislabel_half"
    DUPLICATES_MISLABEL_FULL = "duplicates_mislabel_full"


class PoisonType(enum.Enum):
    RANDOM_IMAGES = "random_images"
    CANARY_DUPLICATES = "canary_duplicates"
    CANARY_DUPLICATES_NOISY = "canary_duplicates_noisy"
    NONCANARY_DUPLICATES_NOISY = "noncanary_duplicates_noisy"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        labels: typing.Optional[torch.Tensor] = None,
        indices: typing.Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self._features = features
        self._labels = labels
        self._indices = (
            indices.to(dtype=torch.long)
            if indices is not None
            else torch.arange(len(features), dtype=torch.long)
        )
        if self._labels is not None:
            assert len(self._features) == len(self._labels)
        if self._indices is not None:
            assert 0 <= int(self._indices.min()) and int(self._indices.max()) < len(self._features)

    def __getitem__(self, index: int):
        target_idx = self._indices[index]
        if self._labels is not None:
            return self._features[target_idx], self._labels[target_idx]
        return self._features[target_idx]

    def __len__(self):
        return len(self._indices)

    def as_unlabeled(self) -> "Dataset":
        return Dataset(features=self._features, labels=None, indices=self._indices)

    # ------------------------------------------------------------------
    # Datapipe replacements — return plain Dataset so DataLoader still works
    # ------------------------------------------------------------------

    def build_datapipe(
        self,
        shuffle: bool = False,
        cycle: bool = False,
        add_sharding_filter: bool = False,
    ) -> "Dataset":
        """Return self (a plain Dataset). Shuffle/cycle args accepted but ignored at
        construction time; pass ``shuffle=True`` to the DataLoader instead."""
        return self

    def build_map_datapipe(self) -> "Dataset":
        """Return self — callers should pass this directly to DataLoader."""
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def targets(self) -> torch.Tensor:
        if not self.is_labeled:
            raise ValueError("Dataset is not labeled")
        return self._labels[self._indices]

    @property
    def is_labeled(self) -> bool:
        return self._labels is not None

    def subset(
        self,
        indices: torch.Tensor,
        labels: typing.Optional[torch.Tensor] = None,
    ) -> "Dataset":
        assert torch.all(indices < len(self))
        selection_indices = self._indices[indices]
        new_features = self._features[selection_indices]
        new_labels = labels if labels is not None else (
            self._labels[selection_indices] if self._labels is not None else None
        )
        assert new_labels is None or len(new_labels) == len(indices)
        return Dataset(features=new_features, labels=new_labels)


class SSLDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            pos_1 = self.transform(data)
            pos_2 = self.transform(data)
        else:
            pos_1 = data
            pos_2 = data
        return pos_1, pos_2, target


# ---------------------------------------------------------------------------
# DatasetGenerator
# ---------------------------------------------------------------------------

class DatasetGenerator:
    _OOD_SAMPLES_MD5 = "54ac377793b3aa2910b106f128aa9142"
    _SSL_INDICES_MD5 = "dcc03147693d5d4c4f8f185fbd590b37"

    def __init__(
        self,
        num_shadow: int,
        num_canaries: int,
        canary_type: CanaryType,
        num_poison: int,
        poison_type: PoisonType,
        data_dir: pathlib.Path,
        seed: int,
        download: bool = False,
        fixed_halves: typing.Optional[bool] = None,
        method: str = None,
    ) -> None:
        self.method = method
        self._seed = seed
        self._num_shadow = num_shadow
        self._num_canaries = num_canaries
        self._canary_type = canary_type
        self._num_poison = num_poison
        self._poison_type = poison_type
        self._data_root = data_dir
        self._download = download

        self._clean_train_xs, self._clean_train_ys = self._load_cifar10(data_dir, train=True, download=download)
        self._test_xs, self._test_ys = self._load_cifar10(data_dir, train=False, download=download)

        if self._canary_type == CanaryType.OOD:
            ood_samples_path = data_dir / "ood_imagenet_samples.pt"
            if not ood_samples_path.exists():
                raise FileNotFoundError(f"OOD canary image file {ood_samples_path} does not exist")
            ood_samples_signature = hashlib.md5(ood_samples_path.read_bytes()).hexdigest()
            if ood_samples_signature != self._OOD_SAMPLES_MD5:
                raise ValueError(f"OOD canary image file {ood_samples_path} has wrong MD5 hash")
            self._ood_xs = torch.load(data_dir / "ood_imagenet_samples.pt")
            self._ood_xs = self._ood_xs.permute(0, 3, 1, 2)
            assert self._ood_xs.dtype == torch.uint8
            assert self._ood_xs.size()[1:] == (3, 32, 32)
            if self._ood_xs.size(0) < num_canaries:
                raise ValueError(f"Requested {num_canaries} OOD canaries, but only {self._ood_xs.size(0)} available")
        elif self._canary_type == CanaryType.SSL_WORST_CASE:
            ssl_indices_path = data_dir / "ssl_indices.pt"
            if not ssl_indices_path.exists():
                raise FileNotFoundError(f"SSL indices file {ssl_indices_path} does not exist")
            ssl_indices_signature = hashlib.md5(ssl_indices_path.read_bytes()).hexdigest()
            if ssl_indices_signature != self._SSL_INDICES_MD5:
                raise ValueError(f"SSL indices file {ssl_indices_path} has wrong MD5 hash")
            self._ssl_indices = torch.load(data_dir / "ssl_indices.pt")

        assert len(self._clean_train_xs) == 50000
        assert len(self._test_xs) == 10000

        rng = np.random.default_rng(seed=self._seed)
        num_raw_train_samples = len(self._clean_train_xs)
        num_classes = 10

        rng_splits_target, rng_splits_shadow, rng = rng.spawn(3)
        del rng_splits_target
        assert self._num_shadow % 2 == 0
        shadow_in_indices_t = np.argsort(
            rng_splits_shadow.uniform(size=(self._num_shadow, num_raw_train_samples)), axis=0
        )[: self._num_shadow // 2].T
        raw_shadow_in_indices = []
        for shadow_idx in range(self._num_shadow):
            raw_shadow_in_indices.append(
                torch.from_numpy(np.argwhere(np.any(shadow_in_indices_t == shadow_idx, axis=1)).flatten())
            )
        rng_splits_half, rng_splits_shadow = rng_splits_shadow.spawn(2)
        del rng_splits_shadow

        rng_canaries, rng = rng.spawn(2)
        self._canary_order = rng_canaries.permutation(num_raw_train_samples)
        if self._canary_type == CanaryType.SSL_WORST_CASE:
            self._canary_order = self._ssl_indices
        del rng_canaries

        self._shadow_in_indices = []
        if fixed_halves is None:
            canary_mask = torch.zeros(num_raw_train_samples, dtype=torch.bool)
            canary_mask[self._canary_order[: self._num_canaries]] = True
            for shadow_idx in range(self._num_shadow):
                current_in_mask = torch.zeros(num_raw_train_samples, dtype=torch.bool)
                current_in_mask[raw_shadow_in_indices[shadow_idx]] = True
                current_in_mask[~canary_mask] = True
                self._shadow_in_indices.append(torch.argwhere(current_in_mask).flatten())
        else:
            if not fixed_halves:
                self._shadow_in_indices = raw_shadow_in_indices
            else:
                canary_mask = torch.zeros(num_raw_train_samples, dtype=torch.bool)
                canary_mask[self._canary_order[: self._num_canaries]] = True
                fixed_membership_full = torch.from_numpy(rng_splits_half.random(num_raw_train_samples) < 0.5)
                for shadow_idx in range(self._num_shadow):
                    current_in_mask = torch.zeros(num_raw_train_samples, dtype=torch.bool)
                    current_in_mask[raw_shadow_in_indices[shadow_idx]] = True
                    current_in_mask[~canary_mask] = False
                    current_in_mask[(~canary_mask) & fixed_membership_full] = True
                    self._shadow_in_indices.append(torch.argwhere(current_in_mask).flatten())
        del rng_splits_half

        rng_canary_transforms, rng = rng.spawn(2)
        rng_noise, rng_canary_transforms = rng_canary_transforms.spawn(2)
        label_changes = torch.from_numpy(rng_noise.integers(num_classes - 1, size=num_raw_train_samples))
        self._noisy_labels = torch.where(label_changes < self._clean_train_ys, label_changes, label_changes + 1)
        del rng_noise
        rng_blank_images, rng_canary_transforms = rng_canary_transforms.spawn(2)
        self._blank_image_colors = torch.from_numpy(
            rng_blank_images.integers(0, 256, size=(num_raw_train_samples, 3, 1, 1), dtype=np.uint8)
        )
        self._blank_image_labels = torch.from_numpy(
            rng_blank_images.integers(0, num_classes, size=num_raw_train_samples)
        )
        del rng_blank_images, rng_canary_transforms

        rng_poison, rng = rng.spawn(2)
        rng_poison_images, rng_poison = rng_poison.spawn(2)
        self._poison_random_images_seed = rng_poison_images.integers(0, 2**32, dtype=np.uint32)
        del rng_poison_images
        rng_poison_duplicates_labels, rng_poison = rng_poison.spawn(2)
        self._poison_duplicate_label_changes = torch.from_numpy(
            rng_poison_duplicates_labels.integers(num_classes - 1, size=num_raw_train_samples)
        )
        del rng_poison_duplicates_labels, rng_poison, rng

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_cifar10(self, data_dir: pathlib.Path, train: bool, download: bool):
        raw_dataset = torchvision.datasets.CIFAR10(
            str(data_dir),
            train=train,
            transform=torchvision.transforms.v2.Compose([
                torchvision.transforms.v2.ToImage(),
                torchvision.transforms.v2.ToDtype(torch.uint8, scale=True),
            ]),
            download=download,
        )
        xs = torch.empty((len(raw_dataset), 3, 32, 32), dtype=torch.uint8)
        ys = torch.empty((len(raw_dataset),), dtype=torch.long)
        for idx, (x, y) in enumerate(raw_dataset):
            xs[idx] = x
            ys[idx] = y
        return xs, ys

    def _build_train_canary_data(self, canaries_dir=None):
        if self._canary_type == CanaryType.CLEAN:
            return self._clean_train_xs, self._clean_train_ys
        elif self._canary_type == CanaryType.LABEL_NOISE:
            noisy_targets = self._clean_train_ys.clone()
            canary_indices = self.get_canary_indices()
            noisy_targets[canary_indices] = self._noisy_labels[canary_indices]
            return self._clean_train_xs, noisy_targets
        elif self._canary_type == CanaryType.BLANK_IMAGES:
            canary_features = self._clean_train_xs.clone()
            canary_targets = self._clean_train_ys.clone()
            canary_indices = self.get_canary_indices()
            canary_features[canary_indices] = self._blank_image_colors[canary_indices]
            canary_targets[canary_indices] = self._blank_image_labels[canary_indices]
            return canary_features, canary_targets
        elif self._canary_type == CanaryType.OOD:
            canary_features = self._clean_train_xs.clone()
            canary_targets = self._clean_train_ys.clone()
            canary_indices = self.get_canary_indices()
            assert canary_indices.shape[0] <= self._ood_xs.size(0)
            canary_features[canary_indices] = self._ood_xs[: canary_indices.shape[0]]
            return canary_features, canary_targets
        elif self._canary_type == CanaryType.SSL_WORST_CASE:
            return self._clean_train_xs, self._clean_train_ys
        elif self._canary_type in (CanaryType.DUPLICATES_MISLABEL_HALF, CanaryType.DUPLICATES_MISLABEL_FULL):
            canary_features = self._clean_train_xs.clone()
            canary_targets = self._clean_train_ys.clone()
            canary_indices = self.get_canary_indices()
            assert len(canary_indices) % 2 == 0
            canary_indices_original = canary_indices[: self._num_canaries // 2]
            canary_indices_duplicate = canary_indices[self._num_canaries // 2 :]
            canary_features[canary_indices_duplicate] = canary_features[canary_indices_original]
            canary_targets[canary_indices_duplicate] = self._noisy_labels[canary_indices_original]
            if self._canary_type == CanaryType.DUPLICATES_MISLABEL_FULL:
                canary_targets[canary_indices_original] = canary_targets[canary_indices_duplicate]
            return canary_features, canary_targets
        else:
            raise ValueError(f"Unknown canary type {self._canary_type}")

    def _build_poison_random_images(self):
        rng = np.random.default_rng(self._poison_random_images_seed)
        num_classes = 10
        num_poison_total = math.ceil(self._num_poison / num_classes) * num_classes
        num_random_images = num_poison_total // num_classes
        poison_targets = torch.tile(torch.arange(num_classes, dtype=torch.long), (num_random_images,))
        poison_features = torch.zeros((num_poison_total, 3, 32, 32), dtype=torch.uint8)
        for idx in range(num_random_images):
            poison_features[idx * num_classes : (idx + 1) * num_classes] = torch.from_numpy(
                rng.integers(0, 256, size=(1, 3, 32, 32), dtype=np.uint8)
            )
        return poison_features, poison_targets

    def _build_train_data(self, indices, include_poison, canaries_dir=None):
        train_xs, train_ys = self._build_train_canary_data(canaries_dir)
        if include_poison and self._num_poison > 0:
            if self._poison_type == PoisonType.RANDOM_IMAGES:
                poison_xs, poison_ys = self._build_poison_random_images()
                poison_indices = torch.arange(len(train_xs), len(train_xs) + len(poison_xs))
                train_xs = torch.cat([train_xs, poison_xs], dim=0)
                train_ys = torch.cat([train_ys, poison_ys], dim=0)
            elif self._poison_type == PoisonType.CANARY_DUPLICATES:
                assert self._num_poison == self._num_canaries
                poison_indices = torch.from_numpy(self.get_canary_indices())
            elif self._poison_type == PoisonType.CANARY_DUPLICATES_NOISY:
                assert self._num_poison == self._num_canaries
                canary_indices = self.get_canary_indices()
                poison_xs = train_xs[canary_indices]
                canary_ys = train_ys[canary_indices]
                label_changes = self._poison_duplicate_label_changes[canary_indices]
                poison_ys = torch.where(label_changes < canary_ys, label_changes, label_changes + 1)
                poison_indices = torch.arange(len(train_xs), len(train_xs) + len(poison_xs))
                train_xs = torch.cat([train_xs, poison_xs], dim=0)
                train_ys = torch.cat([train_ys, poison_ys], dim=0)
            elif self._poison_type == PoisonType.NONCANARY_DUPLICATES_NOISY:
                assert self._num_poison + self._num_canaries <= train_xs.size(0)
                duplicate_indices = self.get_non_canary_indices()[-self._num_poison :]
                poison_xs = train_xs[duplicate_indices]
                duplicate_ys = train_ys[duplicate_indices]
                label_changes = self._poison_duplicate_label_changes[duplicate_indices]
                poison_ys = torch.where(label_changes < duplicate_ys, label_changes, label_changes + 1)
                poison_indices = torch.arange(len(train_xs), len(train_xs) + len(poison_xs))
                train_xs = torch.cat([train_xs, poison_xs], dim=0)
                train_ys = torch.cat([train_ys, poison_ys], dim=0)
            else:
                raise ValueError(f"Unknown poison type {self._poison_type}")
            assert poison_indices.size() == (self._num_poison,)
            indices = torch.cat((indices, poison_indices))
        return Dataset(train_xs, train_ys, indices=indices)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_test_data(self) -> Dataset:
        return Dataset(self._test_xs, self._test_ys)

    def build_full_train_data(self, canaries_dir=None) -> Dataset:
        return self._build_train_data(
            indices=torch.arange(self.num_raw_training_samples),
            include_poison=False,
            canaries_dir=canaries_dir,
        )

    def build_train_data(self, shadow_model_idx: int, canaries_dir=None) -> Dataset:
        return self._build_train_data(
            indices=self._shadow_in_indices[shadow_model_idx],
            include_poison=True,
            canaries_dir=canaries_dir,
        )

    def build_attack_data(self, canaries_dir=None):
        _, attack_ys = self._build_train_canary_data(canaries_dir)
        canary_indices = torch.from_numpy(self.get_canary_indices())
        if self._canary_type == CanaryType.DUPLICATES_MISLABEL_HALF:
            assert len(canary_indices) % 2 == 0
            canary_indices = canary_indices[-self._num_canaries // 2 :]
        shadow_membership_mask = torch.zeros((self.num_raw_training_samples, self._num_shadow), dtype=torch.bool)
        for shadow_model_idx in range(self._num_shadow):
            shadow_membership_mask[:, shadow_model_idx] = self.build_in_mask(shadow_model_idx)
        return attack_ys, shadow_membership_mask, canary_indices

    def build_in_mask(self, shadow_model_idx: int) -> torch.Tensor:
        result = torch.zeros(self.num_raw_training_samples, dtype=torch.bool)
        result[self._shadow_in_indices[shadow_model_idx]] = True
        return result

    def get_canary_indices(self) -> np.ndarray:
        return self._canary_order[: self._num_canaries]

    def get_non_canary_indices(self) -> np.ndarray:
        return self._canary_order[self._num_canaries :]

    def load_cifar100(self):
        return torchvision.datasets.CIFAR100(
            root=str(self._data_root),
            train=True,
            download=self._download,
            transform=torchvision.transforms.v2.Compose([
                torchvision.transforms.v2.ToImage(),
                torchvision.transforms.v2.ToDtype(torch.uint8, scale=True),
            ]),
        )

    def build_train_ssl_data(self, shadow_model_idx, transform, canaries_dir=None):
        indices = self._shadow_in_indices[shadow_model_idx]
        train_xs, train_ys = self._build_train_canary_data(canaries_dir)
        return SSLDataset(train_xs[indices], train_ys[indices], transform)

    def build_full_train_ssl_data(self, transform=None, canaries_dir=None):
        train_xs, train_ys = self._build_train_canary_data(canaries_dir)
        return SSLDataset(train_xs, train_ys, transform)

    @property
    def num_raw_training_samples(self) -> int:
        return len(self._clean_train_xs)

    @property
    def num_shadow(self) -> int:
        return self._num_shadow

    @property
    def num_canaries(self) -> int:
        return self._num_canaries

    @property
    def canary_type(self) -> CanaryType:
        return self._canary_type

    @property
    def num_poison(self) -> int:
        return self._num_poison

    @property
    def poison_type(self) -> PoisonType:
        return self._poison_type
