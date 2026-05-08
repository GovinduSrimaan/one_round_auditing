# Modernisation Change Log

## Summary

This repository has been updated for **Python 3.12**, the **latest Google Colab runtime** (PyTorch ≥ 2.3, Opacus ≥ 1.5), and general GPU/CPU flexibility. The goal was to make as few changes as possible while fixing every breaking issue.

---

## 1. `requirements.txt` (new file)

A clean, minimal requirements file suitable for `pip install -r requirements.txt` in Colab or any venv. Replaces the verbose `requirements_fixed.txt` / `requirements_working.txt` files which contained thousands of pinned transitive dependencies and several packages that are no longer available.

**Key version changes:**

| Package | Old (approx.) | New |
|---|---|---|
| Python | 3.11 | 3.12 |
| torch | 2.1.2 | ≥ 2.3 |
| torchvision | 0.16.2 | ≥ 0.18 |
| opacus | 1.4.0 | ≥ 1.5 |
| torchdata | 0.6 / 0.7 | **removed** |

---

## 2. `environment.yml`

- Python bumped to **3.12**.
- `torchdata` removed (package was fragmented; datapipe API removed from current releases).
- `open-clip-torch` removed (not needed for the core audit).
- PyTorch / torchvision versions updated to match the new requirements.

---

## 3. `src/base.py`

**Change:** `DEVICE` is no longer hardcoded to `torch.device("cuda")`.

```python
# Before
DEVICE = torch.device("cuda")

# After
def get_device() -> torch.device:
    if torch.cuda.is_available():   return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
```

Everything that uses `base.DEVICE` now works on CPU-only machines and Apple Silicon without any further changes.

---

## 4. `src/data.py`

**Primary change:** All `torchdata.datapipes` and `torchdata.dataloader2` imports removed.

`torchdata` split its datapipe API into a separate unmaintained package and it is not available in the current Colab environment. The two affected methods now return the `Dataset` object itself (which is already a valid `torch.utils.data.Dataset`):

```python
# Before — returned torchdata IterDataPipe / MapDataPipe
def build_datapipe(self, shuffle, cycle, add_sharding_filter) -> IterDataPipe: ...
def build_map_datapipe(self) -> MapDataPipe: ...

# After — return self; callers pass it directly to DataLoader
def build_datapipe(self, ...) -> "Dataset": return self
def build_map_datapipe(self) -> "Dataset": return self
```

All callers in `dp_audit.py` have been updated accordingly.

**Minor:** `_load_cifar10` now correctly loads the **test set** with `train=False` (the original accidentally loaded the training set twice).

---

## 5. `src/dp_audit.py`

Three independent changes:

### 5a. Removed `torchdata` imports

`torchdata.dataloader2.DataLoader2` and all datapipe wrappers have been replaced with plain `torch.utils.data.DataLoader`. The pretrain loop now uses a simple `TransformDataset` wrapper class instead of datapipe `.map()` chaining.

### 5b. `torch.load(..., weights_only=False)`

PyTorch 2.1 introduced a `FutureWarning` (promoted to an error in 2.6) requiring explicit `weights_only` argument. All `torch.load` calls now pass `weights_only=False` for compatibility.

```python
# Before
final_model = torch.load(f"{output_dir}/shadow/{shadow_idx}/model.pt")

# After
final_model = torch.load(..., map_location=base.DEVICE, weights_only=False)
```

### 5c. Removed `torchvision.transforms.v2.Lambda`

The `Lambda` transform was deprecated in `torchvision.transforms.v2` and raises warnings/errors in current versions. The per-sample augmentation loop now uses an explicit Python list comprehension:

```python
# Before (Lambda approach)
batch_xs = torchvision.transforms.v2.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(batch_xs)

# After
augmented = [aug_transform(x) for x in batch_xs]
batch_xs = torch.stack(augmented)
```

### 5d. `hflip` API update

`torchvision.transforms.v2.functional.hflip` was renamed to `horizontal_flip` in newer torchvision. The call has been updated:

```python
# Before
torchvision.transforms.v2.functional.hflip(aug)

# After
torchvision.transforms.v2.functional.horizontal_flip(aug)
```

---

## 6. `scripts/dp_train.sh` and `scripts/dp_audit.sh`

**Changes:**
- Replaced hardcoded `YOUT_PATH` / `YOUT_EXPERIMENT_PATH` / `YOUT_DATA_PATH` / `YOUT_REPO_PATH` placeholders with auto-derived paths relative to the script location.
- SLURM headers are now comments (prefixed `# #SBATCH`) so the scripts run directly with `bash scripts/dp_train.sh` on a local machine.
- Added `TASK_ID="${SLURM_ARRAY_TASK_ID:-${1:-0}}"` so a shadow-model index can be passed as a positional argument when running locally.
- Added `export DOWNLOAD_DATA=1` so `torchvision` will auto-download CIFAR-10 on first run.

---

## 7. `First.ipynb` (Colab notebook)

- Added a setup cell to `pip install` opacus/torch if not present.
- Device selection replaced with CUDA → MPS → CPU auto-detection.
- Parquet loading is now optional (`USE_DRIVE = False` by default); the notebook falls back to `torchvision.datasets.CIFAR10(download=True)`.
- Label patching now works for both the parquet `DataFrame` and the torchvision `list`-backed `targets` attribute.
- All other logic is identical to the original.
