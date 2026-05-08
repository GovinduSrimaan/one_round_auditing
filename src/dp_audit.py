"""dp_audit.py — DP-SGD training and privacy auditing.

Changes vs original
--------------------
* Removed all ``torchdata.dataloader2`` and ``torchdata.datapipes`` imports.
  The pretrain loop now uses a plain ``torch.utils.data.DataLoader``.
* ``_predict`` no longer wraps datasets in datapipes; it uses plain DataLoader.
* Device selection uses ``base.get_device()`` (CUDA → MPS → CPU) instead of
  hardcoding ``cuda``.
* ``torch.load`` calls now pass ``weights_only=False`` to silence the
  FutureWarning introduced in PyTorch 2.1+.
* Minor: replaced ``torchvision.transforms.v2.Lambda`` with a regular loop
  (Lambda transform was deprecated and behaves unexpectedly under v2).
* All other logic (DP-SGD, EMA, audit statistics) is unchanged.
"""

import argparse
import json
import math
import os
import pathlib
import typing
import warnings

warnings.filterwarnings("ignore")

import filelock
import mlflow
import numpy as np
import opacus
import opacus.utils.batch_memory_manager
import scipy.stats
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms.v2
import tqdm

import base
import data
import dpsgd_utils
import models


# ---------------------------------------------------------------------------
# Audit statistics (unchanged from original)
# ---------------------------------------------------------------------------

def p_value_DP_audit(m, r, v, eps, delta):
    assert 0 <= v <= r <= m
    assert eps >= 0
    assert 0 <= delta <= 1
    q = 1 / (1 + math.exp(-eps))
    beta = scipy.stats.binom.sf(v - 1, r, q)
    alpha = 0
    total_sum = 0
    for i in range(1, v + 1):
        total_sum += scipy.stats.binom.pmf(v - i, r, q)
        if total_sum > i * alpha:
            alpha = total_sum / i
    return min(beta + alpha * delta * 2 * m, 1)


def get_eps_audit(m, r, v, delta, p):
    assert 0 <= v <= r <= m
    assert 0 <= delta <= 1
    assert 0 < p < 1
    eps_min, eps_max = 0, 1
    while p_value_DP_audit(m, r, v, eps_max, delta) < p:
        eps_max += 1
    for _ in range(30):
        eps = (eps_min + eps_max) / 2
        if p_value_DP_audit(m, r, v, eps, delta) < p:
            eps_min = eps
        else:
            eps_max = eps
    return eps_min


# ---------------------------------------------------------------------------
# Audit loop
# ---------------------------------------------------------------------------

def _audit(args, data_generator, directory_manager):
    output_dir = directory_manager.get_attack_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    canaries_dir = directory_manager.get_auto_canaries_dir()
    attack_ys, shadow_membership_mask, canary_indices = data_generator.build_attack_data(canaries_dir)

    assert shadow_membership_mask.size() == (data_generator.num_raw_training_samples, data_generator.num_shadow)

    best_eps = []
    for shadow_idx in range(0, 31):
        print(f"Shadow model {shadow_idx}")
        initial_model = _build_model(num_classes=10)
        final_model = torch.load(
            f"{output_dir}/shadow/{shadow_idx}/model.pt",
            map_location=base.DEVICE,
            weights_only=False,
        )

        print("Predicting logits on full training data")
        full_train_data = data_generator.build_full_train_data(canaries_dir=canaries_dir)
        init_pred_ = _predict(initial_model, full_train_data.as_unlabeled(), data_augmentation=False).mean(dim=1)
        final_pred_ = _predict(final_model, full_train_data.as_unlabeled(), data_augmentation=False).mean(dim=1)

        init_loss_ = F.cross_entropy(init_pred_, attack_ys, reduction="none")
        final_loss_ = F.cross_entropy(final_pred_, attack_ys, reduction="none")
        sorted_scores_full = init_loss_ - final_loss_
        sorted_scores = sorted_scores_full[canary_indices]
        np.save(f"{output_dir}/sorted_scores.npy", sorted_scores.detach().cpu().numpy())

        canary_idx = shadow_membership_mask[:, shadow_idx][canary_indices] * 1
        in_idx = torch.where(canary_idx == 1)[0]
        out_idx = torch.where(canary_idx == 0)[0]

        delta = 1e-5
        p = 0.05
        m = args.num_canaries

        sort_idx = torch.argsort(sorted_scores, descending=True)
        best_ep = -1
        for k_bottom in range(10, 200, 10):
            for k_top in [80]:
                top_k_idx = sort_idx[:k_top].numpy()
                bottom_k_idx = sort_idx[-k_bottom:].numpy()
                count_in = np.sum(np.isin(top_k_idx, in_idx))
                count_out = np.sum(np.isin(bottom_k_idx, out_idx))
                count = count_in + count_out
                r = k_top + k_bottom
                v = count
                if v <= r <= m:
                    eps_min = get_eps_audit(m, r, v, delta, p)
                    print(f"  top_k={k_top} correct={count_in}, bottom_k={k_bottom} correct={count_out} → eps≥{eps_min:.4f}")
                best_ep = max(best_ep, eps_min) if "eps_min" in dir() else best_ep
        best_eps.append(best_ep)

    for i, ep in enumerate(best_eps):
        print(f"shadow model {i}, best eps: {ep}")
    print(f"average eps: {sum(best_eps) / len(best_eps)}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _run_train(args, shadow_model_idx, data_generator, directory_manager, training_seed, experiment_name, run_suffix, verbose):
    num_epochs = args.num_epochs
    noise_multiplier = args.noise_multiplier
    max_grad_norm = args.max_grad_norm
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    augmult_factor = args.augmult_factor
    pretrain_epochs = args.pretraining_epochs

    print(f"Training shadow model {shadow_model_idx}")

    output_dir = directory_manager.get_training_output_dir(shadow_model_idx)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = directory_manager.get_training_log_dir(shadow_model_idx)
    log_dir.mkdir(parents=True, exist_ok=True)
    canaries_dir = directory_manager.get_auto_canaries_dir()
    train_data = data_generator.build_train_data(shadow_model_idx=shadow_model_idx, canaries_dir=canaries_dir)

    with filelock.FileLock(log_dir / "enter_mlflow.lock"):
        mlflow.set_tracking_uri(f"file:{log_dir}")
        mlflow.set_experiment(experiment_name=experiment_name)
        run_name = f"train_{shadow_model_idx}" + (f"_{run_suffix}" if run_suffix else "")
        run = mlflow.start_run(run_name=run_name)

    with run:
        mlflow.log_params({
            "shadow_model_idx": shadow_model_idx,
            "num_canaries": data_generator.num_canaries,
            "canary_type": data_generator.canary_type.value,
            "num_poison": data_generator.num_poison,
            "poison_type": data_generator.poison_type.value,
            "training_seed": training_seed,
            "num_epochs": num_epochs,
            "noise_multiplier": noise_multiplier,
            "max_grad_norm": max_grad_norm,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "augmult_factor": augmult_factor,
        })

        current_model = _train_model(
            train_data, data_generator=data_generator, training_seed=training_seed,
            num_epochs=num_epochs, noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm,
            learning_rate=learning_rate, batch_size=batch_size, augmult_factor=augmult_factor,
            pretrain_epochs=pretrain_epochs, verbose=verbose,
        )
        current_model.eval()
        torch.save(current_model, output_dir / "model.pt")
        print("Saved model")

        metrics = {}
        full_train_data = data_generator.build_full_train_data(canaries_dir=canaries_dir)
        train_pred_full = _predict(current_model, full_train_data.as_unlabeled(), data_augmentation=True)
        torch.save(train_pred_full, output_dir / "predictions_train.pt")

        train_membership_mask = data_generator.build_in_mask(shadow_model_idx)
        train_ys_pred = torch.argmax(train_pred_full[:, 0], dim=-1)
        train_ys = full_train_data.targets
        correct = torch.eq(train_ys_pred, train_ys).to(dtype=base.DTYPE_EVAL)
        metrics.update({
            "train_accuracy_full": torch.mean(correct).item(),
            "train_accuracy_in": torch.mean(correct[train_membership_mask]).item(),
            "train_accuracy_out": torch.mean(correct[~train_membership_mask]).item(),
        })
        print(f"Train accuracy full={metrics['train_accuracy_full']:.4f} IN={metrics['train_accuracy_in']:.4f} OUT={metrics['train_accuracy_out']:.4f}")

        canary_mask = torch.zeros_like(train_membership_mask)
        canary_mask[data_generator.get_canary_indices()] = True
        metrics.update({
            "train_accuracy_canaries": torch.mean(correct[canary_mask]).item(),
            "train_accuracy_canaries_in": torch.mean(correct[canary_mask & train_membership_mask]).item(),
            "train_accuracy_canaries_out": torch.mean(correct[canary_mask & (~train_membership_mask)]).item(),
        })

        test_metrics, test_pred = _evaluate_model_test(current_model, data_generator)
        metrics.update(test_metrics)
        print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
        torch.save(test_pred, output_dir / "predictions_test.pt")
        mlflow.log_metrics(metrics, step=num_epochs)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)


def _evaluate_model_test(model, data_generator, disable_tqdm=False):
    test_data = data_generator.build_test_data()
    test_ys = test_data.targets
    test_pred = _predict(model, test_data.as_unlabeled(), data_augmentation=False, disable_tqdm=disable_tqdm)
    test_ys_pred = torch.argmax(test_pred[:, 0], dim=-1)
    correct = torch.eq(test_ys_pred, test_ys).to(base.DTYPE_EVAL)
    return {"test_accuracy": torch.mean(correct).item()}, test_pred


def _train_model(train_data, data_generator, training_seed, num_epochs, noise_multiplier, max_grad_norm,
                 learning_rate, batch_size, augmult_factor, pretrain_epochs, verbose=False):
    momentum = 0
    weight_decay = 0

    if pretrain_epochs is not None:
        model, pretrain_metrics = _pretrain(pretrain_epochs, data_generator, training_seed, verbose)
        mlflow.log_metrics(pretrain_metrics, step=0)
    else:
        model = _build_model(num_classes=10)
    model.train()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Secure RNG turned off")
        privacy_engine = dpsgd_utils.PrivacyEngineAugmented(opacus.GradSampleModule.GRAD_SAMPLERS)

    normalize = torchvision.transforms.v2.Compose([
        torchvision.transforms.v2.ToDtype(base.DTYPE, scale=True),
        torchvision.transforms.v2.Normalize(mean=data.CIFAR10_MEAN, std=data.CIFAR10_STD),
    ])

    # Apply normalization transform to the dataset via a wrapper
    class NormalizedDataset(torch.utils.data.Dataset):
        def __init__(self, ds, transform):
            self.ds = ds
            self.transform = transform
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            x, y = self.ds[idx]
            return self.transform(x), y

    norm_train_data = NormalizedDataset(train_data, normalize)

    loss = torch.nn.CrossEntropyLoss(reduction="mean")
    train_loader = torch.utils.data.DataLoader(
        norm_train_data, drop_last=False, num_workers=2, batch_size=batch_size, shuffle=True,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    dp_delta = 1e-5
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model, optimizer=optimizer, data_loader=train_loader,
        noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm,
        poisson_sampling=True, K=augmult_factor, loss_reduction=loss.reduction,
    )

    augmentation = dpsgd_utils.AugmentationMultiplicity(augmult_factor if augmult_factor > 0 else 1)
    model.GRAD_SAMPLERS[torch.nn.modules.conv.Conv2d] = augmentation.augmented_compute_conv_grad_sample
    model.GRAD_SAMPLERS[torch.nn.modules.linear.Linear] = augmentation.augmented_compute_linear_grad_sample
    model.GRAD_SAMPLERS[torch.nn.GroupNorm] = augmentation.augmented_compute_group_norm_grad_sample
    ema_model = dpsgd_utils.create_ema(model)

    aug_transform = torchvision.transforms.v2.Compose([
        torchvision.transforms.v2.RandomCrop(32, padding=4),
        torchvision.transforms.v2.RandomHorizontalFlip(),
    ])

    num_true_updates = 0
    for epoch in (pbar := tqdm.trange(num_epochs, desc="Training", unit="epoch")):
        model.train()
        with opacus.utils.batch_memory_manager.BatchMemoryManager(
            data_loader=train_loader, max_physical_batch_size=64, optimizer=optimizer,
        ) as opt_loader:
            num_samples = 0
            epoch_loss = epoch_accuracy = 0.0
            for batch_xs, batch_ys in tqdm.tqdm(opt_loader, desc="Epoch", unit="batch", leave=False, disable=not verbose):
                batch_xs = batch_xs.to(base.DEVICE)
                batch_ys = batch_ys.to(base.DEVICE)
                original_batch_size = batch_xs.size(0)
                optimizer.zero_grad(set_to_none=True)

                if augmult_factor > 0:
                    batch_xs = torch.repeat_interleave(batch_xs, repeats=augmult_factor, dim=0)
                    batch_ys = torch.repeat_interleave(batch_ys, repeats=augmult_factor, dim=0)
                    # Apply augmentation per-sample (no Lambda to avoid deprecation)
                    augmented = []
                    for x in batch_xs:
                        augmented.append(aug_transform(x))
                    batch_xs = torch.stack(augmented)
                    assert batch_xs.size(0) == augmult_factor * original_batch_size

                batch_pred = model(batch_xs)
                batch_loss = loss(input=batch_pred, target=batch_ys)
                batch_loss.backward()

                will_update = not optimizer._check_skip_next_step(pop_next=False)
                optimizer.step()
                if will_update:
                    num_true_updates += 1
                    dpsgd_utils.update_ema(model, ema_model, num_true_updates)

                epoch_loss += batch_loss.item() * batch_xs.size(0)
                epoch_accuracy += (batch_pred.detach().argmax(-1) == batch_ys).int().sum().item()
                num_samples += batch_xs.size(0)

            epoch_loss /= num_samples
            epoch_accuracy /= num_samples
            dp_eps = privacy_engine.get_epsilon(dp_delta)
            progress_dict = {
                "epoch_loss": epoch_loss, "epoch_accuracy": epoch_accuracy,
                "dp_eps": dp_eps, "dp_delta": dp_delta, "update_steps": num_true_updates,
            }
            mlflow.log_metrics(progress_dict, step=epoch + 1)
            pbar.set_postfix(progress_dict)

    ema_model.eval()
    return ema_model


def _pretrain(pretrain_epochs, data_generator, training_seed, verbose):
    batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4

    print("Pre-training model on CIFAR-100")
    model = _build_model(num_classes=100)
    raw_dataset = data_generator.load_cifar100()

    transform = torchvision.transforms.v2.Compose([
        torchvision.transforms.v2.RandomCrop(32, padding=4),
        torchvision.transforms.v2.RandomHorizontalFlip(),
        torchvision.transforms.v2.ToDtype(base.DTYPE, scale=True),
        torchvision.transforms.v2.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2008, 0.1983, 0.2002)),
    ])

    class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, ds, t):
            self.ds = ds
            self.t = t
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, i):
            x, y = self.ds[i]
            return self.t(x), y

    train_dataset = TransformDataset(raw_dataset, transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(100, 160), gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    progress_dict = {}
    for _ in (pbar := tqdm.trange(pretrain_epochs, desc="Pre-training", unit="epoch")):
        num_samples = epoch_loss = epoch_accuracy = 0
        for batch_xs, batch_ys in tqdm.tqdm(train_loader, leave=False, disable=not verbose):
            batch_xs = batch_xs.to(base.DEVICE)
            batch_ys = batch_ys.to(base.DEVICE)
            optimizer.zero_grad()
            batch_pred = model(batch_xs)
            batch_loss = loss_fn(batch_pred, batch_ys)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item() * batch_xs.size(0)
            epoch_accuracy += (batch_pred.argmax(-1) == batch_ys).int().sum().item()
            num_samples += batch_xs.size(0)
        epoch_loss /= num_samples
        epoch_accuracy /= num_samples
        progress_dict = {"pretrain_epoch_loss": epoch_loss, "pretrain_epoch_accuracy": epoch_accuracy}
        pbar.set_postfix(progress_dict)
        lr_scheduler.step()

    mlflow.log_metrics(progress_dict, step=0)
    model.eval()
    pretrained_test_metrics, _ = _evaluate_model_test(model, data_generator)
    pretrained_test_metrics = {f"pretrain_{k}": v for k, v in pretrained_test_metrics.items()}
    print(f"Pre-train test accuracy: {pretrained_test_metrics['pretrain_test_accuracy']:.4f}")
    mlflow.log_metrics(pretrained_test_metrics, step=0)
    model.replace_dense(num_classes=10)
    return model, pretrained_test_metrics


def _build_model(num_classes):
    return models.WideResNet(
        in_channels=3, depth=16, widen_factor=4, num_classes=num_classes,
        use_group_norm=True, custom_init=True, swap_order=True,
        device=base.DEVICE, dtype=base.DTYPE,
    )


def _predict(model, dataset, data_augmentation, disable_tqdm=False):
    """Run inference; dataset is a plain torch Dataset."""
    model.eval()

    normalize = torchvision.transforms.v2.Normalize(mean=data.CIFAR10_MEAN, std=data.CIFAR10_STD)
    to_float = torchvision.transforms.v2.ToDtype(base.DTYPE, scale=True)

    class PrepDataset(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, i):
            x = self.ds[i]
            x = to_float(x)
            if not data_augmentation:
                x = normalize(x)
            return x

    loader = torch.utils.data.DataLoader(
        PrepDataset(dataset), batch_size=base.EVAL_BATCH_SIZE, shuffle=False, num_workers=2,
    )
    pred_logits = []
    with torch.no_grad():
        for batch_xs in tqdm.tqdm(loader, desc="Predicting", unit="batch", disable=disable_tqdm):
            if not data_augmentation:
                pred_logits.append(model(batch_xs.to(dtype=base.DTYPE, device=base.DEVICE)).cpu().unsqueeze(1))
            else:
                flip_augmentations = (False, True)
                shift_augmentations = (0, -4, 4)
                batch_xs_pad = torchvision.transforms.v2.functional.pad(batch_xs, padding=[4])
                pred_logits_current = []
                for flip in flip_augmentations:
                    for shift_y in shift_augmentations:
                        for shift_x in shift_augmentations:
                            offset_y, offset_x = shift_y + 4, shift_x + 4
                            aug = batch_xs_pad[:, :, offset_y: offset_y + 32, offset_x: offset_x + 32]
                            if flip:
                                aug = torchvision.transforms.v2.functional.horizontal_flip(aug)
                            aug = normalize(aug)
                            pred_logits_current.append(
                                model(aug.to(dtype=base.DTYPE, device=base.DEVICE)).cpu()
                            )
                pred_logits.append(torch.stack(pred_logits_current, dim=1))
    return torch.cat(pred_logits, dim=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class DirectoryManager:
    def __init__(self, experiment_base_dir, experiment_name, run_suffix=None):
        self._experiment_base_dir = experiment_base_dir
        self._experiment_dir = experiment_base_dir / experiment_name
        self._run_suffix = run_suffix

    def get_training_output_dir(self, shadow_model_idx):
        suffix = "" if self._run_suffix is None else f"_{self._run_suffix}"
        return self._experiment_dir / ("shadow" + suffix) / str(shadow_model_idx)

    def get_training_log_dir(self, shadow_model_idx):
        return self._experiment_base_dir / "mlruns"

    def get_auto_canaries_dir(self):
        return self._experiment_dir.parent / "auto_canaries"

    def get_attack_output_dir(self):
        return self._experiment_dir if self._run_suffix is None else self._experiment_dir / f"attack_{self._run_suffix}"


def parse_args():
    default_data_dir = pathlib.Path(os.environ.get("DATA_ROOT", "data"))
    default_base_experiment_dir = pathlib.Path(os.environ.get("EXPERIMENT_DIR", "experiments"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=pathlib.Path, default=default_data_dir)
    parser.add_argument("--experiment-dir", default=default_base_experiment_dir, type=pathlib.Path)
    parser.add_argument("--experiment", type=str, default="dev")
    parser.add_argument("--run-suffix", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--num-shadow", type=int, default=64)
    parser.add_argument("--num-canaries", type=int, default=500)
    parser.add_argument("--canary-type", type=str, default="clean")
    parser.add_argument("--num-poison", type=int, default=0)
    parser.add_argument("--poison-type", type=str, default="canary_duplicates")
    parser.add_argument("--global-variance", action="store_true")

    subparsers = parser.add_subparsers(dest="action", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--shadow-model-idx", type=int, required=True)
    train_parser.add_argument("--num-epochs", type=int, default=200)
    train_parser.add_argument("--noise-multiplier", type=float, default=0.2)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument("--learning-rate", type=float, default=4.0)
    train_parser.add_argument("--batch-size", type=int, default=2048)
    train_parser.add_argument("--augmult-factor", type=int, default=8)
    train_parser.add_argument("--pretraining-epochs", type=int, required=False)
    subparsers.add_parser("audit").add_argument("--global-variance", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    experiment_base_dir = args.experiment_dir.expanduser().resolve()
    global_seed = args.seed
    base.setup_seeds(global_seed)

    num_shadow = args.num_shadow
    num_canaries = args.num_canaries
    num_poison = args.num_poison

    data_generator = data.DatasetGenerator(
        num_shadow=num_shadow, num_canaries=num_canaries,
        canary_type=data.CanaryType(args.canary_type),
        num_poison=num_poison, poison_type=data.PoisonType(args.poison_type),
        data_dir=data_dir, seed=global_seed,
        download=bool(os.environ.get("DOWNLOAD_DATA")), method="dpsgd",
    )
    directory_manager = DirectoryManager(
        experiment_base_dir=experiment_base_dir,
        experiment_name=args.experiment,
        run_suffix=getattr(args, "run_suffix", None),
    )

    if args.action == "audit":
        _audit(args, data_generator, directory_manager)
    elif args.action == "train":
        shadow_model_idx = args.shadow_model_idx
        setting_seed = base.get_setting_seed(global_seed, shadow_model_idx, num_shadow)
        base.setup_seeds(setting_seed)
        _run_train(
            args, shadow_model_idx, data_generator, directory_manager,
            setting_seed, args.experiment, getattr(args, "run_suffix", None), args.verbose,
        )
    else:
        raise ValueError(f"Unknown action {args.action}")


if __name__ == "__main__":
    main()
