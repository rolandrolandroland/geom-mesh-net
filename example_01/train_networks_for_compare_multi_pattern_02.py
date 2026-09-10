"""Cache paper features and run staged neural-field experiments."""

import argparse
import csv
import json
import os
from pathlib import Path
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from geom_mesh_net.core_functions import data_loader as dl
from geom_mesh_net.core_functions import paper_feature_experiments as pfe
from geom_mesh_net.core_functions import paper_spatial_features as psf


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "data"
DEFAULT_CACHE_DIR = SCRIPT_DIR / "paper_feature_cache_02"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "paper_feature_experiments_02"

STAGE_SPECS = {
    "coordinate_per_pattern": ("coordinate", "per_pattern"),
    "global_per_pattern": ("global", "per_pattern"),
    "coordinate_shared": ("coordinate", "shared"),
    "global_shared": ("global", "shared"),
    "local_per_pattern": ("local", "per_pattern"),
    "global_local_per_pattern": ("global_local", "per_pattern"),
}
NULL_DEPENDENT_FEATURE_KINDS = {"global", "local", "global_local"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Precompute global/local paper features, then run matched "
            "neural-field experiments."
        )
    )
    parser.add_argument(
        "command",
        choices=("cache", "train", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--patterns", default="0-4")
    parser.add_argument(
        "--null-models",
        default="csr,random_label",
        help="Comma-separated subset of csr,random_label.",
    )
    parser.add_argument(
        "--stages",
        default=",".join(STAGE_SPECS),
        help=f"Comma-separated subset of {','.join(STAGE_SPECS)}.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--observation-source",
        choices=("original", "thinned"),
        default="thinned",
    )
    parser.add_argument("--observation-probability", type=float, default=0.1)
    parser.add_argument(
        "--target-point-source",
        choices=("original", "thinned"),
        default="original",
    )
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--target-bandwidth", type=float, default=1.5)
    parser.add_argument("--voxel-samples", default="512")
    parser.add_argument("--guest-marks", default="2,3")
    parser.add_argument("--relabelings", type=int, default=19)
    parser.add_argument("--global-g-r-max", type=float, default=10.0)
    parser.add_argument("--global-g-radii", type=int, default=2001)
    parser.add_argument("--global-k-r-max", type=float, default=70.0)
    parser.add_argument("--global-k-radii", type=int, default=1401)
    parser.add_argument("--global-cross-g-r-max", type=float, default=8.0)
    parser.add_argument("--global-cross-g-radii", type=int, default=2665)
    parser.add_argument("--global-f-grid", type=int, default=24)
    parser.add_argument("--global-k-max-points", default="3000")
    parser.add_argument("--local-window-radius", type=float, default=12.0)
    parser.add_argument("--local-g-r-max", type=float, default=8.0)
    parser.add_argument("--local-g-radii", type=int, default=401)
    parser.add_argument("--local-k-r-max", type=float, default=8.0)
    parser.add_argument("--local-k-radii", type=int, default=161)
    parser.add_argument("--local-cross-g-r-max", type=float, default=3.0)
    parser.add_argument("--local-cross-g-radii", type=int, default=301)
    parser.add_argument("--local-f-grid", type=int, default=8)
    parser.add_argument("--local-k-max-points", default="512")
    parser.add_argument(
        "--local-csr-intensity-scope",
        choices=("global", "local"),
        default="global",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()
    args.pattern_indices = parse_integer_ranges(args.patterns)
    args.null_model_names = parse_choices(
        args.null_models,
        set(psf.NULL_MODELS),
        "null model",
    )
    args.stage_names = parse_choices(
        args.stages,
        set(STAGE_SPECS),
        "stage",
    )
    args.guest_mark_values = tuple(
        int(value) for value in args.guest_marks.split(",")
    )
    args.voxel_sample_count = parse_optional_int(args.voxel_samples)
    args.global_k_point_count = parse_optional_int(
        args.global_k_max_points
    )
    args.local_k_point_count = parse_optional_int(args.local_k_max_points)
    validate_training_args(args)
    return args


def parse_integer_ranges(value):
    result = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lower, upper = (int(item) for item in part.split("-", maxsplit=1))
            if upper < lower:
                raise ValueError("pattern ranges must increase")
            result.extend(range(lower, upper + 1))
        else:
            result.append(int(part))
    if not result:
        raise ValueError("at least one pattern must be selected")
    return tuple(dict.fromkeys(result))


def parse_choices(value, valid, label):
    choices = tuple(
        item.strip() for item in value.split(",") if item.strip()
    )
    invalid = set(choices) - valid
    if not choices or invalid:
        raise ValueError(
            f"invalid {label} selection: {sorted(invalid or {'<empty>'})}"
        )
    return choices


def parse_optional_int(value):
    if str(value).lower() in {"all", "none"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("optional integer values must be positive")
    return parsed


def validate_training_args(args):
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")


def build_cache_config(args):
    return pfe.FeatureCacheConfig(
        resolution=args.resolution,
        observation_source=args.observation_source,
        observation_probability=args.observation_probability,
        target_point_source=args.target_point_source,
        guest_marks=args.guest_mark_values,
        target_bandwidth=args.target_bandwidth,
        voxel_samples=args.voxel_sample_count,
        random_seed=args.random_seed,
    )


def build_global_feature_config(args, null_model):
    return psf.PaperFeatureConfig(
        g_r_max=args.global_g_r_max,
        g_num_radii=args.global_g_radii,
        k_r_max=args.global_k_r_max,
        k_num_radii=args.global_k_radii,
        cross_g_r_max=args.global_cross_g_r_max,
        cross_g_num_radii=args.global_cross_g_radii,
        f_grid_points_per_axis=args.global_f_grid,
        n_relabelings=args.relabelings,
        random_seed=args.random_seed,
        k_max_points=args.global_k_point_count,
        k_smoothing_reference_r_max=10.0,
        null_model=null_model,
    )


def build_local_feature_config(args, null_model):
    return psf.PaperFeatureConfig(
        g_r_max=args.local_g_r_max,
        g_num_radii=args.local_g_radii,
        k_r_max=args.local_k_r_max,
        k_num_radii=args.local_k_radii,
        cross_g_r_max=args.local_cross_g_r_max,
        cross_g_num_radii=args.local_cross_g_radii,
        f_grid_points_per_axis=args.local_f_grid,
        n_relabelings=args.relabelings,
        random_seed=args.random_seed,
        k_max_points=args.local_k_point_count,
        k_smoothing_reference_r_max=10.0,
        null_model=null_model,
    )


def prepare_all_caches(args):
    cache_config = build_cache_config(args)
    local_window_config = psf.LocalPaperFeatureConfig(
        neighborhood_radius=args.local_window_radius,
        csr_intensity_scope=args.local_csr_intensity_scope,
    )
    for null_model in args.null_model_names:
        global_config = build_global_feature_config(args, null_model)
        local_config = build_local_feature_config(args, null_model)
        for pattern_index in args.pattern_indices:
            data_path = args.data_dir / f"clust_pattern_{pattern_index}.npz"
            cache_path = pfe.cache_path_for(
                args.cache_dir,
                null_model,
                pattern_index,
            )
            started = time.perf_counter()
            metadata, created = pfe.build_pattern_feature_cache(
                data_path=data_path,
                cache_path=cache_path,
                pattern_index=pattern_index,
                cache_config=cache_config,
                global_feature_config=global_config,
                local_feature_config=local_config,
                local_window_config=local_window_config,
                force=args.force_cache,
            )
            action = "created" if created else "reused"
            elapsed = time.perf_counter() - started
            print(
                f"Cache {action}: {null_model} pattern {pattern_index} | "
                f"valid local voxels "
                f"{metadata['valid_local_feature_count']}/"
                f"{metadata['sampled_voxel_count']} | {elapsed:.2f}s"
            )


def load_caches(args, null_model):
    caches = {}
    for pattern_index in args.pattern_indices:
        cache_path = pfe.cache_path_for(
            args.cache_dir,
            null_model,
            pattern_index,
        )
        if not cache_path.exists():
            raise FileNotFoundError(
                f"missing cache {cache_path}; run the cache command first"
            )
        caches[pattern_index] = pfe.load_pattern_feature_cache(cache_path)
    return caches


def raw_feature_matrix(cache, feature_kind):
    row_count = len(cache["xyz"])
    if feature_kind == "coordinate":
        return np.empty((row_count, 0), dtype=np.float32)
    global_features = np.repeat(
        cache["global_features"][None, :],
        row_count,
        axis=0,
    )
    if feature_kind == "global":
        return global_features
    if feature_kind == "local":
        return cache["local_features"]
    if feature_kind == "global_local":
        return np.concatenate(
            [global_features, cache["local_features"]],
            axis=1,
        )
    raise ValueError(f"unknown feature kind: {feature_kind}")


def valid_rows(cache, feature_kind):
    if feature_kind in {"local", "global_local"}:
        return cache["local_valid"].astype(bool)
    return np.ones(len(cache["xyz"]), dtype=bool)


def fit_feature_scaler(caches, feature_kind):
    matrices = []
    for cache in caches.values():
        selected = valid_rows(cache, feature_kind)
        matrices.append(raw_feature_matrix(cache, feature_kind)[selected])
    combined = np.concatenate(matrices, axis=0)
    if combined.shape[1] == 0:
        return {
            "mean": np.empty(0, dtype=np.float32),
            "scale": np.empty(0, dtype=np.float32),
        }
    mean = np.mean(combined, axis=0).astype(np.float32)
    scale = np.std(combined, axis=0).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    return {"mean": mean, "scale": scale}


def prepare_pattern_arrays(cache, feature_kind, scaler):
    selected = valid_rows(cache, feature_kind)
    xyz = pfe.normalize_coordinates(
        cache["xyz"][selected],
        cache["domain_bounds"],
    )
    raw_features = raw_feature_matrix(cache, feature_kind)[selected]
    if raw_features.shape[1]:
        features = (
            (raw_features - scaler["mean"]) / scaler["scale"]
        ).astype(np.float32)
        inputs = np.concatenate([xyz, features], axis=1)
    else:
        inputs = xyz
    targets = cache["targets"][selected, None].astype(np.float32)
    return inputs.astype(np.float32), targets


def split_arrays(inputs, targets, validation_fraction, random_seed):
    if len(inputs) < 2:
        raise ValueError("at least two valid voxels are required")
    rng = np.random.default_rng(random_seed)
    order = rng.permutation(len(inputs))
    validation_count = max(1, int(round(len(inputs) * validation_fraction)))
    validation_count = min(validation_count, len(inputs) - 1)
    validation_indices = order[:validation_count]
    training_indices = order[validation_count:]
    return (
        inputs[training_indices],
        targets[training_indices],
        inputs[validation_indices],
        targets[validation_indices],
    )


def prepare_shared_arrays(caches, feature_kind, scaler, args):
    training_inputs = []
    training_targets = []
    validation_inputs = []
    validation_targets = []
    for pattern_index, cache in caches.items():
        inputs, targets = prepare_pattern_arrays(
            cache,
            feature_kind,
            scaler,
        )
        split = split_arrays(
            inputs,
            targets,
            args.validation_fraction,
            args.random_seed + pattern_index,
        )
        training_inputs.append(split[0])
        training_targets.append(split[1])
        validation_inputs.append(split[2])
        validation_targets.append(split[3])
    return tuple(
        np.concatenate(items, axis=0)
        for items in (
            training_inputs,
            training_targets,
            validation_inputs,
            validation_targets,
        )
    )


def select_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def train_network(
    training_inputs,
    training_targets,
    validation_inputs,
    validation_targets,
    save_dir,
    args,
    device,
    run_metadata,
):
    torch.manual_seed(args.random_seed)
    model = dl.ContinuousNeuralFieldFeatures(
        feature_count=training_inputs.shape[1] - 3,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )
    loss_fn = nn.BCELoss()
    generator = torch.Generator().manual_seed(args.random_seed)
    training_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(training_inputs),
            torch.from_numpy(training_targets),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(validation_inputs),
            torch.from_numpy(validation_targets),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        **run_metadata,
        "input_dimension": int(training_inputs.shape[1]),
        "training_voxels": int(len(training_inputs)),
        "validation_voxels": int(len(validation_inputs)),
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_width": args.hidden_width,
        "hidden_layers": args.hidden_layers,
    }
    (save_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    metrics_path = save_dir / "training_metrics.csv"
    best_validation_loss = float("inf")
    best_epoch = 0
    final_metrics = None
    cumulative_time = 0.0

    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(
            [
                "epoch",
                "cumulative_training_time_seconds",
                "training_loss",
                "validation_loss",
                "validation_mae",
                "validation_psnr",
                "validation_iou",
            ]
        )
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            training_loss = train_one_epoch(
                model,
                training_loader,
                optimizer,
                loss_fn,
                device,
            )
            cumulative_time += time.perf_counter() - epoch_started
            validation_metrics = evaluate_model(
                model,
                validation_loader,
                loss_fn,
                device,
            )
            final_metrics = validation_metrics
            writer.writerow(
                [
                    epoch,
                    cumulative_time,
                    training_loss,
                    validation_metrics["loss"],
                    validation_metrics["mae"],
                    validation_metrics["psnr"],
                    validation_metrics["iou"],
                ]
            )

            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                best_epoch = epoch
                save_checkpoint(
                    save_dir / "best.pth",
                    model,
                    metadata,
                    epoch,
                )
            if epoch % args.checkpoint_interval == 0:
                save_checkpoint(
                    save_dir / f"epoch_{epoch}.pth",
                    model,
                    metadata,
                    epoch,
                )
                print(
                    f"Epoch {epoch} | val MAE "
                    f"{validation_metrics['mae']:.4f} | "
                    f"val IoU {validation_metrics['iou']:.4f}"
                )

    save_checkpoint(
        save_dir / "final.pth",
        model,
        metadata,
        args.epochs,
    )
    return {
        **run_metadata,
        "training_voxels": len(training_inputs),
        "validation_voxels": len(validation_inputs),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "final_validation_loss": final_metrics["loss"],
        "final_validation_mae": final_metrics["mae"],
        "final_validation_psnr": final_metrics["psnr"],
        "final_validation_iou": final_metrics["iou"],
        "training_time_seconds": cumulative_time,
    }


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    loss_total = 0.0
    sample_count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        batch_count = len(inputs)
        loss_total += loss.item() * batch_count
        sample_count += batch_count
    return loss_total / sample_count


def evaluate_model(model, loader, loss_fn, device):
    model.eval()
    loss_total = 0.0
    absolute_error_total = 0.0
    squared_error_total = 0.0
    intersection_total = 0.0
    union_total = 0.0
    sample_count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            batch_count = len(inputs)
            loss_total += loss_fn(predictions, targets).item() * batch_count
            absolute_error_total += torch.sum(
                torch.abs(predictions - targets)
            ).item()
            squared_error_total += torch.sum(
                (predictions - targets) ** 2
            ).item()
            predicted_positive = predictions >= 0.5
            target_positive = targets >= 0.5
            intersection_total += torch.sum(
                predicted_positive & target_positive
            ).item()
            union_total += torch.sum(
                predicted_positive | target_positive
            ).item()
            sample_count += batch_count
    mse = squared_error_total / sample_count
    return {
        "loss": loss_total / sample_count,
        "mae": absolute_error_total / sample_count,
        "psnr": float(-10.0 * np.log10(mse + 1e-8)),
        "iou": intersection_total / max(union_total, 1.0),
    }


def save_checkpoint(path, model, metadata, epoch):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def scaler_metadata(scaler, feature_kind):
    names = []
    if feature_kind in {"global", "global_local"}:
        names.extend(f"global_{name}" for name in psf.PAPER_FEATURE_NAMES)
    if feature_kind in {"local", "global_local"}:
        names.extend(f"local_{name}" for name in psf.PAPER_FEATURE_NAMES)
    return {
        "feature_kind": feature_kind,
        "feature_names": names,
        "mean": scaler["mean"].tolist(),
        "scale": scaler["scale"].tolist(),
    }


def train_per_pattern_stage(
    stage_name,
    feature_kind,
    null_model,
    caches,
    scaler,
    args,
    device,
):
    summaries = []
    for pattern_index, cache in caches.items():
        inputs, targets = prepare_pattern_arrays(
            cache,
            feature_kind,
            scaler,
        )
        split = split_arrays(
            inputs,
            targets,
            args.validation_fraction,
            args.random_seed + pattern_index,
        )
        save_dir = stage_output_dir(
            args.output_dir,
            null_model,
            stage_name,
        ) / f"pattern_{pattern_index}"
        print(
            f"Training {stage_name} | {null_model or 'none'} | "
            f"pattern {pattern_index}"
        )
        summaries.append(
            train_network(
                *split,
                save_dir=save_dir,
                args=args,
                device=device,
                run_metadata={
                    "stage": stage_name,
                    "feature_kind": feature_kind,
                    "training_scope": "per_pattern",
                    "null_model": null_model,
                    "pattern": pattern_index,
                },
            )
        )
    return summaries


def train_shared_stage(
    stage_name,
    feature_kind,
    null_model,
    caches,
    scaler,
    args,
    device,
):
    split = prepare_shared_arrays(caches, feature_kind, scaler, args)
    save_dir = stage_output_dir(
        args.output_dir,
        null_model,
        stage_name,
    )
    print(f"Training {stage_name} | {null_model or 'none'} | shared")
    return [
        train_network(
            *split,
            save_dir=save_dir,
            args=args,
            device=device,
            run_metadata={
                "stage": stage_name,
                "feature_kind": feature_kind,
                "training_scope": "shared",
                "null_model": null_model,
                "pattern": "all",
                "patterns": list(args.pattern_indices),
            },
        )
    ]


def stage_output_dir(output_dir, null_model, stage_name):
    null_directory = null_model or "no_null_model"
    return Path(output_dir) / null_directory / stage_name


def save_scaler(output_dir, null_model, feature_kind, scaler):
    path = Path(output_dir) / (null_model or "no_null_model") / "scalers"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{feature_kind}.json").write_text(
        json.dumps(
            scaler_metadata(scaler, feature_kind),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def run_training_experiments(args):
    device = select_device(args.device)
    print(f"Training device: {device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_config(args, device)
    summaries = []

    coordinate_stages = [
        stage
        for stage in args.stage_names
        if STAGE_SPECS[stage][0] == "coordinate"
    ]
    if coordinate_stages:
        coordinate_caches = load_caches(args, args.null_model_names[0])
        coordinate_scaler = fit_feature_scaler(
            coordinate_caches,
            "coordinate",
        )
        for stage_name in coordinate_stages:
            _, scope = STAGE_SPECS[stage_name]
            if scope == "per_pattern":
                summaries.extend(
                    train_per_pattern_stage(
                        stage_name,
                        "coordinate",
                        None,
                        coordinate_caches,
                        coordinate_scaler,
                        args,
                        device,
                    )
                )
            else:
                summaries.extend(
                    train_shared_stage(
                        stage_name,
                        "coordinate",
                        None,
                        coordinate_caches,
                        coordinate_scaler,
                        args,
                        device,
                    )
                )

    for null_model in args.null_model_names:
        caches = load_caches(args, null_model)
        for stage_name in args.stage_names:
            feature_kind, scope = STAGE_SPECS[stage_name]
            if feature_kind not in NULL_DEPENDENT_FEATURE_KINDS:
                continue
            scaler = fit_feature_scaler(caches, feature_kind)
            save_scaler(
                args.output_dir,
                null_model,
                feature_kind,
                scaler,
            )
            if scope == "per_pattern":
                summaries.extend(
                    train_per_pattern_stage(
                        stage_name,
                        feature_kind,
                        null_model,
                        caches,
                        scaler,
                        args,
                        device,
                    )
                )
            else:
                summaries.extend(
                    train_shared_stage(
                        stage_name,
                        feature_kind,
                        null_model,
                        caches,
                        scaler,
                        args,
                        device,
                    )
                )
    write_summary(args.output_dir / "experiment_summary.csv", summaries)


def write_experiment_config(args, device):
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"pattern_indices", "null_model_names", "stage_names"}
    }
    serializable.update(
        {
            "pattern_indices": list(args.pattern_indices),
            "null_model_names": list(args.null_model_names),
            "stage_names": list(args.stage_names),
            "device_selected": str(device),
        }
    )
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n"
    )


def write_summary(path, summaries):
    if not summaries:
        return
    fieldnames = [
        "stage",
        "feature_kind",
        "training_scope",
        "null_model",
        "pattern",
        "training_voxels",
        "validation_voxels",
        "best_epoch",
        "best_validation_loss",
        "final_validation_loss",
        "final_validation_mae",
        "final_validation_psnr",
        "final_validation_iou",
        "training_time_seconds",
    ]
    with path.open("w", newline="") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(summaries)


def main():
    args = parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if args.command in {"cache", "all"}:
        prepare_all_caches(args)
    if args.command in {"train", "all"}:
        run_training_experiments(args)


if __name__ == "__main__":
    main()
