"""Execute and report Experimental Methodology 01."""

import argparse
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/physical_ml_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/physical_ml_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label as connected_components
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn

from geom_mesh_net.core_functions import data_loader as dl
from geom_mesh_net.core_functions import paper_spatial_features as psf
from geom_mesh_net.core_functions import point_cloud_fields as pcf


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "methodology_01_results"
MODEL_NAMES = ("coordinate", "local")
FEATURE_NAMES = psf.PAPER_FEATURE_NAMES
RADIUS_FEATURE_INDICES = (1, 3, 7, 8, 9, 12, 13)


@dataclass(frozen=True)
class ExperimentConfig:
    patterns: tuple[int, ...] = (0, 1, 4)
    observation_fraction: float = 0.10
    resolution: float = 0.5
    target_bandwidth: float = 1.5
    guest_marks: tuple[int, ...] = (2, 3)
    random_seed: int = 42
    relabelings: int = 89
    coarse_spacing: float = 4.0
    interpolation_validation_points: int = 256
    local_window_radius: float = 12.0
    local_g_r_max: float = 8.0
    local_g_radii: int = 401
    local_k_r_max: float = 8.0
    local_k_radii: int = 161
    local_cross_g_r_max: float = 3.0
    local_cross_g_radii: int = 301
    local_f_grid: int = 8
    local_k_max_points: int = 512
    spatial_blocks_per_axis: int = 5
    validation_block_fraction: float = 0.20
    cluster_quantile: float = 0.90
    batch_size: int = 2048
    batches_per_epoch: int = 32
    maximum_epochs: int = 150
    validation_interval: int = 5
    early_stopping_checks: int = 4
    early_stopping_min_delta: float = 0.0001
    learning_rate: float = 0.001
    hidden_width: int = 128
    hidden_layers: int = 3
    device: str = "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "train", "report", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patterns", default="0,1,4")
    parser.add_argument("--relabelings", type=int, default=89)
    parser.add_argument("--coarse-spacing", type=float, default=4.0)
    parser.add_argument("--interpolation-points", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batches-per-epoch", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-failed-interpolation", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    patterns = tuple(int(value) for value in args.patterns.split(","))
    if args.smoke:
        patterns = patterns[:1]
        args.relabelings = min(args.relabelings, 2)
        args.coarse_spacing = 20.0
        args.interpolation_points = min(args.interpolation_points, 8)
        args.epochs = min(args.epochs, 2)
        args.batches_per_epoch = min(args.batches_per_epoch, 2)
        args.batch_size = min(args.batch_size, 128)
        args.allow_failed_interpolation = True
    config = ExperimentConfig(
        patterns=patterns,
        relabelings=args.relabelings,
        coarse_spacing=args.coarse_spacing,
        interpolation_validation_points=args.interpolation_points,
        maximum_epochs=args.epochs,
        validation_interval=1 if args.smoke else 5,
        early_stopping_checks=2 if args.smoke else 4,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size,
        device=args.device,
    )
    return args, config


def pattern_dir(output_dir, pattern_index):
    return Path(output_dir) / "patterns" / f"pattern_{pattern_index}"


def prepare_experiment(data_dir, output_dir, config, force=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "experiment_config.json", asdict(config))
    summaries = []
    for pattern_index in config.patterns:
        summaries.append(
            prepare_pattern(
                data_dir / f"clust_pattern_{pattern_index}.npz",
                pattern_dir(output_dir, pattern_index),
                pattern_index,
                config,
                force=force,
            )
        )
    write_csv(output_dir / "preprocessing_summary.csv", summaries)
    return summaries


def prepare_pattern(data_path, output_path, pattern_index, config, force=False):
    output_path.mkdir(parents=True, exist_ok=True)
    signature = configuration_signature(data_path, pattern_index, config)
    metadata_path = output_path / "preprocessing_metadata.json"
    if metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("signature") == signature:
            print(f"Reused pattern {pattern_index} preprocessing cache")
            return preprocessing_summary(metadata)

    started = time.perf_counter()
    with np.load(data_path, allow_pickle=True) as source:
        original_coords = source["coords"].item()
        original_labels = np.asarray(source["labels"])
        domain = source["domain"].item()

    observation_indices, target_indices = split_point_indices(
        len(original_labels),
        config.observation_fraction,
        config.random_seed + pattern_index,
    )
    observation_coords = subset_coords(original_coords, observation_indices)
    observation_labels = original_labels[observation_indices]
    target_coords = subset_coords(original_coords, target_indices)
    target_labels = original_labels[target_indices]

    target_started = time.perf_counter()
    xx, yy, zz, target_grid = pcf.estimate_guest_probability_grid(
        target_coords,
        target_labels,
        domain,
        config.resolution,
        guest_marks=config.guest_marks,
        bandwidth=config.target_bandwidth,
    )
    target_seconds = time.perf_counter() - target_started
    grid_shape = target_grid.shape
    grid_axes = (
        xx[:, 0, 0].astype(np.float32),
        yy[0, :, 0].astype(np.float32),
        zz[0, 0, :].astype(np.float32),
    )

    coarse_axes = coarse_feature_axes(domain, config.coarse_spacing)
    coarse_queries = mesh_points(coarse_axes)
    exact_indices = sample_exact_validation_indices(
        grid_shape,
        config.interpolation_validation_points,
        config.random_seed + 10000 + pattern_index,
    )
    exact_queries = coordinates_for_indices(exact_indices, grid_axes, grid_shape)
    all_queries = np.concatenate([coarse_queries, exact_queries], axis=0)

    feature_started = time.perf_counter()
    feature_result = psf.calculate_local_paper_features(
        observation_coords,
        observation_labels,
        domain,
        all_queries,
        guest_marks=config.guest_marks,
        config=local_paper_config(config, pattern_index),
        local_config=psf.LocalPaperFeatureConfig(
            neighborhood_radius=config.local_window_radius,
            csr_intensity_scope="global",
        ),
    )
    feature_seconds = time.perf_counter() - feature_started
    if not np.all(feature_result.valid):
        invalid_count = int(np.sum(~feature_result.valid))
        raise RuntimeError(
            f"pattern {pattern_index} has {invalid_count} invalid local features"
        )

    coarse_count = len(coarse_queries)
    coarse_features = feature_result.values[:coarse_count]
    exact_features = feature_result.values[coarse_count:]
    coarse_feature_grid = coarse_features.reshape(
        tuple(len(axis) for axis in coarse_axes) + (len(FEATURE_NAMES),)
    )
    interpolator = RegularGridInterpolator(
        coarse_axes,
        coarse_feature_grid,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    interpolated_exact = interpolate_points(
        interpolator,
        exact_queries,
        coarse_axes,
    )
    interpolation_rows, interpolation_pass = interpolation_statistics(
        pattern_index,
        coarse_features,
        exact_features,
        interpolated_exact,
    )
    write_csv(output_path / "interpolation_metrics.csv", interpolation_rows)

    interpolation_started = time.perf_counter()
    local_feature_path = output_path / "local_features.npy"
    interpolate_full_feature_grid(
        interpolator,
        grid_axes,
        coarse_axes,
        grid_shape,
        local_feature_path,
    )
    interpolation_seconds = time.perf_counter() - interpolation_started

    train_mask, validation_mask, block_ids = spatial_block_split(
        target_grid,
        config,
        pattern_index,
    )
    save_array(output_path / "target.npy", target_grid.astype(np.float32))
    save_array(output_path / "train_mask.npy", train_mask)
    save_array(output_path / "validation_mask.npy", validation_mask)
    save_array(output_path / "block_ids.npy", block_ids.astype(np.int16))
    save_array(output_path / "observation_indices.npy", observation_indices)
    save_array(output_path / "target_indices.npy", target_indices)
    save_array(output_path / "coarse_queries.npy", coarse_queries.astype(np.float32))
    save_array(output_path / "coarse_features.npy", coarse_features.astype(np.float32))
    save_array(output_path / "exact_queries.npy", exact_queries.astype(np.float32))
    save_array(output_path / "exact_features.npy", exact_features.astype(np.float32))
    save_array(
        output_path / "interpolated_exact_features.npy",
        interpolated_exact.astype(np.float32),
    )
    save_array(output_path / "local_point_counts.npy", feature_result.point_counts)
    save_array(output_path / "local_guest_counts.npy", feature_result.guest_counts)

    elapsed = time.perf_counter() - started
    metadata = {
        "signature": signature,
        "pattern": pattern_index,
        "domain": {
            axis: [float(value) for value in domain[axis]]
            for axis in ("x", "y", "z")
        },
        "grid_shape": list(grid_shape),
        "grid_axes": [axis.tolist() for axis in grid_axes],
        "coarse_axes": [axis.tolist() for axis in coarse_axes],
        "point_count": int(len(original_labels)),
        "observation_count": int(len(observation_indices)),
        "target_point_count": int(len(target_indices)),
        "observation_guest_count": int(
            np.sum(np.isin(observation_labels, config.guest_marks))
        ),
        "coarse_feature_locations": int(coarse_count),
        "interpolation_validation_locations": int(len(exact_queries)),
        "interpolation_pass": bool(interpolation_pass),
        "training_voxels": int(np.sum(train_mask)),
        "validation_voxels": int(np.sum(validation_mask)),
        "target_seconds": target_seconds,
        "feature_seconds": feature_seconds,
        "interpolation_seconds": interpolation_seconds,
        "total_preprocessing_seconds": elapsed,
        "config": asdict(config),
    }
    write_json(metadata_path, metadata)
    print(
        f"Prepared pattern {pattern_index} in {elapsed / 60:.2f} min | "
        f"interpolation {'PASS' if interpolation_pass else 'FAIL'}"
    )
    return preprocessing_summary(metadata)


def local_paper_config(config, pattern_index):
    return psf.PaperFeatureConfig(
        g_r_max=config.local_g_r_max,
        g_num_radii=config.local_g_radii,
        k_r_max=config.local_k_r_max,
        k_num_radii=config.local_k_radii,
        cross_g_r_max=config.local_cross_g_r_max,
        cross_g_num_radii=config.local_cross_g_radii,
        f_grid_points_per_axis=config.local_f_grid,
        n_relabelings=config.relabelings,
        random_seed=config.random_seed + pattern_index,
        workers=-1,
        k_max_points=config.local_k_max_points,
        k_smoothing_reference_r_max=10.0,
        null_model="random_label",
    )


def split_point_indices(point_count, observation_fraction, random_seed):
    rng = np.random.default_rng(random_seed)
    observation_count = int(round(point_count * observation_fraction))
    observation_indices = np.sort(
        rng.choice(point_count, size=observation_count, replace=False)
    )
    selected = np.zeros(point_count, dtype=bool)
    selected[observation_indices] = True
    target_indices = np.flatnonzero(~selected)
    return observation_indices.astype(np.int64), target_indices.astype(np.int64)


def subset_coords(coords, indices):
    return {
        axis: np.asarray(values)[indices]
        for axis, values in coords.items()
    }


def coarse_feature_axes(domain, spacing):
    axes = []
    for axis_name in ("x", "y", "z"):
        lower, upper = (float(value) for value in domain[axis_name])
        first = lower + spacing / 2.0
        axes.append(np.arange(first, upper, spacing, dtype=np.float32))
    return tuple(axes)


def mesh_points(axes):
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([values.ravel() for values in mesh]).astype(
        np.float32
    )


def sample_exact_validation_indices(
    grid_shape,
    sample_count,
    random_seed,
):
    voxel_count = int(np.prod(grid_shape))
    rng = np.random.default_rng(random_seed)
    return np.sort(
        rng.choice(voxel_count, size=sample_count, replace=False)
    ).astype(np.int64)


def coordinates_for_indices(indices, grid_axes, grid_shape):
    unravelled = np.unravel_index(indices, grid_shape)
    return np.column_stack(
        [grid_axes[axis][unravelled[axis]] for axis in range(3)]
    ).astype(np.float32)


def interpolate_points(interpolator, points, coarse_axes):
    clipped = np.asarray(points, dtype=np.float32).copy()
    for axis_index, axis in enumerate(coarse_axes):
        np.clip(
            clipped[:, axis_index],
            float(axis[0]),
            float(axis[-1]),
            out=clipped[:, axis_index],
        )
    return interpolator(clipped).astype(np.float32)


def interpolation_statistics(
    pattern_index,
    coarse_features,
    exact_features,
    interpolated_features,
):
    rows = []
    normalized_errors = []
    correlations = []
    radius_errors = []
    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        exact = exact_features[:, feature_index].astype(float)
        interpolated = interpolated_features[:, feature_index].astype(float)
        errors = interpolated - exact
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        robust_range = float(
            np.percentile(coarse_features[:, feature_index], 95)
            - np.percentile(coarse_features[:, feature_index], 5)
        )
        normalized_rmse = rmse / max(abs(robust_range), 1e-8)
        pearson = safe_correlation(pearsonr, exact, interpolated)
        spearman = safe_correlation(spearmanr, exact, interpolated)
        rows.append(
            {
                "pattern": pattern_index,
                "feature": feature_name,
                "mae": mae,
                "rmse": rmse,
                "robust_range": robust_range,
                "normalized_rmse": normalized_rmse,
                "pearson": pearson,
                "spearman": spearman,
            }
        )
        normalized_errors.append(normalized_rmse)
        correlations.append(spearman)
        if feature_index in RADIUS_FEATURE_INDICES:
            radius_errors.append(normalized_rmse)
    passed = (
        float(np.median(normalized_errors)) <= 0.15
        and int(np.sum(np.asarray(correlations) >= 0.80)) >= 12
        and max(radius_errors, default=0.0) <= 0.30
    )
    return rows, passed


def safe_correlation(function, first, second):
    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return 1.0 if np.allclose(first, second) else 0.0
    return float(function(first, second).statistic)


def interpolate_full_feature_grid(
    interpolator,
    grid_axes,
    coarse_axes,
    grid_shape,
    output_path,
    chunk_size=65536,
):
    temporary_path = output_path.with_suffix(".npy.tmp")
    feature_map = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float32,
        shape=(int(np.prod(grid_shape)), len(FEATURE_NAMES)),
    )
    for start in range(0, len(feature_map), chunk_size):
        stop = min(start + chunk_size, len(feature_map))
        indices = np.arange(start, stop, dtype=np.int64)
        points = coordinates_for_indices(indices, grid_axes, grid_shape)
        feature_map[start:stop] = interpolate_points(
            interpolator,
            points,
            coarse_axes,
        )
    feature_map.flush()
    del feature_map
    temporary_path.replace(output_path)


def spatial_block_split(target_grid, config, pattern_index):
    shape = target_grid.shape
    block_shape = tuple(
        size // config.spatial_blocks_per_axis for size in shape
    )
    axis_blocks = [
        np.minimum(
            np.arange(size) // block_size,
            config.spatial_blocks_per_axis - 1,
        )
        for size, block_size in zip(shape, block_shape)
    ]
    block_mesh = np.meshgrid(*axis_blocks, indexing="ij")
    block_ids = (
        block_mesh[0] * config.spatial_blocks_per_axis**2
        + block_mesh[1] * config.spatial_blocks_per_axis
        + block_mesh[2]
    )
    target_threshold = float(np.quantile(target_grid, config.cluster_quantile))
    total_blocks = config.spatial_blocks_per_axis**3
    enriched = []
    background = []
    for block_id in range(total_blocks):
        block_mask = block_ids == block_id
        destination = (
            enriched
            if np.any(target_grid[block_mask] >= target_threshold)
            else background
        )
        destination.append(block_id)
    rng = np.random.default_rng(config.random_seed + pattern_index)
    validation_blocks = []
    for group in (enriched, background):
        if not group:
            continue
        count = max(1, int(round(len(group) * config.validation_block_fraction)))
        validation_blocks.extend(
            rng.choice(group, size=min(count, len(group)), replace=False).tolist()
        )
    validation_mask = np.isin(block_ids, validation_blocks)
    training_mask = ~validation_mask
    return training_mask, validation_mask, block_ids


def train_experiment(output_dir, config, allow_failed_interpolation=False):
    device = select_device(config.device)
    torch.set_num_threads(8)
    summaries = []
    for pattern_index in config.patterns:
        current_dir = pattern_dir(output_dir, pattern_index)
        preprocessing = json.loads(
            (current_dir / "preprocessing_metadata.json").read_text()
        )
        if not preprocessing["interpolation_pass"] and not allow_failed_interpolation:
            raise RuntimeError(
                f"pattern {pattern_index} failed interpolation validation"
            )
        for model_name in MODEL_NAMES:
            summaries.append(
                train_pattern_model(
                    current_dir,
                    pattern_index,
                    model_name,
                    config,
                    device,
                )
            )
    write_csv(output_dir / "model_summary.csv", summaries)
    return summaries


def select_device(name):
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    return torch.device("cpu")


def train_pattern_model(
    current_dir,
    pattern_index,
    model_name,
    config,
    device,
):
    model_dir = current_dir / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    target = np.load(current_dir / "target.npy", mmap_mode="r").reshape(-1)
    training_mask = np.load(
        current_dir / "train_mask.npy", mmap_mode="r"
    ).reshape(-1)
    validation_mask = np.load(
        current_dir / "validation_mask.npy", mmap_mode="r"
    ).reshape(-1)
    preprocessing = json.loads(
        (current_dir / "preprocessing_metadata.json").read_text()
    )
    grid_shape = tuple(preprocessing["grid_shape"])
    grid_axes = tuple(
        np.asarray(axis, dtype=np.float32)
        for axis in preprocessing["grid_axes"]
    )
    features = None
    scaler = None
    feature_count = 0
    if model_name == "local":
        features = np.load(current_dir / "local_features.npy", mmap_mode="r")
        scaler = fit_training_scaler(features, training_mask)
        write_json(
            model_dir / "feature_scaler.json",
            {"mean": scaler[0].tolist(), "scale": scaler[1].tolist()},
        )
        feature_count = len(FEATURE_NAMES)

    training_indices = np.flatnonzero(training_mask)
    validation_indices = np.flatnonzero(validation_mask)
    cluster_threshold = float(
        np.quantile(target[training_indices], config.cluster_quantile)
    )
    cluster_indices = training_indices[
        target[training_indices] >= cluster_threshold
    ]
    background_indices = training_indices[
        target[training_indices] < cluster_threshold
    ]

    torch.manual_seed(config.random_seed)
    model = dl.ContinuousNeuralFieldFeatures(
        feature_count=feature_count,
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCELoss()
    sampler = BalancedIndexSampler(
        cluster_indices,
        background_indices,
        config.batch_size,
        config.random_seed,
    )
    history = []
    best_loss = float("inf")
    best_epoch = 0
    stale_checks = 0
    training_seconds = 0.0
    started = time.perf_counter()

    for epoch in range(1, config.maximum_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        losses = []
        for _ in range(config.batches_per_epoch):
            batch_indices = sampler.next_batch()
            inputs = model_inputs(
                batch_indices,
                grid_axes,
                grid_shape,
                features,
                scaler,
            )
            targets = target[batch_indices, None].astype(np.float32)
            inputs_tensor = torch.from_numpy(inputs).to(device)
            targets_tensor = torch.from_numpy(targets).to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs_tensor)
            loss = loss_fn(predictions, targets_tensor)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        training_seconds += time.perf_counter() - epoch_started

        if epoch % config.validation_interval != 0:
            continue
        validation_predictions = predict_indices(
            model,
            validation_indices,
            grid_axes,
            grid_shape,
            features,
            scaler,
            device,
        )
        validation_loss = binary_cross_entropy(
            target[validation_indices],
            validation_predictions,
        )
        history.append(
            {
                "epoch": epoch,
                "training_bce": float(np.mean(losses)),
                "validation_bce": validation_loss,
                "training_seconds": training_seconds,
            }
        )
        if validation_loss < best_loss - config.early_stopping_min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            stale_checks = 0
            save_checkpoint(
                model_dir / "best.pth",
                model,
                model_name,
                pattern_index,
                epoch,
                config,
            )
        else:
            stale_checks += 1
        print(
            f"Pattern {pattern_index} {model_name} epoch {epoch} | "
            f"train BCE {np.mean(losses):.5f} | val BCE {validation_loss:.5f}"
        )
        if stale_checks >= config.early_stopping_checks:
            break

    write_csv(model_dir / "training_history.csv", history)
    checkpoint = torch.load(
        model_dir / "best.pth",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    inference_started = time.perf_counter()
    all_indices = np.arange(len(target), dtype=np.int64)
    predictions = predict_indices(
        model,
        all_indices,
        grid_axes,
        grid_shape,
        features,
        scaler,
        device,
    )
    inference_seconds = time.perf_counter() - inference_started
    save_array(
        model_dir / "predictions.npy",
        predictions.reshape(grid_shape).astype(np.float32),
    )
    metrics = prediction_metrics(
        target.reshape(grid_shape),
        predictions.reshape(grid_shape),
        validation_mask.reshape(grid_shape),
        config.resolution,
    )
    total_seconds = time.perf_counter() - started
    summary = {
        "pattern": pattern_index,
        "model": model_name,
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "total_model_seconds": total_seconds,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        **metrics,
    }
    write_json(model_dir / "metrics.json", summary)
    return summary


class BalancedIndexSampler:
    def __init__(self, cluster_indices, background_indices, batch_size, seed):
        if batch_size % 2:
            raise ValueError("batch_size must be even")
        self.rng = np.random.default_rng(seed)
        self.cluster = np.asarray(cluster_indices)
        self.background = np.asarray(background_indices)
        self.half_batch = batch_size // 2
        self.cluster_position = 0
        self.background_position = 0
        self.rng.shuffle(self.cluster)
        self.rng.shuffle(self.background)

    def next_batch(self):
        cluster = self._take("cluster")
        background = self._take("background")
        batch = np.concatenate([cluster, background])
        self.rng.shuffle(batch)
        return batch

    def _take(self, name):
        values = getattr(self, name)
        position_name = f"{name}_position"
        position = getattr(self, position_name)
        pieces = []
        remaining = self.half_batch
        while remaining:
            available = len(values) - position
            take = min(remaining, available)
            pieces.append(values[position : position + take])
            position += take
            remaining -= take
            if position == len(values):
                self.rng.shuffle(values)
                position = 0
        setattr(self, position_name, position)
        return np.concatenate(pieces)


def fit_training_scaler(features, training_mask, chunk_size=131072):
    indices = np.flatnonzero(training_mask)
    total = np.zeros(features.shape[1], dtype=np.float64)
    squared = np.zeros(features.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(indices), chunk_size):
        values = np.asarray(features[indices[start : start + chunk_size]], dtype=float)
        total += np.sum(values, axis=0)
        squared += np.sum(values**2, axis=0)
        count += len(values)
    mean = total / count
    variance = np.maximum(squared / count - mean**2, 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def model_inputs(indices, grid_axes, grid_shape, features, scaler):
    coordinates = coordinates_for_indices(indices, grid_axes, grid_shape)
    lower = np.asarray([axis[0] for axis in grid_axes], dtype=np.float32)
    upper = np.asarray([axis[-1] for axis in grid_axes], dtype=np.float32)
    normalized = 2.0 * (coordinates - lower) / (upper - lower) - 1.0
    if features is None:
        return normalized.astype(np.float32)
    selected = np.asarray(features[indices], dtype=np.float32)
    standardized = (selected - scaler[0]) / scaler[1]
    return np.concatenate([normalized, standardized], axis=1).astype(np.float32)


def predict_indices(
    model,
    indices,
    grid_axes,
    grid_shape,
    features,
    scaler,
    device,
    chunk_size=32768,
):
    predictions = np.empty(len(indices), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), chunk_size):
            stop = min(start + chunk_size, len(indices))
            inputs = model_inputs(
                indices[start:stop],
                grid_axes,
                grid_shape,
                features,
                scaler,
            )
            output = model(torch.from_numpy(inputs).to(device))
            predictions[start:stop] = output.detach().cpu().numpy().ravel()
    return predictions


def binary_cross_entropy(target, prediction):
    clipped = np.clip(prediction.astype(float), 1e-7, 1.0 - 1e-7)
    target = target.astype(float)
    return float(
        -np.mean(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))
    )


def prediction_metrics(target_grid, prediction_grid, validation_mask, resolution):
    target = target_grid[validation_mask].astype(float)
    prediction = prediction_grid[validation_mask].astype(float)
    target_threshold = float(np.quantile(target, 0.90))
    prediction_threshold = float(np.quantile(prediction, 0.90))
    target_positive = target >= target_threshold
    prediction_positive = prediction >= prediction_threshold
    intersection = int(np.sum(target_positive & prediction_positive))
    dice = 2.0 * intersection / max(
        int(np.sum(target_positive)) + int(np.sum(prediction_positive)),
        1,
    )
    brier = float(np.mean((prediction - target) ** 2))
    morphology = morphology_metrics(
        target_grid >= target_threshold,
        prediction_grid >= prediction_threshold,
        validation_mask,
        resolution,
    )
    return {
        "bce": binary_cross_entropy(target, prediction),
        "brier": brier,
        "mae": float(np.mean(np.abs(prediction - target))),
        "rmse": float(np.sqrt(brier)),
        "pearson": safe_correlation(pearsonr, target, prediction),
        "spearman": safe_correlation(spearmanr, target, prediction),
        "ece_10": expected_calibration_error(target, prediction, 10),
        "pr_auc_top10": precision_recall_auc(target_positive, prediction),
        "dice_top10": float(dice),
        "target_top10_threshold": target_threshold,
        "prediction_top10_threshold": prediction_threshold,
        **morphology,
    }


def expected_calibration_error(target, prediction, bins):
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(prediction, edges) - 1, bins - 1)
    error = 0.0
    for bin_index in range(bins):
        selected = assignments == bin_index
        if np.any(selected):
            error += (
                np.mean(selected)
                * abs(float(np.mean(prediction[selected]) - np.mean(target[selected])))
            )
    return float(error)


def precision_recall_auc(target_positive, scores):
    order = np.argsort(scores)[::-1]
    truth = target_positive[order].astype(float)
    positives = max(float(np.sum(truth)), 1.0)
    true_positives = np.cumsum(truth)
    false_positives = np.cumsum(1.0 - truth)
    recall = true_positives / positives
    precision = true_positives / np.maximum(true_positives + false_positives, 1.0)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum(np.diff(recall) * precision[1:]))


def morphology_metrics(target_mask, prediction_mask, validation_mask, resolution):
    structure = np.ones((3, 3, 3), dtype=int)
    target_labels, target_count = connected_components(
        target_mask & validation_mask,
        structure=structure,
    )
    prediction_labels, prediction_count = connected_components(
        prediction_mask & validation_mask,
        structure=structure,
    )
    target_centroids, target_volumes = component_properties(
        target_labels,
        target_count,
        resolution,
    )
    prediction_centroids, prediction_volumes = component_properties(
        prediction_labels,
        prediction_count,
        resolution,
    )
    if len(target_centroids) and len(prediction_centroids):
        rows, columns = linear_sum_assignment(
            cdist(target_centroids, prediction_centroids)
        )
        centroid_distances = np.linalg.norm(
            target_centroids[rows] - prediction_centroids[columns],
            axis=1,
        )
        volume_errors = np.abs(
            prediction_volumes[columns] - target_volumes[rows]
        ) / np.maximum(target_volumes[rows], 1e-8)
        median_centroid = float(np.median(centroid_distances))
        median_volume_error = float(np.median(volume_errors))
    else:
        median_centroid = float("nan")
        median_volume_error = float("nan")

    recovered = 0
    for target_index in range(1, target_count + 1):
        selected = target_labels == target_index
        if np.mean(prediction_mask[selected]) >= 0.10:
            recovered += 1
    return {
        "target_component_count": int(target_count),
        "prediction_component_count": int(prediction_count),
        "median_centroid_error": median_centroid,
        "median_component_volume_relative_error": median_volume_error,
        "target_component_recovery_fraction": (
            recovered / max(target_count, 1)
        ),
    }


def component_properties(labels, count, resolution):
    centroids = []
    volumes = []
    voxel_volume = resolution**3
    for component_index in range(1, count + 1):
        coordinates = np.argwhere(labels == component_index)
        centroids.append(np.mean(coordinates, axis=0) * resolution)
        volumes.append(len(coordinates) * voxel_volume)
    return np.asarray(centroids, dtype=float), np.asarray(volumes, dtype=float)


def save_checkpoint(path, model, model_name, pattern_index, epoch, config):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model": model_name,
            "pattern": pattern_index,
            "epoch": epoch,
            "config": asdict(config),
        },
        path,
    )


def generate_report(output_dir, config):
    report_dir = output_dir / "report"
    figure_dir = report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    preprocessing = read_csv(output_dir / "preprocessing_summary.csv")
    interpolation = []
    models = []
    histories = {}
    for pattern_index in config.patterns:
        current_dir = pattern_dir(output_dir, pattern_index)
        interpolation.extend(read_csv(current_dir / "interpolation_metrics.csv"))
        for model_name in MODEL_NAMES:
            model_dir = current_dir / "models" / model_name
            metrics_path = model_dir / "metrics.json"
            if metrics_path.exists():
                models.append(json.loads(metrics_path.read_text()))
                histories[(pattern_index, model_name)] = read_csv(
                    model_dir / "training_history.csv"
                )
    if models:
        write_csv(report_dir / "model_metrics.csv", models)
    write_csv(report_dir / "interpolation_metrics.csv", interpolation)
    plot_interpolation(interpolation, figure_dir / "interpolation_quality.png", config)
    plot_runtime(preprocessing, models, figure_dir / "runtime_benchmarks.png", config)
    if models:
        plot_model_metrics(models, figure_dir / "model_metric_comparison.png", config)
        plot_training(histories, figure_dir / "training_curves.png", config)
        plot_prediction_slices(output_dir, figure_dir / "prediction_slices.png", config)
        plot_residual_slices(output_dir, figure_dir / "residual_slices.png", config)
        plot_calibration(output_dir, figure_dir / "calibration_curves.png", config)
        plot_feature_slices(output_dir, figure_dir / "local_feature_slices.png", config)
    report_text = build_report_text(preprocessing, interpolation, models, config)
    (report_dir / "REPORT.md").write_text(report_text)
    print(f"Report written to {report_dir / 'REPORT.md'}")


def plot_interpolation(rows, path, config):
    error_matrix = np.full((len(FEATURE_NAMES), len(config.patterns)), np.nan)
    correlation_matrix = np.full_like(error_matrix, np.nan)
    for row in rows:
        feature_index = FEATURE_NAMES.index(row["feature"])
        pattern_index = config.patterns.index(int(row["pattern"]))
        error_matrix[feature_index, pattern_index] = float(row["normalized_rmse"])
        correlation_matrix[feature_index, pattern_index] = float(row["spearman"])
    figure, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    for axis, matrix, title, label, cmap, limits in (
        (
            axes[0],
            error_matrix,
            "Robust normalized RMSE",
            "Normalized RMSE",
            "viridis",
            None,
        ),
        (
            axes[1],
            correlation_matrix,
            "Spearman correlation",
            "Spearman rho",
            "coolwarm",
            (-1.0, 1.0),
        ),
    ):
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            vmin=None if limits is None else limits[0],
            vmax=None if limits is None else limits[1],
        )
        axis.set_xticks(
            range(len(config.patterns)),
            [f"Pattern {p}" for p in config.patterns],
        )
        axis.set_yticks(range(len(FEATURE_NAMES)), FEATURE_NAMES)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label=label)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_runtime(preprocessing, models, path, config):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    patterns = [int(row["pattern"]) for row in preprocessing]
    feature_minutes = [float(row["feature_seconds"]) / 60 for row in preprocessing]
    other_minutes = [
        (
            float(row["target_seconds"])
            + float(row["interpolation_seconds"])
        )
        / 60
        for row in preprocessing
    ]
    axes[0].bar(patterns, feature_minutes, label="Local features")
    axes[0].bar(patterns, other_minutes, bottom=feature_minutes, label="Target + interpolation")
    axes[0].set_title("Preprocessing time")
    axes[0].set_xlabel("Pattern")
    axes[0].set_ylabel("Minutes")
    axes[0].legend()
    if models:
        labels = [f"P{row['pattern']} {row['model'][0].upper()}" for row in models]
        seconds = [float(row["training_seconds"]) for row in models]
        axes[1].bar(labels, seconds)
        axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("Model training time")
    axes[1].set_ylabel("Seconds")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_model_metrics(models, path, config):
    metrics = (
        ("brier", "Brier score", False),
        ("dice_top10", "Top-10% Dice", True),
        ("pr_auc_top10", "Top-10% PR-AUC", True),
        ("ece_10", "Calibration error", False),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    width = 0.36
    positions = np.arange(len(config.patterns))
    for axis, (metric, title, _) in zip(axes.ravel(), metrics):
        for model_offset, model_name in enumerate(MODEL_NAMES):
            values = [
                float(next(row[metric] for row in models if int(row["pattern"]) == pattern and row["model"] == model_name))
                for pattern in config.patterns
            ]
            axis.bar(
                positions + (model_offset - 0.5) * width,
                values,
                width,
                label=model_name,
            )
        axis.set_xticks(positions, [f"P{pattern}" for pattern in config.patterns])
        axis.set_title(title)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_training(histories, path, config):
    figure, axes = plt.subplots(1, len(config.patterns), figsize=(15, 4), sharey=True)
    if len(config.patterns) == 1:
        axes = [axes]
    for axis, pattern_index in zip(axes, config.patterns):
        for model_name in MODEL_NAMES:
            rows = histories[(pattern_index, model_name)]
            axis.plot(
                [int(row["epoch"]) for row in rows],
                [float(row["validation_bce"]) for row in rows],
                marker="o",
                label=model_name,
            )
        axis.set_title(f"Pattern {pattern_index}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation BCE")
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_prediction_slices(output_dir, path, config):
    figure, axes = plt.subplots(len(config.patterns), 3, figsize=(11, 3.5 * len(config.patterns)))
    if len(config.patterns) == 1:
        axes = axes[None, :]
    for row_index, pattern_index in enumerate(config.patterns):
        current_dir = pattern_dir(output_dir, pattern_index)
        target = np.load(current_dir / "target.npy", mmap_mode="r")
        coordinate = np.load(current_dir / "models" / "coordinate" / "predictions.npy", mmap_mode="r")
        local = np.load(current_dir / "models" / "local" / "predictions.npy", mmap_mode="r")
        middle = target.shape[2] // 2
        vmax = float(np.max(target[:, :, middle]))
        for axis, values, title in zip(
            axes[row_index],
            (target, coordinate, local),
            ("Target", "Coordinate", "Local features"),
        ):
            axis.imshow(values[:, :, middle].T, origin="lower", vmin=0, vmax=vmax, cmap="magma")
            axis.set_title(f"Pattern {pattern_index}: {title}")
            axis.set_xticks([])
            axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_residual_slices(output_dir, path, config):
    figure, axes = plt.subplots(
        len(config.patterns),
        2,
        figsize=(8, 3.5 * len(config.patterns)),
    )
    if len(config.patterns) == 1:
        axes = axes[None, :]
    for row_index, pattern_index in enumerate(config.patterns):
        current_dir = pattern_dir(output_dir, pattern_index)
        target = np.load(current_dir / "target.npy", mmap_mode="r")
        predictions = {
            model_name: np.load(
                current_dir / "models" / model_name / "predictions.npy",
                mmap_mode="r",
            )
            for model_name in MODEL_NAMES
        }
        middle = target.shape[2] // 2
        residual_limit = max(
            float(
                np.max(
                    np.abs(
                        predictions[model_name][:, :, middle]
                        - target[:, :, middle]
                    )
                )
            )
            for model_name in MODEL_NAMES
        )
        for column_index, model_name in enumerate(MODEL_NAMES):
            axis = axes[row_index, column_index]
            image = axis.imshow(
                (
                    predictions[model_name][:, :, middle]
                    - target[:, :, middle]
                ).T,
                origin="lower",
                vmin=-residual_limit,
                vmax=residual_limit,
                cmap="coolwarm",
            )
            axis.set_title(f"Pattern {pattern_index}: {model_name} residual")
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_calibration(output_dir, path, config):
    figure, axes = plt.subplots(1, len(config.patterns), figsize=(15, 4))
    if len(config.patterns) == 1:
        axes = [axes]
    edges = np.linspace(0.0, 1.0, 11)
    for axis, pattern_index in zip(axes, config.patterns):
        current_dir = pattern_dir(output_dir, pattern_index)
        target = np.load(current_dir / "target.npy", mmap_mode="r").reshape(-1)
        validation = np.load(current_dir / "validation_mask.npy", mmap_mode="r").reshape(-1)
        for model_name in MODEL_NAMES:
            prediction = np.load(
                current_dir / "models" / model_name / "predictions.npy",
                mmap_mode="r",
            ).reshape(-1)
            observed = []
            predicted = []
            assignments = np.minimum(np.digitize(prediction[validation], edges) - 1, 9)
            for bin_index in range(10):
                selected = assignments == bin_index
                if np.any(selected):
                    observed.append(float(np.mean(target[validation][selected])))
                    predicted.append(float(np.mean(prediction[validation][selected])))
            axis.plot(predicted, observed, marker="o", label=model_name)
        axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        axis.set_title(f"Pattern {pattern_index}")
        axis.set_xlabel("Predicted probability")
        axis.set_ylabel("Mean target probability")
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_feature_slices(output_dir, path, config):
    selected_features = ("G_max_diff", "Tm", "GXGH_min_diff")
    figure, axes = plt.subplots(
        len(config.patterns),
        len(selected_features),
        figsize=(11, 3.5 * len(config.patterns)),
    )
    if len(config.patterns) == 1:
        axes = axes[None, :]
    for row_index, pattern_index in enumerate(config.patterns):
        current_dir = pattern_dir(output_dir, pattern_index)
        metadata = json.loads((current_dir / "preprocessing_metadata.json").read_text())
        shape = tuple(metadata["grid_shape"])
        features = np.load(current_dir / "local_features.npy", mmap_mode="r").reshape(shape + (len(FEATURE_NAMES),))
        middle = shape[2] // 2
        for column_index, feature_name in enumerate(selected_features):
            feature_index = FEATURE_NAMES.index(feature_name)
            axis = axes[row_index, column_index]
            image = axis.imshow(
                features[:, :, middle, feature_index].T,
                origin="lower",
                cmap="coolwarm",
            )
            axis.set_title(f"P{pattern_index} {feature_name}")
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_report_text(preprocessing, interpolation, models, config):
    preprocessing_by_pattern = {
        int(row["pattern"]): row for row in preprocessing
    }
    interpolation_summary = []
    for pattern_index in config.patterns:
        rows = [row for row in interpolation if int(row["pattern"]) == pattern_index]
        errors = np.asarray([float(row["normalized_rmse"]) for row in rows])
        correlations = np.asarray([float(row["spearman"]) for row in rows])
        radius_errors = np.asarray(
            [
                float(row["normalized_rmse"])
                for row in rows
                if FEATURE_NAMES.index(row["feature"]) in RADIUS_FEATURE_INDICES
            ]
        )
        passed = (
            np.median(errors) <= 0.15
            and np.sum(correlations >= 0.80) >= 12
            and np.max(radius_errors) <= 0.30
        )
        interpolation_summary.append(
            (
                pattern_index,
                float(np.median(errors)),
                int(np.sum(correlations >= 0.80)),
                float(np.max(radius_errors)),
                passed,
            )
        )

    lines = [
        "# Experimental Methodology 01 Report",
        "",
        "## Executive Summary",
        "",
    ]
    if not models:
        lines.append(
            "Model training was not completed. The report therefore covers preprocessing and interpolation validation only."
        )
    else:
        advancement, reasons = advancement_decision(models, interpolation_summary, config)
        lines.append(
            f"The local-feature approach **{'ADVANCES' if advancement else 'DOES NOT ADVANCE'}** under the prespecified screening criteria."
        )
        lines.extend(["", *[f"- {reason}" for reason in reasons]])
        if not all(row[-1] for row in interpolation_summary):
            lines.extend(
                [
                    "",
                    "Because interpolation failed, the trained-model results below are exploratory diagnostics. They do not override the prespecified interpolation stopping rule.",
                ]
            )
    lines.extend(
        [
            "",
            "## Experimental Design",
            "",
            f"Patterns: {', '.join(str(value) for value in config.patterns)}.",
            f"Whole-pattern random relabelings: {config.relabelings}.",
            f"Coarse feature spacing: {config.coarse_spacing} units.",
            f"Exact interpolation checks per pattern: {config.interpolation_validation_points}.",
            "Models: coordinate-only and coordinate-plus-14-local-features for each pattern.",
            "Loss: unweighted binary cross-entropy with balanced cluster/background training batches.",
            "Validation: complete held-out spatial blocks.",
            "",
            "## Interpolation Validation",
            "",
            "| Pattern | Median normalized RMSE | Features with Spearman >= 0.80 | Maximum radius-feature error | Result |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for pattern_index, error, count, radius_error, passed in interpolation_summary:
        lines.append(
            f"| {pattern_index} | {error:.4f} | {count}/14 | {radius_error:.4f} | {'PASS' if passed else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "### Feature-level interpolation statistics",
            "",
            "| Feature | Mean normalized RMSE | Mean Spearman | Minimum Spearman |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for feature_name in FEATURE_NAMES:
        feature_rows = [row for row in interpolation if row["feature"] == feature_name]
        feature_errors = [float(row["normalized_rmse"]) for row in feature_rows]
        feature_correlations = [float(row["spearman"]) for row in feature_rows]
        lines.append(
            f"| {feature_name} | {np.mean(feature_errors):.4f} | {np.mean(feature_correlations):.4f} | {np.min(feature_correlations):.4f} |"
        )
    lines.extend(
        [
            "",
            "![Interpolation quality](figures/interpolation_quality.png)",
            "",
            "## Computational Benchmarks",
            "",
            "| Pattern | Feature time (min) | Interpolation time (min) | Total preprocessing (min) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for pattern_index in config.patterns:
        row = preprocessing_by_pattern[pattern_index]
        lines.append(
            f"| {pattern_index} | {float(row['feature_seconds']) / 60:.2f} | {float(row['interpolation_seconds']) / 60:.2f} | {float(row['total_preprocessing_seconds']) / 60:.2f} |"
        )
    total_preprocessing_minutes = sum(
        float(row["total_preprocessing_seconds"]) for row in preprocessing
    ) / 60.0
    lines.extend(
        [
            "",
            f"Total preprocessing time was {total_preprocessing_minutes:.2f} minutes. This was much faster than the seven-hour planning estimate because the small calibration run did not scale linearly to the larger, reused whole-pattern relabeling workload.",
            "",
            "![Runtime benchmarks](figures/runtime_benchmarks.png)",
        ]
    )

    if models:
        lines.extend(
            [
                "",
                "## Model Metrics",
                "",
                "| Pattern | Model | Brier | BCE | MAE | RMSE | Dice | PR-AUC | ECE | Spearman | Components | Centroid error |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in models:
            lines.append(
                f"| {row['pattern']} | {row['model']} | {row['brier']:.6f} | {row['bce']:.6f} | {row['mae']:.6f} | {row['rmse']:.6f} | {row['dice_top10']:.4f} | {row['pr_auc_top10']:.4f} | {row['ece_10']:.6f} | {row['spearman']:.4f} | {row['prediction_component_count']} | {row['median_centroid_error']:.3f} |"
            )
        lines.extend(
            [
                "",
                "### Correlation, morphology, and compute",
                "",
                "| Pattern | Model | Pearson | Target components | Predicted components | Component recovery | Median volume error | Best epoch | Parameters | Train time (s) | Inference (s) |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in models:
            lines.append(
                f"| {row['pattern']} | {row['model']} | {row['pearson']:.4f} | {row['target_component_count']} | {row['prediction_component_count']} | {row['target_component_recovery_fraction']:.4f} | {row['median_component_volume_relative_error']:.4f} | {row['best_epoch']} | {row['parameter_count']} | {row['training_seconds']:.2f} | {row['inference_seconds']:.2f} |"
            )
        lines.extend(
            [
                "",
                f"Total optimizer time across all six models was {sum(float(row['training_seconds']) for row in models):.2f} seconds. Connected-component metrics are calculated only inside the held-out blocks, so regions crossing block boundaries can be fragmented; these morphology values are secondary diagnostics.",
            ]
        )
        lines.extend(
            [
                "",
                "![Metric comparison](figures/model_metric_comparison.png)",
                "",
                "![Training curves](figures/training_curves.png)",
                "",
                "![Prediction slices](figures/prediction_slices.png)",
                "",
                "![Residual slices](figures/residual_slices.png)",
                "",
                "![Calibration curves](figures/calibration_curves.png)",
                "",
                "![Local feature slices](figures/local_feature_slices.png)",
                "",
                "## Paired Comparisons",
                "",
                "| Metric | Coordinate mean | Local mean | Median local-minus-coordinate | Local favorable patterns |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for metric, higher_is_better in (
            ("brier", False),
            ("dice_top10", True),
            ("pr_auc_top10", True),
            ("ece_10", False),
            ("median_centroid_error", False),
        ):
            differences = paired_differences(models, metric, config)
            favorable = differences if higher_is_better else -differences
            coordinate_values = [
                float(row[metric])
                for row in models
                if row["model"] == "coordinate"
            ]
            local_values = [
                float(row[metric])
                for row in models
                if row["model"] == "local"
            ]
            lines.append(
                f"| {metric} | {np.mean(coordinate_values):.6f} | {np.mean(local_values):.6f} | {np.median(differences):.6f} | {int(np.sum(favorable > 0))}/{len(config.patterns)} |"
            )
        lines.extend(
            [
                "",
                "With only three paired patterns and one training seed, inferential p-values are not interpreted. The paired differences are screening effect sizes.",
            ]
        )

        pattern_effects = []
        calibration_wins = 0
        for pattern_index in config.patterns:
            coordinate = next(
                row
                for row in models
                if int(row["pattern"]) == pattern_index
                and row["model"] == "coordinate"
            )
            local = next(
                row
                for row in models
                if int(row["pattern"]) == pattern_index and row["model"] == "local"
            )
            relative_brier_improvement = 100.0 * (
                coordinate["brier"] - local["brier"]
            ) / coordinate["brier"]
            dice_change = local["dice_top10"] - coordinate["dice_top10"]
            pr_auc_change = local["pr_auc_top10"] - coordinate["pr_auc_top10"]
            ece_change = local["ece_10"] - coordinate["ece_10"]
            calibration_wins += ece_change < 0
            pattern_effects.append(
                f"- Pattern {pattern_index}: relative Brier improvement {relative_brier_improvement:+.2f}%, "
                f"Dice change {dice_change:+.4f}, PR-AUC change {pr_auc_change:+.4f}, "
                f"and ECE change {ece_change:+.4f}."
            )

        feature_quality = []
        for feature_name in FEATURE_NAMES:
            feature_rows = [
                row for row in interpolation if row["feature"] == feature_name
            ]
            feature_quality.append(
                (
                    feature_name,
                    float(
                        np.mean(
                            [float(row["normalized_rmse"]) for row in feature_rows]
                        )
                    ),
                    float(
                        np.mean([float(row["spearman"]) for row in feature_rows])
                    ),
                )
            )
        smoothest = sorted(feature_quality, key=lambda value: value[2], reverse=True)[:5]
        least_reliable = sorted(feature_quality, key=lambda value: value[2])[:5]
        smoothest_text = ", ".join(
            f"{name} (rho={correlation:.3f})"
            for name, _, correlation in smoothest
        )
        least_reliable_text = ", ".join(
            f"{name} (rho={correlation:.3f})"
            for name, _, correlation in least_reliable
        )
        lines.extend(
            [
                "",
                "## Interpretation and Decision",
                "",
                *pattern_effects,
                "",
                f"Calibration improved for the local model in {calibration_wins}/{len(config.patterns)} patterns, but calibration alone is not sufficient to satisfy the advancement gate.",
                "",
                f"The smoothest interpolated channels were {smoothest_text}. The least reliable were {least_reliable_text}.",
                "",
                "The primary result is that spacing-4 trilinear interpolation of all 14 extracted local features is not validated. Several extrema and radius-derived channels are too spatially irregular at this resolution. The local model's substantial improvement on the fine-cluster pattern shows that local information may still be useful, but the present representation is not reliable enough to advance.",
                "",
                "The next controlled experiment should keep the same split, loss, sampling, and evaluation protocol while changing only feature construction: use a finer or adaptive local grid, interpolate the underlying summary-function curves before extracting extrema, or prespecify a reduced subset of the smoothly interpolated channels. Repeat across multiple seeds only after the interpolation gate passes.",
                "",
                "Limitations: this is a three-pattern, one-seed screening experiment; no inferential significance claim is supported. Held-out blocks can fragment connected targets, so connected-component metrics remain secondary to voxelwise, ranking, and calibration metrics.",
            ]
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `interpolation_metrics.csv`: per-feature interpolation statistics.",
            "- `model_metrics.csv`: complete model metrics when training was run.",
            "- Pattern directories contain point splits, feature grids, masks, predictions, histories, checkpoints, and timings.",
            "",
        ]
    )
    return "\n".join(lines)


def advancement_decision(models, interpolation_summary, config):
    interpolation_pass = all(row[-1] for row in interpolation_summary)
    local_better_brier = 0
    local_better_dice = 0
    brier_relative = []
    dice_absolute = []
    worst_brier_degradation = 0.0
    for pattern_index in config.patterns:
        coordinate = next(
            row for row in models if int(row["pattern"]) == pattern_index and row["model"] == "coordinate"
        )
        local = next(
            row for row in models if int(row["pattern"]) == pattern_index and row["model"] == "local"
        )
        local_better_brier += local["brier"] < coordinate["brier"]
        local_better_dice += local["dice_top10"] > coordinate["dice_top10"]
        brier_relative.append(
            (coordinate["brier"] - local["brier"]) / coordinate["brier"]
        )
        dice_absolute.append(local["dice_top10"] - coordinate["dice_top10"])
        worst_brier_degradation = max(
            worst_brier_degradation,
            (local["brier"] - coordinate["brier"]) / coordinate["brier"],
        )
    paired_success = sum(
        1
        for pattern_index in config.patterns
        if (
            next(row for row in models if int(row["pattern"]) == pattern_index and row["model"] == "local")["brier"]
            < next(row for row in models if int(row["pattern"]) == pattern_index and row["model"] == "coordinate")["brier"]
            and next(row for row in models if int(row["pattern"]) == pattern_index and row["model"] == "local")["dice_top10"]
            > next(row for row in models if int(row["pattern"]) == pattern_index and row["model"] == "coordinate")["dice_top10"]
        )
    )
    effect_pass = np.median(brier_relative) >= 0.05 or np.median(dice_absolute) >= 0.05
    degradation_pass = worst_brier_degradation <= 0.10
    passed = interpolation_pass and paired_success >= 2 and effect_pass and degradation_pass
    reasons = [
        f"Interpolation passed for {sum(row[-1] for row in interpolation_summary)}/{len(interpolation_summary)} patterns.",
        f"Local improved both Brier and Dice in {paired_success}/{len(config.patterns)} patterns.",
        f"Median relative Brier improvement was {np.median(brier_relative) * 100:.2f}%.",
        f"Median absolute Dice improvement was {np.median(dice_absolute):.4f}.",
        f"Worst relative Brier degradation was {worst_brier_degradation * 100:.2f}%.",
    ]
    return passed, reasons


def paired_differences(models, metric, config):
    values = []
    for pattern_index in config.patterns:
        coordinate = next(
            row for row in models if int(row["pattern"]) == pattern_index and row["model"] == "coordinate"
        )
        local = next(
            row for row in models if int(row["pattern"]) == pattern_index and row["model"] == "local"
        )
        values.append(float(local[metric]) - float(coordinate[metric]))
    return np.asarray(values)


def preprocessing_summary(metadata):
    return {
        key: metadata[key]
        for key in (
            "pattern",
            "point_count",
            "observation_count",
            "target_point_count",
            "observation_guest_count",
            "coarse_feature_locations",
            "interpolation_validation_locations",
            "interpolation_pass",
            "training_voxels",
            "validation_voxels",
            "target_seconds",
            "feature_seconds",
            "interpolation_seconds",
            "total_preprocessing_seconds",
        )
    }


def configuration_signature(data_path, pattern_index, config):
    payload = {
        "data_path": str(data_path.resolve()),
        "data_size": data_path.stat().st_size,
        "data_mtime_ns": data_path.stat().st_mtime_ns,
        "pattern": pattern_index,
        "config": asdict(config),
    }
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def save_array(path, array):
    temporary_path = path.with_suffix(".npy.tmp")
    with temporary_path.open("wb") as output:
        np.save(output, array, allow_pickle=False)
    temporary_path.replace(path)


def write_json(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def main():
    args, config = parse_args()
    if args.command in {"prepare", "all"}:
        prepare_experiment(
            args.data_dir,
            args.output_dir,
            config,
            force=args.force,
        )
    if args.command in {"train", "all"}:
        train_experiment(
            args.output_dir,
            config,
            allow_failed_interpolation=args.allow_failed_interpolation,
        )
    if args.command in {"report", "all"}:
        generate_report(args.output_dir, config)


if __name__ == "__main__":
    main()
