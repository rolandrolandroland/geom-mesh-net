"""Offline caches for neural-field experiments with paper spatial features."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from geom_mesh_net.core_functions import paper_spatial_features as psf
from geom_mesh_net.core_functions import point_cloud_fields as pcf


CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FeatureCacheConfig:
    resolution: float = 0.5
    observation_source: str = "thinned"
    observation_probability: float = 0.1
    target_point_source: str = "original"
    guest_marks: tuple[int, ...] = (2, 3)
    target_bandwidth: float = 1.5
    voxel_samples: int | None = 512
    random_seed: int = 42

    def __post_init__(self):
        if self.resolution <= 0:
            raise ValueError("resolution must be greater than zero")
        if self.observation_source not in {"original", "thinned"}:
            raise ValueError(
                "observation_source must be 'original' or 'thinned'"
            )
        if self.target_point_source not in {"original", "thinned"}:
            raise ValueError(
                "target_point_source must be 'original' or 'thinned'"
            )
        if not 0 < self.observation_probability <= 1:
            raise ValueError(
                "observation_probability must be in the interval (0, 1]"
            )
        if self.target_bandwidth < 0:
            raise ValueError(
                "target_bandwidth must be greater than or equal to zero"
            )
        if self.voxel_samples is not None and self.voxel_samples <= 0:
            raise ValueError("voxel_samples must be positive or None")


def cache_path_for(cache_dir, null_model, pattern_index):
    return Path(cache_dir) / null_model / f"pattern_{pattern_index}.npz"


def build_pattern_feature_cache(
    data_path,
    cache_path,
    pattern_index,
    cache_config,
    global_feature_config,
    local_feature_config,
    local_window_config,
    force=False,
):
    data_path = Path(data_path)
    cache_path = Path(cache_path)
    if global_feature_config.null_model != local_feature_config.null_model:
        raise ValueError("global and local null models must match")

    signature_payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "pattern_index": pattern_index,
        "data_path": str(data_path.resolve()),
        "data_size": data_path.stat().st_size,
        "data_mtime_ns": data_path.stat().st_mtime_ns,
        "cache_config": asdict(cache_config),
        "global_feature_config": asdict(global_feature_config),
        "local_feature_config": asdict(local_feature_config),
        "local_window_config": asdict(local_window_config),
    }
    signature = _configuration_signature(signature_payload)
    if cache_path.exists() and not force:
        cached_metadata = read_cache_metadata(cache_path)
        if cached_metadata.get("signature") == signature:
            return cached_metadata, False

    with np.load(data_path, allow_pickle=True) as pattern_data:
        original_coords = pattern_data["coords"].item()
        domain = pattern_data["domain"].item()
        original_labels = np.asarray(pattern_data["labels"])

    thinned_coords, thinned_labels = _thin_point_cloud(
        original_coords,
        original_labels,
        probability=cache_config.observation_probability,
        random_seed=cache_config.random_seed + pattern_index,
    )
    observation_coords, observation_labels = pcf.choose_point_cloud(
        cache_config.observation_source,
        original_coords,
        original_labels,
        thinned_coords,
        thinned_labels,
    )
    target_coords, target_labels = pcf.choose_point_cloud(
        cache_config.target_point_source,
        original_coords,
        original_labels,
        thinned_coords,
        thinned_labels,
    )

    xx, yy, zz, target_grid = pcf.estimate_guest_probability_grid(
        coords=target_coords,
        labels=target_labels,
        domain=domain,
        resolution=cache_config.resolution,
        guest_marks=cache_config.guest_marks,
        bandwidth=cache_config.target_bandwidth,
    )
    all_xyz = np.column_stack(
        [xx.ravel(), yy.ravel(), zz.ravel()]
    ).astype(np.float32)
    all_targets = target_grid.ravel().astype(np.float32)
    voxel_indices = _sample_voxel_indices(
        len(all_xyz),
        cache_config.voxel_samples,
        cache_config.random_seed + pattern_index,
    )
    xyz = all_xyz[voxel_indices]
    targets = all_targets[voxel_indices]

    global_result = psf.calculate_global_paper_features(
        observation_coords,
        observation_labels,
        domain,
        guest_marks=cache_config.guest_marks,
        config=global_feature_config,
    )
    local_result = psf.calculate_local_paper_features(
        observation_coords,
        observation_labels,
        domain,
        xyz,
        guest_marks=cache_config.guest_marks,
        config=local_feature_config,
        local_config=local_window_config,
    )

    domain_bounds = np.asarray(
        [domain[axis] for axis in ("x", "y", "z")],
        dtype=np.float32,
    )
    metadata = {
        **signature_payload,
        "signature": signature,
        "null_model": global_feature_config.null_model,
        "point_count": int(len(original_labels)),
        "observation_point_count": int(len(observation_labels)),
        "guest_count": int(
            np.sum(
                np.isin(
                    observation_labels,
                    np.asarray(cache_config.guest_marks),
                )
            )
        ),
        "voxel_count": int(len(all_xyz)),
        "sampled_voxel_count": int(len(xyz)),
        "valid_local_feature_count": int(np.sum(local_result.valid)),
        "feature_names": list(psf.PAPER_FEATURE_NAMES),
    }
    arrays = {
        "voxel_indices": voxel_indices.astype(np.int64),
        "xyz": xyz,
        "targets": targets,
        "domain_bounds": domain_bounds,
        "global_features": global_result.values.astype(np.float32),
        "local_features": local_result.values.astype(np.float32),
        "local_valid": local_result.valid,
        "local_point_counts": local_result.point_counts,
        "local_guest_counts": local_result.guest_counts,
        "g_radii": global_result.radii["g"].astype(np.float32),
        "k_radii": global_result.radii["k"].astype(np.float32),
        "cross_g_radii": global_result.radii["cross_g"].astype(
            np.float32
        ),
        "observed_guest_g": global_result.observed.guest_g.astype(
            np.float32
        ),
        "observed_guest_f": global_result.observed.guest_f.astype(
            np.float32
        ),
        "observed_guest_k": global_result.observed.guest_k.astype(
            np.float32
        ),
        "observed_cross_g": (
            global_result.observed.guest_to_host_g.astype(np.float32)
        ),
        "expected_guest_g": global_result.expected.guest_g.astype(
            np.float32
        ),
        "expected_guest_f": global_result.expected.guest_f.astype(
            np.float32
        ),
        "expected_guest_k": global_result.expected.guest_k.astype(
            np.float32
        ),
        "expected_cross_g": (
            global_result.expected.guest_to_host_g.astype(np.float32)
        ),
    }
    _write_cache(cache_path, metadata, arrays)
    return metadata, True


def load_pattern_feature_cache(cache_path):
    cache_path = Path(cache_path)
    with np.load(cache_path, allow_pickle=False) as cached:
        result = {
            name: cached[name].copy()
            for name in cached.files
            if name != "metadata_json"
        }
        result["metadata"] = json.loads(
            str(cached["metadata_json"].item())
        )
    return result


def read_cache_metadata(cache_path):
    with np.load(cache_path, allow_pickle=False) as cached:
        return json.loads(str(cached["metadata_json"].item()))


def normalize_coordinates(xyz, domain_bounds):
    xyz = np.asarray(xyz, dtype=np.float32)
    bounds = np.asarray(domain_bounds, dtype=np.float32)
    extents = bounds[:, 1] - bounds[:, 0]
    if np.any(extents <= 0):
        raise ValueError("domain bounds must have positive extents")
    return 2.0 * (xyz - bounds[:, 0]) / extents - 1.0


def _thin_point_cloud(coords, labels, probability, random_seed):
    rng = np.random.default_rng(random_seed)
    selected = rng.random(len(labels)) <= probability
    return (
        {
            axis: np.asarray(values)[selected]
            for axis, values in coords.items()
        },
        np.asarray(labels)[selected],
    )


def _sample_voxel_indices(voxel_count, sample_count, random_seed):
    if sample_count is None or sample_count >= voxel_count:
        return np.arange(voxel_count, dtype=np.int64)
    rng = np.random.default_rng(random_seed)
    return np.sort(
        rng.choice(voxel_count, size=sample_count, replace=False)
    )


def _configuration_signature(payload):
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def _write_cache(cache_path, metadata, arrays):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as cache_file:
        np.savez_compressed(
            cache_file,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            **arrays,
        )
    temporary_path.replace(cache_path)
    metadata_path = cache_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
