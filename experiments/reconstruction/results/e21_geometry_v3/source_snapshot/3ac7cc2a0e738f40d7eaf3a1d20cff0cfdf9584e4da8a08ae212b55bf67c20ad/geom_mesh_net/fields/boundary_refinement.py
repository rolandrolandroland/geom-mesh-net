"""Fixed-catalogue radial boundary proposals from interchangeable field functions.

This module never detects, deletes, merges, moves, or truth-matches a particle.
One observation-derived catalogue and one set of physical sampling locations are
shared across Gaussian and neural fields. Quality thresholds are development
settings, not uncertainty estimates. The proposed radius only initializes the
subsequent physics fit; it is not a validated material-interface measurement.
"""

from dataclasses import dataclass, field
import copy
import hashlib
import json

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .particle_detection import ParticleGeometry


def _readonly(array, dtype=float):
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CandidateCatalog:
    """Stable detector identities, initial geometry, and fitting-data provenance."""

    centres: np.ndarray
    radii: np.ndarray
    candidate_ids: tuple | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        geometry = ParticleGeometry(self.centres, self.radii)
        ids = (tuple(f"candidate-{i:05d}" for i in range(len(geometry.radii)))
               if self.candidate_ids is None else tuple(self.candidate_ids))
        if (len(ids) != len(geometry.radii) or any(not isinstance(i, str) or not i for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("candidate_ids must be distinct nonempty strings, one per candidate")
        provenance = copy.deepcopy(self.provenance)
        if not isinstance(provenance, dict):
            raise ValueError("provenance must be a dictionary")
        json.dumps(provenance, allow_nan=False)
        object.__setattr__(self, "centres", _readonly(geometry.centres))
        object.__setattr__(self, "radii", _readonly(geometry.radii))
        object.__setattr__(self, "candidate_ids", ids)
        object.__setattr__(self, "provenance", provenance)

    @classmethod
    def from_geometry(cls, geometry, *, candidate_ids=None, provenance=None):
        metadata = {"detector": copy.deepcopy(geometry.metadata)}
        metadata.update(provenance or {})
        return cls(geometry.centres, geometry.radii, candidate_ids, metadata)

    def to_dict(self):
        return {"centres": self.centres.tolist(), "radii": self.radii.tolist(),
                "candidate_ids": list(self.candidate_ids), "provenance": copy.deepcopy(self.provenance)}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def geometry(self):
        return ParticleGeometry(self.centres, self.radii,
                                {"candidate_ids": list(self.candidate_ids),
                                 "catalog_fingerprint": self.fingerprint,
                                 "provenance": copy.deepcopy(self.provenance)})


@dataclass(frozen=True)
class ProfileSamples:
    """Reusable radius/direction locations and observation-derived geometric mask."""

    points: np.ndarray
    mask: np.ndarray
    radial_positions: np.ndarray
    candidate_ids: tuple
    catalog_fingerprint: str
    settings: dict


def sample_candidate_profiles(catalog, *, box=60.0, lower=0.0,
                              radial_factors=(0.6, 1.4), n_radial=41, n_directions=64):
    """Sample deterministic Fibonacci directions in a bounded radial neighbourhood.

    ``box`` is the upper box edge (scalar or vector), not its length when ``lower``
    is nonzero. Samples outside the box or closer to another catalogue centre are
    masked. A direction is retained only when its entire radial search window is
    valid: changing angular membership with radius could create a spurious radial
    gradient. Equal-distance samples belong to the first catalogue index. No atom
    labels, held-back positions, or truth information are accepted by this API.
    """
    lo, hi = np.broadcast_to(lower, (3,)).astype(float), np.broadcast_to(box, (3,)).astype(float)
    factors = np.asarray(radial_factors, dtype=float)
    if not np.all(np.isfinite([lo, hi])) or np.any(hi <= lo):
        raise ValueError("box must have finite upper edges greater than lower")
    if (factors.shape != (2,) or not np.all(np.isfinite(factors))
            or not 0 < factors[0] < factors[1]):
        raise ValueError("radial_factors must be two increasing positive values")
    for value, minimum, name in [(n_radial, 7, "n_radial"), (n_directions, 4, "n_directions")]:
        if not np.isscalar(value) or not np.isfinite(value) or int(value) != value or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    n_radial, n_directions = int(n_radial), int(n_directions)
    i = np.arange(n_directions) + 0.5
    z = 1 - 2 * i / n_directions
    azimuth = np.pi * (1 + np.sqrt(5)) * i
    directions = np.column_stack([np.sqrt(1 - z * z) * np.cos(azimuth),
                                  np.sqrt(1 - z * z) * np.sin(azimuth), z])
    radial = catalog.radii[:, None] * np.linspace(*factors, n_radial)[None, :]
    points = catalog.centres[:, None, None, :] + radial[:, :, None, None] * directions[None, None, :, :]
    mask = np.all((points >= lo) & (points <= hi), axis=-1)
    # Iterate over candidates, not all sample/candidate pairs at once.
    for k, locations in enumerate(points):
        distances = np.sum((locations[..., None, :] - catalog.centres) ** 2, axis=-1)
        mask[k] &= np.argmin(distances, axis=-1) == k
        mask[k] &= np.all(mask[k], axis=0)[None, :]
    return ProfileSamples(_readonly(points), _readonly(mask, bool), _readonly(radial),
                          catalog.candidate_ids, catalog.fingerprint,
                          {"lower": lo.tolist(), "box": hi.tolist(),
                           "radial_factors": factors.tolist(), "n_radial": n_radial,
                           "n_directions": n_directions,
                           "mask_rule": "box_and_nearest_fixed_centre_complete_rays"})


DEFAULT_SETTINGS = {"min_angular_coverage": 0.5, "min_contrast": 0.05,
                    "smooth_sigma_bins": 1.0, "competing_peak_ratio": 0.6,
                    "peak_separation_bins": 4, "bound_margin_bins": 2,
                    "contrast_fraction": 0.2}


@dataclass(frozen=True)
class RadiusProposals:
    radii: np.ndarray
    accepted: np.ndarray
    diagnostics: tuple
    candidate_ids: tuple
    catalog_fingerprint: str
    settings: dict

    def to_dict(self):
        return {"radii": self.radii.tolist(), "accepted": self.accepted.tolist(),
                "diagnostics": copy.deepcopy(list(self.diagnostics)),
                "candidate_ids": list(self.candidate_ids), "catalog_fingerprint": self.catalog_fingerprint,
                "settings": copy.deepcopy(self.settings)}


def _settings(settings):
    values = dict(DEFAULT_SETTINGS)
    if settings:
        unknown = set(settings) - set(values)
        if unknown:
            raise ValueError(f"unknown boundary settings: {sorted(unknown)}")
        values.update(settings)
    if any(not np.isscalar(v) or not np.isfinite(v) for v in values.values()):
        raise ValueError("boundary settings must be finite numbers")
    for name in ("min_angular_coverage", "competing_peak_ratio"):
        if not 0 < values[name] <= 1:
            raise ValueError(f"{name} must lie in (0, 1]")
    if not 0 < values["contrast_fraction"] <= .5 or not 0 <= values["min_contrast"] <= 1:
        raise ValueError("invalid concentration contrast settings")
    if values["smooth_sigma_bins"] < 0:
        raise ValueError("smooth_sigma_bins must be nonnegative")
    for name, minimum in (("peak_separation_bins", 1), ("bound_margin_bins", 0)):
        if values[name] != int(values[name]) or values[name] < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
        values[name] = int(values[name])
    return values


def propose_radii(catalog, samples, field_callable, *, settings=None):
    """Use the maximum outward concentration drop, subject to fixed quality checks.

    Both fields receive the exact same flattened locations, in the same order.
    A proposal is rejected for inadequate coverage/contrast, competing transitions,
    a peak against the search bound, or any invalid sampled probability. Rejected
    radii are placeholders equal to the original radius; the fallback operation
    decides which baseline to retain. Diagnostics deliberately contain no labels.
    """
    if samples.catalog_fingerprint != catalog.fingerprint or samples.candidate_ids != catalog.candidate_ids:
        raise ValueError("samples belong to a different candidate catalogue")
    options = _settings(settings)
    values = np.full(samples.mask.shape, np.nan)
    if samples.mask.any():
        predictions = np.asarray(field_callable(samples.points[samples.mask]), dtype=float)
        if predictions.shape != (int(samples.mask.sum()),):
            raise ValueError("field_callable must return one probability per supplied location")
        values[samples.mask] = predictions
    radii, accepted, diagnostics = catalog.radii.copy(), np.zeros(len(catalog.radii), dtype=bool), []
    for k, candidate_id in enumerate(catalog.candidate_ids):
        valid = samples.mask[k]
        counts = valid.sum(axis=-1)
        coverage = counts / valid.shape[-1]
        good_probability = np.isfinite(values[k]) & (values[k] >= 0) & (values[k] <= 1)
        invalid_count = int(np.sum(valid & ~good_probability))
        reasons = []
        if np.min(coverage) < options["min_angular_coverage"]:
            reasons.append("insufficient_angular_coverage")
        if invalid_count:
            reasons.append("invalid_field_probability")
        profile = np.divide(np.where(valid & good_probability, values[k], 0).sum(axis=-1), counts,
                            out=np.zeros(len(counts), dtype=float), where=counts > 0)
        contrast = peak_radius = peak_drop = None
        peak_count, bound_contact = 0, False
        if not reasons:
            smooth = (gaussian_filter1d(profile, options["smooth_sigma_bins"], mode="nearest")
                      if options["smooth_sigma_bins"] else profile)
            n_edge = max(1, int(np.ceil(len(profile) * options["contrast_fraction"])))
            contrast = float(smooth[:n_edge].mean() - smooth[-n_edge:].mean())
            drop = -np.gradient(smooth, samples.radial_positions[k])
            peak = int(np.argmax(drop))
            peak_radius, peak_drop = float(samples.radial_positions[k, peak]), float(drop[peak])
            peaks, _ = find_peaks(drop, height=max(0, peak_drop * options["competing_peak_ratio"]),
                                  distance=options["peak_separation_bins"])
            peak_count = len(peaks)
            bound_contact = peak <= options["bound_margin_bins"] or peak >= len(drop) - 1 - options["bound_margin_bins"]
            if contrast < options["min_contrast"] or peak_drop <= 0:
                reasons.append("insufficient_contrast")
            if peak_count > 1:
                reasons.append("competing_transitions")
            if bound_contact:
                reasons.append("search_bound_contact")
            if not reasons:
                radii[k], accepted[k] = peak_radius, True
        diagnostics.append({"candidate_id": candidate_id, "accepted": bool(accepted[k]),
                            "reasons": reasons, "angular_coverage_min": float(np.min(coverage)),
                            "angular_coverage_mean": float(np.mean(coverage)),
                            "contrast": contrast, "peak_radius": peak_radius, "peak_drop": peak_drop,
                            "competing_peak_count": int(peak_count), "search_bound_contact": bound_contact,
                            "invalid_probability_count": invalid_count})
    return RadiusProposals(_readonly(radii), _readonly(accepted, bool), tuple(diagnostics),
                           catalog.candidate_ids, catalog.fingerprint, options)


def apply_proposals_with_fallback(catalog, proposals, *, fallback=None):
    """Retain original radii for Gaussian rejection, or Gaussian radii for neural rejection.

    ``fallback`` must itself preserve this catalogue's identity and fixed centres.
    With every neural proposal rejected, numerical geometry equals the Gaussian
    refinement exactly, including when that refinement retained original radii.
    """
    if proposals.catalog_fingerprint != catalog.fingerprint or proposals.candidate_ids != catalog.candidate_ids:
        raise ValueError("proposals belong to a different candidate catalogue")
    baseline = catalog.geometry() if fallback is None else fallback
    if (not np.array_equal(baseline.centres, catalog.centres)
            or baseline.radii.shape != catalog.radii.shape
            or tuple(baseline.metadata.get("candidate_ids", ())) != catalog.candidate_ids
            or baseline.metadata.get("catalog_fingerprint") != catalog.fingerprint):
        raise ValueError("fallback must preserve catalogue identities and centres")
    radii = np.where(proposals.accepted, proposals.radii, baseline.radii)
    metadata = copy.deepcopy(baseline.metadata)
    metadata.update({"boundary_proposals": proposals.to_dict(),
                     "fallback_count": int(np.sum(~proposals.accepted)),
                     "fallback_source": "original_detector" if fallback is None else "supplied_baseline"})
    return ParticleGeometry(catalog.centres, radii, metadata)
