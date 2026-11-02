"""Versioned, domain-checked inference for separated spherical precipitates.

This is an experimental numerical fitter, not an uncertainty or identification
certificate. It uses the surface-average screened-field equations of ``physics``.
Unlike the archived joint fitter, a source is continued *constantly* inside its
own sphere, by replacing its radial distance with max(distance, radius). This
removes the unphysical centre singularity from the interior blend and leaves the
exterior matrix field unchanged. Every corrected comparison arm must use this
same version. The old-to-new comparison therefore changes both model and solver.

Geometry is checked before every coupling solve. Raw probabilities are checked
before a likelihood is evaluated; invalid values never receive a clipped finite
loss. SciPy trust-constr receives analytic geometry Jacobians and autograd
likelihood/probability derivatives. The default v2 backend projects search
directions onto linearized feasible constraints, and backtracks prohibited trials
before evaluating their likelihood. The original trust-constr prototype remains
selectable and terminates an attempt on a prohibited trial. Constraints are finite-sample checks, not a
proof of probability validity at every point in the box.
"""

from dataclasses import dataclass
import math
import time
import warnings

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import BFGS, Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult, linprog, minimize

from geom_mesh_net.fields.physics import screening_length


MODEL_VERSION = "screened_spheres_constant_interior_v1"
BACKEND_VERSION = "feasible_lbfgs_v2"
DEFAULT_STARTS = ((None, None), (1.0, 3.0), (6.0, 12.0))
PHYSICAL_BOUNDS = {"c_eq": (1e-6, 0.999), "ell": (0.0, 50.0),
                   "xi": (0.5, 200.0), "c_inf": (1e-6, 0.999)}
GEOMETRY_SCALE = 5.0
NUMERICAL_INTERIOR_SLACK = 1e-8
PROBABILITY_MARGIN = 1e-8
DOMAIN_TOLERANCE = 1e-10
LOG_EPSILON = 1e-12
CHUNK = 20000


class DomainError(ValueError):
    """An evaluation was refused before an invalid physical likelihood existed."""


class EvaluationBudgetExceeded(RuntimeError):
    """The declared total evaluation budget of one start was exhausted."""


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _geometry_arrays(centres, radii):
    centres = np.array(centres, dtype=float, copy=True)
    radii = np.array(radii, dtype=float, copy=True)
    if centres.size == 0:
        centres = centres.reshape(0, 3)
    if centres.ndim != 2 or centres.shape[1] != 3 or radii.shape != (len(centres),):
        raise ValueError("centres must be (K, 3), and radii (K,)")
    return centres, radii


def _candidate_ids(candidate_ids, count):
    ids = [f"candidate_{i:04d}" for i in range(count)] if candidate_ids is None else list(candidate_ids)
    if len(ids) != count or len({str(i) for i in ids}) != count:
        raise ValueError("candidate_ids must contain exactly one unique ID per sphere")
    if any(not isinstance(i, (str, int, np.integer)) for i in ids):
        raise ValueError("candidate IDs must be strings or integers")
    return _json_safe(ids)


def geometry_diagnostics(centres, radii, *, box=60.0, min_gap=1e-5,
                         radius_bounds=(0.5, 20.0)):
    """Check complete, bounded, separated spheres without deleting candidates."""
    centres, radii = _geometry_arrays(centres, radii)
    finite = bool(np.isfinite(centres).all() and np.isfinite(radii).all())
    i, j = np.triu_indices(len(radii), 1)
    gaps = np.linalg.norm(centres[i] - centres[j], axis=1) - radii[i] - radii[j]
    box_clearance = np.minimum(centres - radii[:, None], box - centres - radii[:, None])
    low, high = radius_bounds
    separated = finite and bool(np.all(gaps >= min_gap - DOMAIN_TOLERANCE))
    contained = finite and bool(np.all(box_clearance >= -DOMAIN_TOLERANCE))
    bounded = finite and bool(np.all((radii >= low - DOMAIN_TOLERANCE)
                                     & (radii <= high + DOMAIN_TOLERANCE)))
    return _json_safe({"finite": finite, "separated": separated, "contained": contained,
                       "radius_bounds_satisfied": bounded,
                       "feasible": bool(len(radii) and separated and contained and bounded),
                       "minimum_surface_gap": float(gaps.min()) if len(gaps) else None,
                       "minimum_box_clearance": float(box_clearance.min()) if len(radii) else None,
                       "violating_pairs": int(np.sum(gaps < min_gap - DOMAIN_TOLERANCE)),
                       "min_gap": min_gap, "numerical_tolerance": DOMAIN_TOLERANCE})


def prepare_feasible_geometry(centres, radii, *, candidate_ids=None, box=60.0,
                              min_gap=1e-5, radius_bounds=(0.5, 20.0)):
    """Project radii onto a convex feasible set, preserving centres and IDs.

    Returns ``(radii_or_None, JSON_record)``. The objective is half the squared
    radius change, in physical units. An input with numerical interior clearance
    is returned unchanged; otherwise a recorded 1e-8 length slack avoids roundoff
    at the solver's boundary, where that slack is feasible at the radius floor.
    An impossible radius-floor configuration fails rather than dropping a sphere.
    """
    centres, radii = _geometry_arrays(centres, radii)
    ids = _candidate_ids(candidate_ids, len(radii))
    low, high = map(float, radius_bounds)
    if not np.isfinite(box) or box <= 0 or not np.isfinite(min_gap) or min_gap <= 0:
        raise ValueError("box and min_gap must be positive and finite")
    if not 0 < low < high or not np.isfinite(high):
        raise ValueError("radius_bounds must be finite with 0 < lower < upper")
    before = geometry_diagnostics(centres, radii, box=box, min_gap=min_gap, radius_bounds=radius_bounds)
    record = {"status": "failed", "candidate_ids": ids, "centres_unchanged": True,
              "candidate_count_unchanged": True, "before": before, "after": None,
              "original_radii": radii.tolist(), "radii": None, "changes": []}
    if not len(radii) or not before["finite"] or np.any(radii <= 0):
        record["reason"] = "empty_or_nonfinite_or_nonpositive_geometry"
        return None, _json_safe(record)
    strictly_prepared = (before["feasible"] and before["minimum_box_clearance"] >= NUMERICAL_INTERIOR_SLACK
                         and (before["minimum_surface_gap"] is None
                              or before["minimum_surface_gap"] >= min_gap + NUMERICAL_INTERIOR_SLACK))
    if strictly_prepared:
        record.update(status="unchanged", after=before, radii=radii.tolist())
        return radii.copy(), _json_safe(record)
    upper = np.minimum(high, np.minimum(centres, box - centres).min(axis=1) - NUMERICAL_INTERIOR_SLACK)
    i, j = np.triu_indices(len(radii), 1)
    separation = np.linalg.norm(centres[i] - centres[j], axis=1) - min_gap - NUMERICAL_INTERIOR_SLACK
    if np.any(upper < low) or np.any(separation < 2 * low):
        if before["feasible"]:
            record.update(status="unchanged_boundary", after=before, radii=radii.tolist(),
                          numerical_interior_slack=0.0,
                          reason="strict_numerical_interior_unavailable_at_radius_floor")
            return radii.copy(), _json_safe(record)
        record["reason"] = "no_feasible_radii_at_preserved_centres"
        return None, _json_safe(record)
    rows = np.arange(len(i))
    matrix = sparse.coo_matrix((np.ones(2 * len(i)),
                                (np.r_[rows, rows], np.r_[i, j])),
                               shape=(len(i), len(radii))).toarray()
    constraints = [] if not len(i) else [LinearConstraint(matrix, -np.inf, separation)]
    result = minimize(lambda r: 0.5 * np.sum((r - radii) ** 2), np.full(len(radii), low),
                      jac=lambda r: r - radii, method="SLSQP", bounds=Bounds(low, upper),
                      constraints=constraints, options={"ftol": 1e-12, "maxiter": 500})
    adjusted = np.asarray(result.x, dtype=float)
    # A tiny inward move repairs floating-point residuals only, not an infeasible
    # optimization. Its magnitude and every final radius change are recorded.
    inward = 0.0
    if len(i) and np.max(adjusted[i] + adjusted[j] - separation) > 0:
        inward = max(0.0, float(np.max(adjusted[i] + adjusted[j] - separation)) / 2 + 1e-10)
        if inward <= 1e-7:
            adjusted = np.maximum(low, adjusted - inward)
    after = geometry_diagnostics(centres, adjusted, box=box, min_gap=min_gap, radius_bounds=radius_bounds)
    record.update(after=after, optimizer_success=bool(result.success), message=str(result.message),
                  numerical_interior_slack=NUMERICAL_INTERIOR_SLACK,
                  objective=float(result.fun), numerical_inward_adjustment=inward)
    if not result.success or not after["feasible"]:
        record["reason"] = "radius_projection_failed"
        return None, _json_safe(record)
    record.update(status="projected", radii=adjusted.tolist(),
                  changes=[{"candidate_id": ids[k], "before": float(radii[k]),
                            "after": float(adjusted[k]), "change": float(adjusted[k] - radii[k])}
                           for k in range(len(radii)) if adjusted[k] != radii[k]])
    return adjusted, _json_safe(record)


def surface_directions(count=32):
    """Deterministic quadrature directions, independent of any atom labels."""
    index = np.arange(count, dtype=float) + 0.5
    z = 1 - 2 * index / count
    angle = np.pi * (3 - np.sqrt(5)) * index
    ring = np.sqrt(1 - z * z)
    return np.vstack([np.column_stack([ring * np.cos(angle), ring * np.sin(angle), z]),
                      np.eye(3), -np.eye(3)])


@dataclass(frozen=True)
class ParameterLayout:
    count: int
    knots: int = 6
    fixed_ell: float | None = None

    @property
    def chemistry(self):
        return 4 * self.count

    @property
    def shape_start(self):
        return self.chemistry + (4 if self.fixed_ell is None else 3)

    @property
    def size(self):
        return self.shape_start + self.knots + 1

    def bounds(self, box, radius_bounds):
        low = [0.0] * (3 * self.count) + [radius_bounds[0] / GEOMETRY_SCALE] * self.count
        high = [box / GEOMETRY_SCALE] * (3 * self.count) + [radius_bounds[1] / GEOMETRY_SCALE] * self.count
        for key in ("c_eq", "xi", "c_inf"):
            low.append(math.log(PHYSICAL_BOUNDS[key][0]))
            high.append(math.log(PHYSICAL_BOUNDS[key][1]))
        if self.fixed_ell is None:
            low.append(0.0)
            high.append(PHYSICAL_BOUNDS["ell"][1] / GEOMETRY_SCALE)
        low += [-12.0] * (self.knots + 1)
        high += [12.0] * (self.knots + 1)
        return np.asarray(low), np.asarray(high)


class ConstrainedDiffusionModel:
    """Float64 screened field with explicit parameter coordinates and raw output."""

    def __init__(self, centres, radii, *, c_eq=0.03, ell=3.0, xi=6.0, c_inf=0.1,
                 width=0.3, knots=6, levels=None, candidate_ids=None, box=60.0,
                 min_gap=1e-5, radius_bounds=(0.5, 20.0), fixed_ell=None):
        centres, radii = _geometry_arrays(centres, radii)
        self.candidate_ids = _candidate_ids(candidate_ids, len(radii))
        self.layout = ParameterLayout(len(radii), int(knots), fixed_ell)
        self.box, self.min_gap = float(box), float(min_gap)
        self.radius_bounds, self.fixed_width = tuple(radius_bounds), float(width)
        if self.layout.knots < 1 or int(knots) != knots or width <= 0:
            raise ValueError("knots must be a positive integer and width positive")
        if fixed_ell is not None and (not np.isfinite(fixed_ell) or not 0 <= fixed_ell <= 50):
            raise ValueError("fixed_ell must lie in [0, 50]")
        if levels is None:
            u = np.linspace(0, 1, int(knots) + 1)
            levels = np.minimum(1 - 0.7 * u ** 3, 0.9999)
        levels = np.asarray(levels, dtype=float)
        if levels.shape != (int(knots) + 1,) or np.any(levels <= 0) or np.any(levels >= 1) or np.any(np.diff(levels) >= 0):
            raise ValueError("levels must be strictly decreasing probabilities, one at each knot")
        ratios = levels[1:] / levels[:-1]
        logits = np.r_[math.log(levels[0] / (1 - levels[0])), np.log(ratios / (1 - ratios))]
        vector = np.r_[centres.ravel() / GEOMETRY_SCALE, radii / GEOMETRY_SCALE,
                       np.log([c_eq, xi, c_inf]), [] if fixed_ell is not None else [ell / GEOMETRY_SCALE], logits]
        self.vector = torch.tensor(vector, dtype=torch.float64)
        self.model_version = MODEL_VERSION

    def set_vector(self, vector):
        if np.shape(vector) != (self.layout.size,):
            raise ValueError("incorrect parameter vector size")
        self.vector = vector if isinstance(vector, torch.Tensor) else torch.tensor(vector, dtype=torch.float64)
        return self

    def centres(self):
        return self.vector[:3 * self.layout.count].reshape(-1, 3) * GEOMETRY_SCALE

    def radii(self):
        return self.vector[3 * self.layout.count:4 * self.layout.count] * GEOMETRY_SCALE

    def value(self, name):
        if name == "ell":
            if self.layout.fixed_ell is not None:
                return self.vector.new_tensor(self.layout.fixed_ell)
            return self.vector[self.layout.chemistry + 3] * GEOMETRY_SCALE
        index = {"c_eq": 0, "xi": 1, "c_inf": 2}[name]
        return torch.exp(self.vector[self.layout.chemistry + index])

    def constants(self):
        return {key: float(self.value(key).detach()) for key in PHYSICAL_BOUNDS}

    def levels(self):
        logits = self.vector[self.layout.shape_start:]
        return torch.sigmoid(logits[0]) * torch.cat([logits.new_ones(1),
                                                    torch.cumprod(torch.sigmoid(logits[1:]), dim=0)])

    def shape(self):
        return {"knots": self.layout.knots, "levels": self.levels().detach().numpy().tolist(),
                "width": self.fixed_width}

    def interior(self, u):
        position = torch.clamp(u, 0.0, 1.0) * self.layout.knots
        lower = torch.clamp(position.detach().floor().long(), max=self.layout.knots - 1)
        fraction, levels = position - lower, self.levels()
        return levels[lower] * (1 - fraction) + levels[lower + 1] * fraction

    def guard_geometry(self):
        record = geometry_diagnostics(self.centres().detach().numpy(), self.radii().detach().numpy(),
                                      box=self.box, min_gap=self.min_gap, radius_bounds=self.radius_bounds)
        if not record["feasible"]:
            raise DomainError(f"geometry refused before coupling solve: {record}")
        return record

    def coupling(self):
        self.guard_geometry()
        radii, centres, xi = self.radii(), self.centres(), self.value("xi")
        eye = torch.eye(self.layout.count, dtype=torch.float64)
        distances = torch.cdist(centres, centres) + eye
        # Algebraically equivalent to exp(-(d-Rj)/xi)*sinh(Ri/xi)/(Ri/xi),
        # but separated spheres keep both exponentials bounded by one.
        z = radii / xi
        attenuation = (torch.exp(-(distances - radii[None, :] - radii[:, None]) / xi)
                       - torch.exp(-(distances - radii[None, :] + radii[:, None]) / xi)) / (2 * z[:, None])
        matrix = (radii[None, :] / distances) * attenuation
        return matrix * (1 - eye) + eye

    def amplitudes(self, *, return_diagnostics=False):
        matrix = self.coupling()
        surface = self.value("c_eq") * torch.exp(self.value("ell") / self.radii())
        if not bool(torch.isfinite(matrix).all() and torch.isfinite(surface).all()):
            raise DomainError("nonfinite coupling matrix or Gibbs-Thomson surface means")
        condition = float(torch.linalg.cond(matrix.detach()))
        if not np.isfinite(condition) or condition > 1e10:
            raise DomainError(f"ill-conditioned coupling matrix: {condition}")
        rhs = surface - self.value("c_inf")
        amplitudes = torch.linalg.solve(matrix, rhs)
        relative_residual = float((torch.linalg.vector_norm(matrix @ amplitudes - rhs)
                                   / (torch.linalg.vector_norm(rhs) + 1e-15)).detach())
        if not bool(torch.isfinite(amplitudes).all()) or relative_residual > 1e-9:
            raise DomainError("coupling solve failed finite-value/residual check")
        diagnostics = {"condition_number": condition, "relative_residual": relative_residual,
                       "maximum_absolute_amplitude": float(amplitudes.detach().abs().max())}
        return (amplitudes, diagnostics) if return_diagnostics else amplitudes

    def _raw(self, points, amplitudes, assignment=None):
        distance = torch.cdist(points, self.centres())
        radii, xi = self.radii(), self.value("xi")
        extended_distance = torch.maximum(distance, radii[None, :])
        kernel = (radii[None, :] / extended_distance) * torch.exp(-(extended_distance - radii[None, :]) / xi)
        matrix = self.value("c_inf") + kernel @ amplitudes
        if assignment is None:
            assignment = torch.argmin(distance - radii[None, :], dim=1)
        nearest = distance.gather(1, assignment[:, None]).squeeze(1)
        radius = radii[assignment]
        blend = torch.sigmoid((radius - nearest) / self.fixed_width)
        full = matrix + (self.interior(nearest / radius) - matrix) * blend
        outside = torch.all(distance >= radii[None, :] - DOMAIN_TOLERANCE, dim=1)
        return full, matrix, outside

    def matrix_field(self, points):
        points = torch.as_tensor(points, dtype=torch.float64)
        return self._raw(points, self.amplitudes())[1]

    def forward(self, points, assignment=None):
        points = torch.as_tensor(points, dtype=torch.float64)
        return self._raw(points, self.amplitudes(), assignment)[0]

    __call__ = forward

    def assign(self, points):
        points = torch.as_tensor(points, dtype=torch.float64)
        with torch.no_grad():
            return torch.argmin(torch.cdist(points, self.centres()) - self.radii()[None, :], dim=1)

    def _predict_numpy(self, points, matrix_only=False, chunk=CHUNK):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        out = []
        with torch.no_grad():
            amplitudes = self.amplitudes()
            for start in range(0, len(points), chunk):
                values = self._raw(torch.as_tensor(points[start:start + chunk], dtype=torch.float64), amplitudes)
                out.append(values[1 if matrix_only else 0].numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def raw_predict(self, points, chunk=CHUNK):
        return self._predict_numpy(points, chunk=chunk)

    def raw_matrix(self, points, chunk=CHUNK):
        return self._predict_numpy(points, matrix_only=True, chunk=chunk)

    predict = raw_predict


def raw_probability_diagnostics(full, matrix, outside):
    """Validate the actual blend and the matrix only at exterior sample points."""
    full = np.asarray(full, dtype=float)
    matrix = np.asarray(matrix, dtype=float)[np.asarray(outside, dtype=bool)]
    def describe(values):
        finite = np.isfinite(values)
        invalid = ~finite | (values < -DOMAIN_TOLERANCE) | (values > 1 + DOMAIN_TOLERANCE)
        return {"points": len(values), "nonfinite": int((~finite).sum()),
                "outside_probability_range": int(invalid.sum()),
                "minimum": float(values[finite].min()) if finite.any() else None,
                "maximum": float(values[finite].max()) if finite.any() else None,
                "passed": bool(len(values) and not invalid.any())}
    full_record, matrix_record = describe(full), describe(matrix)
    return {"full": full_record, "matrix": matrix_record,
            "passed": full_record["passed"] and matrix_record["passed"],
            "domain_tolerance": DOMAIN_TOLERANCE, "scope": "sampled_positions_only"}


def _probe_points(model, fixed):
    directions = torch.as_tensor(surface_directions(), dtype=torch.float64)
    surface = model.centres()[:, None, :] + model.radii()[:, None, None] * directions[None, :, :]
    return torch.cat([fixed, surface.reshape(-1, 3)])


def _fixed_probes(box):
    grid = np.linspace(0, box, 5)
    return torch.as_tensor(np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3),
                           dtype=torch.float64)


class _Evaluation:
    """Cache a value, gradient and sampled-domain Jacobian at one parameter vector."""

    def __init__(self, model, x, y, assignment, budget, trace, fixed_probes):
        self.model, self.x, self.y, self.assignment = model, x, y, assignment
        self.budget, self.trace, self.fixed_probes = budget, trace, fixed_probes
        self.cached_vector = None
        self.lower, self.upper = model.layout.bounds(model.box, model.radius_bounds)

    def _differentiate(self):
        value, gradient, constraint_values, jacobian = self.cached
        if gradient is not None:
            return self.cached
        gradient = torch.autograd.grad(self.cached_loss, self.cached_parameter, retain_graph=True)[0].detach().numpy()
        jacobian = np.vstack([torch.autograd.grad(item, self.cached_parameter, retain_graph=True)[0].detach().numpy()
                              for item in self.cached_constraints])
        if not np.isfinite(gradient).all() or not np.isfinite(jacobian).all():
            raise DomainError("nonfinite likelihood or constraint derivative")
        self.cached = (value, gradient, constraint_values, sparse.csr_matrix(jacobian))
        return self.cached

    def evaluate(self, vector, *, derivatives=True):
        vector = np.asarray(vector, dtype=float)
        if self.cached_vector is not None and np.array_equal(vector, self.cached_vector):
            self.model.set_vector(self.cached_parameter)
            return self._differentiate() if derivatives else self.cached
        if len(self.trace) >= self.budget:
            raise EvaluationBudgetExceeded("maximum total evaluations reached for this start")
        row = {"evaluation": len(self.trace), "geometry_feasible": False,
               "probability_valid": False, "likelihood_evaluated": False,
               "mean_nll": None, "minimum_surface_gap": None}
        self.trace.append(row)
        parameter = torch.tensor(vector, dtype=torch.float64, requires_grad=True)
        self.model.set_vector(parameter)
        try:
            geometry = geometry_diagnostics(self.model.centres().detach().numpy(),
                                              self.model.radii().detach().numpy(), box=self.model.box,
                                              min_gap=self.model.min_gap, radius_bounds=self.model.radius_bounds)
            row.update(geometry_feasible=geometry["feasible"],
                       minimum_surface_gap=geometry["minimum_surface_gap"],
                       minimum_box_clearance=geometry["minimum_box_clearance"])
            if not np.isfinite(vector).all() or np.any(vector < self.lower - DOMAIN_TOLERANCE) or np.any(vector > self.upper + DOMAIN_TOLERANCE):
                raise DomainError("parameter search bound violated before coupling solve")
            if not geometry["feasible"]:
                raise DomainError("geometry refused before coupling solve")
            amplitudes, solve = self.model.amplitudes(return_diagnostics=True)
            full, matrix, outside = self.model._raw(self.x, amplitudes, self.assignment)
            probe_full, probe_matrix, probe_outside = self.model._raw(_probe_points(self.model, self.fixed_probes), amplitudes)
            all_full = torch.cat([full, probe_full])
            all_matrix = torch.cat([matrix[outside], probe_matrix[probe_outside]])
            probability = raw_probability_diagnostics(all_full.detach().numpy(), all_matrix.detach().numpy(),
                                                       np.ones(len(all_matrix), dtype=bool))
            row.update(solve=solve, probability=probability)
            if not probability["passed"]:
                raise DomainError("raw probability outside physical domain; likelihood not evaluated")
            row["probability_valid"] = True
            # Safeguarding logs applies only after raw-domain checks above.
            safe = torch.clamp(full, LOG_EPSILON, 1 - LOG_EPSILON)
            nll = -(self.y * torch.log(safe) + (1 - self.y) * torch.log1p(-safe)).mean()
            probability_constraints = torch.stack([all_full.min() - PROBABILITY_MARGIN,
                                                    1 - PROBABILITY_MARGIN - all_full.max(),
                                                    all_matrix.min() - PROBABILITY_MARGIN,
                                                    1 - PROBABILITY_MARGIN - all_matrix.max()])
            value = float(nll.detach())
            if not np.isfinite(value):
                raise DomainError("nonfinite likelihood")
            row.update(likelihood_evaluated=True, mean_nll=value,
                       minimum_probability_constraint=float(probability_constraints.detach().min()))
            self.cached = (value, None, probability_constraints.detach().numpy(), None)
            self.cached_parameter, self.cached_loss = parameter, nll
            self.cached_constraints = probability_constraints
            self.cached_vector = vector.copy()
            return self._differentiate() if derivatives else self.cached
        except (DomainError, RuntimeError) as error:
            row["error"] = str(error)
            raise

    def fun(self, vector):
        return self.evaluate(vector)[0]

    def jac(self, vector):
        return self.evaluate(vector)[1]

    def constraints(self, vector):
        return self.evaluate(vector)[2]

    def constraint_jac(self, vector):
        return self.evaluate(vector)[3]


def geometry_constraints(layout, *, box=60.0, min_gap=1e-5):
    """Sparse, analytic separation and sphere-containment constraints."""
    count, size = layout.count, layout.size
    i, j = np.triu_indices(count, 1)
    def pairs(vector):
        centres = vector[:3 * count].reshape(count, 3) * GEOMETRY_SCALE
        radii = vector[3 * count:4 * count] * GEOMETRY_SCALE
        return np.linalg.norm(centres[i] - centres[j], axis=1) - radii[i] - radii[j] - min_gap
    def pair_jac(vector):
        centres = vector[:3 * count].reshape(count, 3) * GEOMETRY_SCALE
        delta = centres[i] - centres[j]
        distance = np.linalg.norm(delta, axis=1)
        direction = np.divide(delta, distance[:, None], out=np.zeros_like(delta), where=distance[:, None] > 0)
        rows = np.repeat(np.arange(len(i)), 8)
        columns = np.column_stack([3 * i, 3 * i + 1, 3 * i + 2,
                                   3 * j, 3 * j + 1, 3 * j + 2, 3 * count + i, 3 * count + j]).ravel()
        values = GEOMETRY_SCALE * np.column_stack([direction, -direction, -np.ones((len(i), 2))]).ravel()
        return sparse.csr_matrix((values, (rows, columns)), shape=(len(i), size))
    row, col, val = [], [], []
    for k in range(count):
        for d in range(3):
            lower_row, upper_row = 6 * k + 2 * d, 6 * k + 2 * d + 1
            row.extend([lower_row, lower_row, upper_row, upper_row])
            col.extend([3 * k + d, 3 * count + k, 3 * k + d, 3 * count + k])
            val.extend([GEOMETRY_SCALE, -GEOMETRY_SCALE, -GEOMETRY_SCALE, -GEOMETRY_SCALE])
    containment = sparse.csr_matrix((val, (row, col)), shape=(6 * count, size))
    lower = np.tile([0.0, -box], 3 * count)
    constraints = [LinearConstraint(containment, lower, np.inf, keep_feasible=True)]
    if len(i):
        constraints.append(NonlinearConstraint(pairs, 0.0, np.inf, jac=pair_jac,
                                               hess=BFGS(), keep_feasible=True))
    return constraints


def _linearized_constraints(vector, evaluated, constraints, lower, upper):
    """Return A,b for A d + b >= 0, including bounds and raw-probability guards."""
    matrices, slacks = [], []
    for constraint in constraints:
        if isinstance(constraint, LinearConstraint):
            matrix, values = sparse.csr_matrix(constraint.A), constraint.A @ vector
        else:
            matrix, values = sparse.csr_matrix(constraint.jac(vector)), constraint.fun(vector)
        low = np.broadcast_to(constraint.lb, np.shape(values))
        high = np.broadcast_to(constraint.ub, np.shape(values))
        finite_low, finite_high = np.isfinite(low), np.isfinite(high)
        matrices.extend([matrix[finite_low], -matrix[finite_high]])
        slacks.extend([values[finite_low] - low[finite_low], high[finite_high] - values[finite_high]])
    identity = sparse.eye(len(vector), format="csr")
    matrices.extend([identity, -identity, evaluated[3]])
    slacks.extend([vector - lower, upper - vector, evaluated[2]])
    matrix, slack = sparse.vstack(matrices, format="csr"), np.concatenate(slacks)
    if np.min(slack) < -DOMAIN_TOLERANCE:
        raise DomainError("accepted point outside linearized feasible constraint domain")
    # Row scaling changes no feasible halfspace and improves the sparse dual QP.
    scale = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    scale = np.maximum(scale, 1e-12)
    matrix = sparse.diags(1 / scale) @ matrix
    return matrix.tocsr(), np.maximum(slack, 0.0) / scale


def _refine_projection_dual(matrix, linear, initial, *, tolerance, maxiter=500):
    """Active-set refinement of a convex nonnegative dual quadratic program.

    Approximate L-BFGS-B dual solutions can stop on objective precision while
    still missing a strict feasibility certificate. Solve the active KKT system
    directly, moving only through nonnegative multipliers; add a violated
    inactive row or remove a multiplier that reaches zero deterministically.
    """
    multipliers = np.maximum(np.asarray(initial, dtype=float), 0.)
    active = set(np.flatnonzero(multipliers > 1e-12).tolist())
    for iteration in range(maxiter):
        indices = np.array(sorted(active), dtype=int)
        candidate = np.zeros_like(multipliers)
        if len(indices):
            rows = matrix[indices]
            gram = (rows @ rows.T).toarray()
            candidate[indices] = np.linalg.lstsq(gram, -linear[indices], rcond=1e-13)[0]
            # Handle an inconsistent singular active system before stepping
            # toward its least-squares candidate. Its residual is in the Gram
            # nullspace, so -residual is a dual descent direction from the
            # current nonnegative multipliers until a bound is encountered.
            active_residual = gram @ candidate[indices] + linear[indices]
            if np.max(np.abs(active_residual)) > 10 * tolerance:
                positive = active_residual > tolerance * .1
                if not np.any(positive):
                    return multipliers, iteration + 1
                step = float(np.min(multipliers[indices[positive]] / active_residual[positive]))
                multipliers[indices] -= step * active_residual
                removed = np.flatnonzero(multipliers <= 1e-12)
                active.difference_update(removed.tolist())
                multipliers[removed] = 0.
                continue
            negative = indices[candidate[indices] < 0.]
            if len(negative):
                denominator = multipliers[negative] - candidate[negative]
                fractions = np.divide(multipliers[negative], denominator,
                                      out=np.zeros(len(negative)), where=denominator > 0)
                step = min(1., float(np.min(fractions)))
                multipliers += step * (candidate - multipliers)
                removed = np.flatnonzero(multipliers <= 1e-12)
                active.difference_update(removed.tolist())
                multipliers[removed] = 0.
                continue
        multipliers = candidate
        gradient = matrix @ (matrix.T @ multipliers) + linear
        inactive = np.ones(len(multipliers), dtype=bool)
        inactive[list(active)] = False
        if np.any(inactive):
            violation = np.where(inactive, gradient, np.inf)
            worst = int(np.argmin(violation))
            if violation[worst] < -tolerance * .1:
                active.add(worst)
                continue
        return multipliers, iteration + 1
    return multipliers, maxiter


def _project_direction(desired, matrix, slack, *, tolerance=1e-9):
    """Euclidean projection onto linearized constraints using a convex dual QP.

    The dual is min_{lambda>=0} .5||A.T lambda||^2 + (A desired+b).lambda.
    Its primal solution is d=desired+A.T lambda. Both primal feasibility and
    dual complementarity are checked explicitly; a solver success flag alone is
    insufficient. Returning a direction also returns its KKT evidence.
    """
    desired = np.asarray(desired, dtype=float)
    initial_slack = matrix @ desired + slack
    if np.min(initial_slack) >= 0:
        return desired.copy(), np.zeros(len(slack)), {
            "converged": True, "iterations": 0, "primal_violation": 0.0,
            "dual_projected_gradient": 0.0, "complementarity": 0.0,
            "stationarity": 0.0}
    def dual(multipliers):
        correction = matrix.T @ multipliers
        return (0.5 * np.dot(correction, correction) + np.dot(initial_slack, multipliers),
                matrix @ correction + initial_slack)
    result = minimize(dual, np.zeros(len(slack)), jac=True, method="L-BFGS-B",
                      bounds=[(0., None)] * len(slack),
                      options={"maxiter": 500, "maxls": 40, "ftol": 1e-15,
                               "gtol": tolerance * .1, "maxcor": 15})
    def certify(multipliers):
        correction = matrix.T @ multipliers
        direction = desired + correction
        residual_slack = matrix @ direction + slack
        dual_residual = float(np.max(np.abs(np.where(multipliers > 1e-10,
                                                     residual_slack, np.minimum(residual_slack, 0)))))
        primal_violation = float(max(0., -np.min(residual_slack)))
        complementarity = float(np.max(np.abs(multipliers * residual_slack)))
        stationarity = float(np.max(np.abs(direction - desired - correction)))
        passed = (primal_violation <= tolerance and dual_residual <= 10 * tolerance
                  and complementarity <= 10 * tolerance and stationarity <= tolerance)
        return direction, {"converged": passed, "primal_violation": primal_violation,
                           "dual_projected_gradient": dual_residual,
                           "complementarity": complementarity, "stationarity": stationarity}
    multipliers = np.maximum(result.x, 0.)
    direction, record = certify(multipliers)
    refinement_iterations = 0
    if not record["converged"]:
        multipliers, refinement_iterations = _refine_projection_dual(matrix, initial_slack, multipliers,
                                                                      tolerance=tolerance)
        direction, record = certify(multipliers)
    record.update(iterations=int(result.nit), active_set_iterations=refinement_iterations,
                  termination=str(result.message))
    if not record["converged"]:
        raise DomainError(f"direction projection did not meet its KKT tolerance: {record}")
    return direction, multipliers, record


def _lbfgs_direction(gradient, memory):
    working, alpha = gradient.copy(), []
    for step, change in reversed(memory):
        rho = 1 / np.dot(step, change)
        coefficient = rho * np.dot(step, working)
        alpha.append(coefficient)
        working -= coefficient * change
    scale = 1.0
    if memory:
        step, change = memory[-1]
        scale = float(np.clip(np.dot(step, change) / np.dot(change, change), 1e-4, 1e4))
    working *= scale
    for (step, change), coefficient in zip(memory, reversed(alpha)):
        beta = np.dot(change, working) / np.dot(step, change)
        working += step * (coefficient - beta)
    return -working


def _minimize_feasible(evaluator, vector, constraints, lower, upper, *, maxiter, gtol,
                       progress_callback=None):
    """Projected quasi-Newton steps with exact physical-domain backtracking.

    A separate unit-step Euclidean projected-gradient QP supplies the KKT
    stationarity criterion. A short step, stalled line search, or iteration cap
    can never certify convergence. All rejected physical trials remain in the
    evaluator's trace, including ones refused before their likelihood existed.
    """
    vector = np.asarray(vector, dtype=float).copy()
    evaluated = evaluator.evaluate(vector)
    memory, history = [], []
    success, message, status = False, "maximum accepted iterations reached", 0
    optimality, complementarity, violation = math.inf, math.inf, math.inf
    for iteration in range(maxiter + 1):
        value, gradient = evaluated[:2]
        matrix, slack = _linearized_constraints(vector, evaluated, constraints, lower, upper)
        projected_gradient, multipliers, certificate = _project_direction(-gradient, matrix, slack)
        optimality = float(np.max(np.abs(projected_gradient)))
        complementarity = float(np.max(np.abs(multipliers * slack)))
        violation = float(max(0., -np.min(slack)))
        row = {"iteration": iteration, "mean_nll": value, "projected_gradient": optimality,
               "complementarity": complementarity, "constraint_violation": violation,
               "projection_certificate": certificate}
        history.append(row)
        if progress_callback is not None and (iteration % 20 == 0 or iteration == maxiter):
            progress_callback(_json_safe({"event": "iteration", **row}))
        if optimality <= gtol and complementarity <= gtol and violation <= DOMAIN_TOLERANCE:
            success, message, status = True, "projected-gradient and complementarity tolerances satisfied", 1
            break
        if iteration == maxiter:
            break
        desired = _lbfgs_direction(gradient, memory)
        # This finite trust radius limits physical moves, without defining
        # convergence or preventing exact feasible backtracking.
        desired /= max(1.0, np.max(np.abs(desired)))
        direction, _, direction_certificate = _project_direction(desired, matrix, slack)
        directional_derivative = float(np.dot(gradient, direction))
        if directional_derivative >= -1e-12 * max(1., np.linalg.norm(gradient)):
            direction = projected_gradient
            directional_derivative = float(np.dot(gradient, direction))
            row["direction"] = "projected_gradient"
        else:
            row["direction"] = "projected_lbfgs"
        row["direction_projection"] = direction_certificate
        if directional_derivative >= 0:
            message, status = "no feasible descent direction despite nonzero KKT residual", 3
            break
        step_size, accepted, backtracks = 1.0, False, 0
        for backtracks in range(40):
            trial = vector + step_size * direction
            try:
                trial_evaluated = evaluator.evaluate(trial, derivatives=False)
                # Safety-margin feasibility is distinct from material raw-domain
                # validity; the latter was checked before this likelihood.
                sampled_feasible = np.min(trial_evaluated[2]) >= -DOMAIN_TOLERANCE
                if sampled_feasible and trial_evaluated[0] <= value + 1e-4 * step_size * directional_derivative:
                    trial_evaluated = evaluator.evaluate(trial)
                    accepted = True
                    break
            except DomainError:
                pass  # Recorded rejected trial; halve and retry without a penalty.
            step_size *= .5
        row.update(backtracks=backtracks, step_size=step_size, accepted=accepted)
        if not accepted:
            message, status = "feasible Armijo line search stalled with nonzero KKT residual", 2
            break
        step, change = trial - vector, trial_evaluated[1] - gradient
        if np.dot(step, change) > 1e-10 * max(1e-20, np.linalg.norm(step) * np.linalg.norm(change)):
            memory.append((step, change))
            memory = memory[-10:]
        vector, evaluated = trial, trial_evaluated
    evaluator.model.set_vector(vector)
    return OptimizeResult(x=vector, fun=evaluated[0], jac=evaluated[1], success=success,
                          status=status, message=message, nit=len(history) - 1,
                          optimality=max(optimality, complementarity),
                          constr_violation=violation, history=history)


def _initial_probability_projection(model, points):
    """Find nearby feasible chemistry at prescribed ell/xi using a tiny linear program.

    For fixed geometry, ell, xi and interior, both raw predictions are affine in
    c_eq and c_inf. Minimize their maximum absolute change from the supplied
    concentrations (by default mean/2 and mean; explicit warm starts are retained).
    This feasibility preparation has no held-back labels and is not fit selection.
    """
    before = model.constants()
    with torch.no_grad():
        coupling = model.coupling()
        condition = float(torch.linalg.cond(coupling))
        if not np.isfinite(condition) or condition > 1e10:
            raise DomainError(f"ill-conditioned initial coupling matrix: {condition}")
        c_basis = torch.exp(model.value("ell") / model.radii())
        amplitude_eq = torch.linalg.solve(coupling, c_basis)
        amplitude_inf = torch.linalg.solve(coupling, -torch.ones_like(c_basis))
        distance = torch.cdist(points, model.centres())
        extended = torch.maximum(distance, model.radii()[None, :])
        kernels = model.radii()[None, :] / extended * torch.exp(-(extended - model.radii()[None, :]) / model.value("xi"))
        matrix_eq, matrix_inf = kernels @ amplitude_eq, 1 + kernels @ amplitude_inf
        assignment = torch.argmin(distance - model.radii()[None, :], dim=1)
        nearest = distance.gather(1, assignment[:, None]).squeeze(1)
        radius = model.radii()[assignment]
        blend = torch.sigmoid((radius - nearest) / model.fixed_width)
        offset = blend * model.interior(nearest / radius)
        outside = torch.all(distance >= model.radii()[None, :] - DOMAIN_TOLERANCE, dim=1)
        coefficients = torch.cat([torch.stack([(1 - blend) * matrix_eq, (1 - blend) * matrix_inf], dim=1),
                                  torch.stack([matrix_eq[outside], matrix_inf[outside]], dim=1)]).numpy()
        offset = torch.cat([offset, torch.zeros(int(outside.sum()), dtype=torch.float64)]).numpy()
    target = np.array([before["c_eq"], before["c_inf"]])
    matrix = np.column_stack([coefficients, np.zeros(len(coefficients))])
    # t bounds the absolute change in the two concentration parameters.
    deviation = np.array([[1, 0, -1], [-1, 0, -1], [0, 1, -1], [0, -1, -1]], dtype=float)
    result = linprog([0.0, 0.0, 1.0], A_ub=np.vstack([matrix, -matrix, deviation]),
                     b_ub=np.r_[1 - 10 * PROBABILITY_MARGIN - offset,
                                offset - 10 * PROBABILITY_MARGIN,
                                target[0], -target[0], target[1], -target[1]],
                     bounds=[PHYSICAL_BOUNDS["c_eq"], PHYSICAL_BOUNDS["c_inf"], (0, None)], method="highs")
    if not result.success:
        raise DomainError(f"prescribed physical start has no sampled feasible chemistry: {result.message}")
    vector = model.vector.detach().numpy().copy()
    vector[model.layout.chemistry] = math.log(float(result.x[0]))
    vector[model.layout.chemistry + 2] = math.log(float(result.x[1]))
    model.set_vector(vector)
    return {"method": "minimum_linf_concentration_change_linear_program", "before": before,
            "after": model.constants(), "maximum_change": float(result.x[2]),
            "uses_validation_labels_or_positions": False}


def _final_diagnostics(model, points):
    with torch.no_grad():
        amplitudes, solve = model.amplitudes(return_diagnostics=True)
        full, matrix, outside = model._raw(torch.as_tensor(points, dtype=torch.float64), amplitudes)
    probability = raw_probability_diagnostics(full.numpy(), matrix.numpy(), outside.numpy())
    return {"geometry": model.guard_geometry(), "probability": probability, "solve": solve}


def _bound_contacts(model, tolerance=0.01):
    constants = model.constants()
    contacts = []
    for name, value in constants.items():
        lo, hi = PHYSICAL_BOUNDS[name]
        if name == "ell" and model.layout.fixed_ell is not None:
            continue
        if value <= lo + tolerance * (1.0 if lo == 0 else lo):
            contacts.append({"parameter": name, "side": "lower", "value": value,
                             "kind": "physical_null_boundary" if name == "ell" else "search_bound"})
        if value >= hi / (1 + tolerance):
            contacts.append({"parameter": name, "side": "upper", "value": value, "kind": "search_bound"})
    radii = model.radii().detach().numpy()
    for k, radius in enumerate(radii):
        if radius <= model.radius_bounds[0] * (1 + tolerance) or radius >= model.radius_bounds[1] / (1 + tolerance):
            contacts.append({"parameter": "radius", "candidate_id": model.candidate_ids[k],
                             "value": radius, "kind": "search_bound"})
    logits = model.vector[model.layout.shape_start:].detach().numpy()
    for k in np.flatnonzero(np.abs(logits) >= 12 * (1 - tolerance)):
        contacts.append({"parameter": "interior_profile_logit", "index": int(k),
                         "value": float(logits[k]), "kind": "search_bound"})
    return _json_safe(contacts)


def fit_constrained(centres, radii, x, y, *, candidate_ids=None, box=60.0,
                    width=0.3, knots=6, starts=DEFAULT_STARTS, maxiter=150,
                    max_evaluations=600, rounds=3, fixed_ell=None, validation_points=None,
                    min_gap=1e-5, radius_bounds=(0.5, 20.0), gtol=1e-5,
                    initial_constants=None, backend="feasible_lbfgs", progress_callback=None):
    """Fit from fixed starts, returning ``(model_or_None, JSON_record)``.

    Centres and radii are optimized, preserving candidate order/IDs. Only feasible,
    converged, assignment-stable attempts compete by *fitting* likelihood. Optional
    validation positions are inspected after that selection; they never select a
    start. A failed attempt and its full evaluation trace remain in the record.
    ``initial_constants`` is an optional list, one dictionary per start, useful
    for profile warm starts. Its ell is ignored when ``fixed_ell`` is supplied.
    """
    started = time.perf_counter()
    centres, radii = _geometry_arrays(centres, radii)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or y.shape != (len(x),):
        raise ValueError("x must be (N, 3), y (N,)")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("finite fitting observations with labels in [0, 1] required")
    if any(int(v) != v or v < 1 for v in (knots, maxiter, max_evaluations, rounds)):
        raise ValueError("knots and iteration/evaluation/round budgets must be positive integers")
    if not np.isfinite(width) or width <= 0 or not np.isfinite(gtol) or gtol <= 0:
        raise ValueError("width and gtol must be positive and finite")
    if backend not in ("feasible_lbfgs", "trust_constr"):
        raise ValueError("backend must be feasible_lbfgs or trust_constr")
    check_points = x if validation_points is None else np.asarray(validation_points, dtype=float)
    if check_points.ndim != 2 or check_points.shape[1] != 3 or not np.isfinite(check_points).all():
        raise ValueError("validation_points must be finite and (N, 3)")
    starts = tuple(tuple(start) for start in starts)
    if not starts or any(len(start) != 2 for start in starts):
        raise ValueError("starts must contain (ell, xi) pairs")
    if any(value is not None and (not np.isfinite(value) or value < 0 or (axis == 1 and value == 0))
           for start in starts for axis, value in enumerate(start)):
        raise ValueError("ell starts must be nonnegative; xi starts positive")
    if initial_constants is not None and len(initial_constants) != len(starts):
        raise ValueError("initial_constants must have one dictionary per start")
    ids = _candidate_ids(candidate_ids, len(radii))
    prepared, preparation = prepare_feasible_geometry(centres, radii, candidate_ids=ids, box=box,
                                                       min_gap=min_gap, radius_bounds=radius_bounds)
    record = {"model_version": MODEL_VERSION, "status": "fit_failed", "reliable": False,
              "candidate_ids": ids, "particles": len(radii), "fitting_atoms": len(x),
              "preparation": preparation, "starts": [], "selected_start": None,
              "geometry_feasible": False, "sampled_probability_valid": False,
              "converged": False, "assignment_stable": False, "identified": None,
              "identifiability": "not_assessed", "ambiguous": False, "at_bounds": [],
              "constants": None, "centres": None, "radii": None, "shape": None,
              "train_loss": None, "summed_nll": None, "flags": [],
              "configuration": {"box": box, "min_gap": min_gap, "radius_bounds": radius_bounds,
                                "width": width, "knots": knots, "starts": starts,
                                "maxiter_per_round": maxiter, "max_evaluations_per_start": max_evaluations,
                                "rounds": rounds, "fixed_ell": fixed_ell, "gtol": gtol,
                                "backend": backend,
                                "backend_version": BACKEND_VERSION if backend == "feasible_lbfgs" else "trust_constr_v1",
                                "numerical_interior_slack": NUMERICAL_INTERIOR_SLACK,
                                "probability_margin": PROBABILITY_MARGIN,
                                "fixed_domain_probe_count": 125, "surface_probes_per_sphere": len(surface_directions()),
                                "selection": "minimum fitting NLL among feasible converged assignment-stable attempts",
                                "ambiguity_nll_tolerance": 1.0,
                                "model_change": "constant own-source continuation inside spheres"}}
    def finish(model, status, flags):
        record.update(status=status, flags=flags, seconds=time.perf_counter() - started)
        return model, _json_safe(record)
    if prepared is None:
        return finish(None, "invalid_geometry", [preparation.get("reason", "invalid_geometry")])
    if not len(x):
        return finish(None, "no_observations", ["no_observations"])
    radii = prepared
    tx, ty = torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)
    fixed_probes = _fixed_probes(box)
    mean = float(np.clip(y.mean(), 1e-5, 1 - 1e-5))
    converged = []
    for number, (ell, xi) in enumerate(starts):
        initial = {"c_eq": mean / 2, "ell": radii.mean() if ell is None else ell,
                   "xi": screening_length(radii, box ** 3) if xi is None else xi, "c_inf": mean}
        if initial_constants is not None:
            initial.update(initial_constants[number])
        if fixed_ell is not None:
            initial["ell"] = fixed_ell
        attempt = {"start": number, "status": "failed", "initial_constants": dict(initial),
                   "evaluations": [], "rounds": [], "converged": False, "assignment_stable": False,
                   "train_loss": None, "summed_nll": None}
        record["starts"].append(attempt)
        if progress_callback is not None:
            progress_callback(_json_safe({"event": "start_started", "start": number, "initial_constants": dict(initial)}))
        try:
            if any(not np.isfinite(value) or value < PHYSICAL_BOUNDS[name][0] or value > PHYSICAL_BOUNDS[name][1]
                   for name, value in initial.items()):
                raise DomainError("physical starting value outside declared search bounds")
            model = ConstrainedDiffusionModel(centres, radii, **initial, width=width, knots=knots,
                                              candidate_ids=ids, box=box, min_gap=min_gap,
                                              radius_bounds=radius_bounds, fixed_ell=fixed_ell)
            initial_points = torch.cat([tx, _probe_points(model, fixed_probes)])
            attempt["initial_feasibility_projection"] = _initial_probability_projection(model, initial_points)
            vector = model.vector.detach().numpy().copy()
            lower, upper = model.layout.bounds(box, radius_bounds)
            constraints = geometry_constraints(model.layout, box=box, min_gap=min_gap)
            for round_number in range(rounds):
                model.set_vector(vector)
                assignment = model.assign(tx)
                evaluator = _Evaluation(model, tx, ty, assignment, max_evaluations,
                                        attempt["evaluations"], fixed_probes)
                if backend == "trust_constr":
                    sampled = NonlinearConstraint(evaluator.constraints, 0.0, np.inf, jac=evaluator.constraint_jac,
                                                   hess=BFGS(), keep_feasible=True)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="delta_grad == 0.0.*", category=UserWarning)
                        result = minimize(evaluator.fun, vector, jac=evaluator.jac, hess=BFGS(),
                                          method="trust-constr", bounds=Bounds(lower, upper, keep_feasible=True),
                                          constraints=[*constraints, sampled],
                                          options={"maxiter": maxiter, "gtol": gtol, "xtol": 1e-10,
                                                   "barrier_tol": 1e-8, "sparse_jacobian": True,
                                                   "factorization_method": "AugmentedSystem"})
                else:
                    callback = (None if progress_callback is None else
                                lambda event: progress_callback({"start": number, "round": round_number, **event}))
                    result = _minimize_feasible(evaluator, vector, constraints, lower, upper,
                                                maxiter=maxiter, gtol=gtol, progress_callback=callback)
                vector = result.x.copy()
                old = evaluator.evaluate(vector)
                model.set_vector(vector)
                reassigned = model.assign(tx)
                changed = int((reassigned != assignment).sum())
                # Re-evaluate the full likelihood and derivative with current
                # assignments. Far-away assignment changes may be numerically
                # immaterial, but this must be demonstrated, not assumed.
                refreshed = _Evaluation(model, tx, ty, reassigned, max_evaluations,
                                         attempt["evaluations"], fixed_probes)
                new = refreshed.evaluate(vector)
                loss_change = abs(new[0] - old[0])
                gradient_change = float(np.max(np.abs(new[1] - old[1])))
                stable = changed == 0 or (loss_change <= 1e-10 and gradient_change <= gtol * 0.1)
                optimizer_converged = bool(result.success and result.optimality <= gtol
                                           and result.constr_violation <= 1e-8)
                final_matrix, final_slack = _linearized_constraints(vector, new, constraints, lower, upper)
                final_direction, final_multipliers, final_projection = _project_direction(-new[1], final_matrix, final_slack)
                reassigned_stationarity = float(np.max(np.abs(final_direction)))
                reassigned_complementarity = float(np.max(np.abs(final_multipliers * final_slack)))
                final_optimality = max(reassigned_stationarity, reassigned_complementarity)
                numerical = optimizer_converged and final_optimality <= gtol
                round_record = {"round": round_number, "iterations": int(result.nit),
                                "optimizer_status": int(result.status), "termination": str(result.message),
                                "optimizer_optimality": float(result.optimality), "optimality": final_optimality,
                                "constraint_violation": float(result.constr_violation),
                                "reassigned_stationarity": reassigned_stationarity,
                                "reassigned_complementarity": reassigned_complementarity,
                                "reassigned_projection_certificate": final_projection,
                                "numerically_converged": numerical, "assignment_changes": changed,
                                "assignment_objective_change": loss_change, "assignment_gradient_change": gradient_change,
                                "assignment_stable": stable, "train_loss": new[0], "summed_nll": new[0] * len(y)}
                if backend == "feasible_lbfgs":
                    round_record["iteration_history"] = result.history
                attempt["rounds"].append(round_record)
                attempt.update(train_loss=new[0], summed_nll=new[0] * len(y), assignment_stable=stable)
                if progress_callback is not None:
                    progress_callback(_json_safe({"event": "round_completed", "start": number, **round_record}))
                if numerical and stable:
                    attempt["converged"] = True
                    break
                if not optimizer_converged:
                    break
            model.set_vector(vector)
            final_points = torch.cat([tx, _probe_points(model, fixed_probes)]).detach().numpy()
            diagnostics = _final_diagnostics(model, final_points)
            attempt.update(constants=model.constants(), centres=model.centres().detach().numpy(),
                           radii=model.radii().detach().numpy(), shape=model.shape(), diagnostics=diagnostics,
                           at_bounds=_bound_contacts(model), status="converged" if attempt["converged"] else "not_converged")
            if attempt["converged"] and diagnostics["geometry"]["feasible"] and diagnostics["probability"]["passed"]:
                converged.append((number, model, attempt["summed_nll"]))
        except (DomainError, EvaluationBudgetExceeded, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            attempt.update(status="evaluation_budget" if isinstance(error, EvaluationBudgetExceeded) else "domain_or_backend_failure",
                           error=f"{type(error).__name__}: {error}")
        if progress_callback is not None:
            last_round = attempt["rounds"][-1] if attempt["rounds"] else {}
            progress_callback(_json_safe({"event": "start_completed", "start": number, "status": attempt["status"],
                                          "converged": attempt["converged"], "train_loss": attempt["train_loss"],
                                          **{key: last_round.get(key) for key in ("iterations", "optimality",
                                                                                "constraint_violation", "termination")}}))
    if not converged:
        return finish(None, "no_converged_start", ["no_feasible_converged_assignment_stable_start"])
    number, selected, nll = min(converged, key=lambda item: item[2])
    selected_attempt = record["starts"][number]
    diagnostics = _final_diagnostics(selected, check_points)
    alternatives = []
    for other_number, other, other_nll in converged:
        if other_number == number or other_nll - nll > 1.0:
            continue
        physical_difference = max(abs(other.constants()[key] - selected.constants()[key])
                                  / max(abs(selected.constants()[key]), 0.1)
                                  for key in PHYSICAL_BOUNDS)
        radius_difference = float(np.max(np.abs(other.radii().detach().numpy() - selected.radii().detach().numpy())))
        centre_difference = float(np.linalg.norm(other.centres().detach().numpy() - selected.centres().detach().numpy(), axis=1).max())
        if physical_difference > 0.1 or radius_difference > 0.1 or centre_difference > 0.2:
            alternatives.append({"start": other_number, "delta_nll": other_nll - nll,
                                 "maximum_relative_constant_difference": physical_difference,
                                 "maximum_radius_difference": radius_difference, "maximum_centre_difference": centre_difference})
    flags = []
    if alternatives:
        flags.append("multiple_materially_different_near_optima")
    if selected_attempt["at_bounds"]:
        flags.append("bound_contact")
    if not diagnostics["probability"]["passed"]:
        flags.append("invalid_probability_at_final_check_positions")
    if np.ptp(selected.radii().detach().numpy()) <= 1e-5 and fixed_ell is None:
        flags.append("equal_radius_ell_ceq_confounding")
    record.update(selected_start=number, constants=selected.constants(),
                  centres=selected.centres().detach().numpy(), radii=selected.radii().detach().numpy(),
                  shape=selected.shape(), train_loss=nll / len(y), summed_nll=nll,
                  geometry_feasible=True, sampled_probability_valid=diagnostics["probability"]["passed"],
                  converged=True, assignment_stable=True, ambiguous=bool(alternatives),
                  at_bounds=selected_attempt["at_bounds"], near_optimum_alternatives=alternatives,
                  fitting_diagnostics=selected_attempt["diagnostics"], final_diagnostics=diagnostics,
                  final_check_source="fitting_positions" if validation_points is None else "validation_positions",
                  reliable=not flags)
    return finish(selected, "ok" if not flags else "diagnostic_flags", flags)
