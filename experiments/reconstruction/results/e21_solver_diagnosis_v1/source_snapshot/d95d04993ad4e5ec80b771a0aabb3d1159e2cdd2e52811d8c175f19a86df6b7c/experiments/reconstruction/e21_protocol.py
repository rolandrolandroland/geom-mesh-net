"""Strict identities, provenance and release decisions for the E21 study.

No data or reserved outcomes are loaded here. Attempt 1 is always the primary
record, including errors; later attempts are diagnostic and cannot rescue a gate.
Every comparison uses the full prescribed pattern denominator.
"""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import scipy
import torch


ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = Path(__file__).with_name("e21_design.json")
ARMS = ("A", "B", "C", "D")
PHASES = ("development", "confirmation", "test")
SEED_STREAMS = ("parameters", "positions", "geometry", "labels", "thinning",
                "split", "network", "resampling")
NUMERIC_METRICS = ("radius_mare", "ell_absolute_relative_error",
                   "xi_absolute_relative_error", "full_heldback_log_loss",
                   "matrix_heldback_log_loss")
BOOLEAN_METRICS = ("feasible", "probability_valid", "converged",
                   "search_bound_contact", "ambiguous", "objective_contract_passed")


def json_safe(value):
    """Convert arrays/scalars, but refuse nonfinite values rather than hiding them."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite report value; use an explicit failed status and null diagnostic")
    return value


def digest(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_hash(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def validate_design(design):
    """The initial 12/24/24 study is explicit; another scope needs another protocol."""
    if not isinstance(design, dict):
        raise ValueError("design must be an object")
    for key in ("version", "dataset_id", "seed_namespace"):
        if not isinstance(design.get(key), str) or not design[key]:
            raise ValueError(f"missing {key}")
    scope = design.get("scope", {})
    for phase, count in zip(PHASES, (12, 24, 24)):
        ids = scope.get(phase)
        if not isinstance(ids, list) or len(ids) != count or any(
                type(i) is not int or i < 0 for i in ids) or len(set(ids)) != count:
            raise ValueError(f"{phase} needs {count} unique nonnegative integer identities")
    smoke = design.get("smoke_cases")
    if (not isinstance(smoke, list) or len(smoke) != 4 or len(set(smoke)) != 4
            or any(type(i) is not int or i not in scope["development"] for i in smoke)):
        raise ValueError("four distinct development smoke cases are required")
    if design.get("seed_streams") != list(SEED_STREAMS):
        raise ValueError("all independent seed streams are required in their declared order")
    if set(design.get("arms", {})) != set(ARMS):
        raise ValueError("exactly four arms A, B, C and D are required")
    if design.get("canonical_attempt") != 1 or type(design["canonical_attempt"]) is not int:
        raise ValueError("attempt 1, including errors, is the canonical attempt")
    if design.get("efficiency") != 0.37 or design.get("held_back_fraction") != 0.2:
        raise ValueError("the initial E21 protocol uses efficiency .37 and held-back fraction .2")
    runtime = design.get("runtime", {})
    if runtime.get("fit_device") != "cpu" or runtime.get("fit_dtype") != "float64":
        raise ValueError("physical fits require CPU float64")
    if runtime.get("neural_device") not in ("cpu", "mps"):
        raise ValueError("freeze an explicit cpu or mps neural device before cases")
    if type(runtime.get("threads")) is not int or runtime["threads"] < 1:
        raise ValueError("runtime threads must be a positive integer")
    physics = design.get("physics", {})
    if physics.get("births") is not False or physics.get("deletions") is not False:
        raise ValueError("E21 prohibits particle birth and deletion")
    rules = design.get("promotion", {})
    for name in ("require_all_C_D_feasible_converged", "require_no_higher_bound_frequency",
                 "require_no_higher_ambiguity_frequency"):
        if rules.get(name) is not True:
            raise ValueError(f"required promotion condition {name}")
    for name in ("minimum_radius_wins", "minimum_capillary_wins"):
        if type(rules.get(name)) is not int or not 1 <= rules[name] <= 24:
            raise ValueError(f"invalid {name}")
    for name in ("minimum_median_radius_reduction", "minimum_median_capillary_reduction",
                 "maximum_median_full_heldback_loss_change"):
        if not _finite_number(rules.get(name)):
            raise ValueError(f"invalid {name}")
    if min(rules["minimum_median_radius_reduction"], rules["minimum_median_capillary_reduction"]) < 0:
        raise ValueError("improvement thresholds cannot be negative")
    if rules["maximum_median_full_heldback_loss_change"] > 0:
        raise ValueError("held-back loss degradation cannot authorize promotion")
    return design


def load_design(path=DESIGN_PATH):
    return validate_design(json.loads(Path(path).read_text()))


def case_spec(phase, index, design):
    if phase not in PHASES or type(index) is not int or index not in design["scope"][phase]:
        raise ValueError("case is outside the frozen phase scope")
    seeds = {}
    for name in SEED_STREAMS:
        payload = [design["seed_namespace"], design["dataset_id"], phase, index, name]
        seeds[name] = int(digest(payload)[:8], 16)
    return {"case_id": f'{design["dataset_id"]}:{phase}:{index:03d}:{digest(seeds)[:12]}',
            "phase": phase, "case_index": index, "seeds": seeds}


def resolve_device(prefer_mps=True):
    """Resolve once, before provenance is saved; never fall back during a run."""
    return "mps" if prefer_mps and torch.backends.mps.is_available() else "cpu"


def source_hashes(root=ROOT):
    """Hash complete relevant source subtrees, including newly added helpers."""
    root = Path(root)
    sources = []
    for subtree in ("geom_mesh_net", "experiments/reconstruction"):
        sources.extend((root / subtree).rglob("*.py"))
        sources.extend((root / subtree).rglob("*.json"))
    # Results are outputs, not executable design inputs. Including them would
    # invalidate resume every time an atomic report was written.
    sources = [p for p in sources if not any(part in ("results", "__pycache__")
                                           for part in p.relative_to(root).parts)]
    if (root / "pyproject.toml").exists():
        sources.append(root / "pyproject.toml")
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(sources))}


def make_provenance(design, input_metadata=None, root=ROOT):
    validate_design(design)
    runtime = {"python": platform.python_version(), "numpy": np.__version__,
               "scipy": scipy.__version__, "torch": str(torch.__version__),
               "platform": platform.platform(), **deepcopy(design["runtime"]),
               "actual_torch_threads": torch.get_num_threads(),
               "actual_torch_interop_threads": torch.get_num_interop_threads(),
               "thread_environment": {name: os.environ.get(name) for name in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")}}
    manifest = {"schema": "e21-provenance-v1", "design": deepcopy(design),
                "design_id": digest(design), "sources": source_hashes(root),
                "runtime": runtime, "inputs": deepcopy(input_metadata or {}),
                "case_manifest": [case_spec(phase, i, design)
                                  for phase in PHASES for i in design["scope"][phase]]}
    return json_safe(manifest)


def new_report(phase, design, provenance):
    if phase not in PHASES:
        raise ValueError("unknown E21 phase")
    if provenance.get("design_id") != digest(design) or provenance.get("design") != design:
        raise ValueError("provenance design mismatch")
    return {"schema": "e21-report-v1", "phase": phase, "design_id": digest(design),
            "fingerprint": digest(provenance), "provenance": deepcopy(provenance), "rows": []}


def write_report(path, report):
    """Atomic replacement of a report; attempts themselves are append-only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        previous = json.loads(path.read_text())
        if any(previous.get(key) != report.get(key) for key in
               ("schema", "phase", "design_id", "fingerprint", "provenance")):
            raise RuntimeError("cannot replace an existing report's immutable identity")
        old_rows = previous.get("rows", [])
        if report.get("rows", [])[:len(old_rows)] != old_rows:
            raise RuntimeError("recorded attempts are immutable; append a diagnostic retry")
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_compatible(path, provenance):
    report = json.loads(Path(path).read_text())
    if report.get("provenance") != provenance or report.get("fingerprint") != digest(provenance):
        raise RuntimeError("source, design, runtime or inputs changed; use a new experiment directory")
    if report.get("design_id") != provenance.get("design_id"):
        raise RuntimeError("report design identity changed")
    return report


def allocate_attempt_dir(root, case_id, arm, attempt):
    if arm not in ARMS or type(attempt) is not int or attempt < 1:
        raise ValueError("invalid arm or attempt")
    # Digest avoids path traversal and keeps the original full identity in reports.
    path = Path(root) / digest(case_id)[:24] / arm / f"attempt_{attempt:03d}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _row_key(row):
    if not isinstance(row, dict):
        raise ValueError("row must be an object")
    if (not isinstance(row.get("case_id"), str) or row.get("arm") not in ARMS
            or type(row.get("attempt")) is not int or row["attempt"] < 1):
        raise ValueError("malformed case, arm or attempt identity")
    return row["case_id"], row["arm"], row["attempt"]


def record_attempt(report, row):
    key = _row_key(row)
    if key in {_row_key(existing) for existing in report["rows"]}:
        raise ValueError("attempt already recorded; retain it and use a new attempt number")
    report["rows"].append(deepcopy(json_safe(row)))


def canonical_rows(report):
    # Validate every row, including diagnostic retries, before selecting attempt 1.
    keys = [_row_key(row) for row in report.get("rows", [])]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate attempt identities")
    return [row for row in report.get("rows", []) if row["attempt"] == 1]


def _finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _accounting(report, design, expected_indices=None):
    errors = []
    phase = report.get("phase")
    if phase not in PHASES:
        return [], ["unknown phase"], 0
    indices = design["scope"][phase] if expected_indices is None else expected_indices
    expected = {(case_spec(phase, i, design)["case_id"], arm): i
                for i in indices for arm in ARMS}
    if report.get("design_id") != digest(design):
        errors.append("design identity mismatch")
    provenance = report.get("provenance")
    if (not isinstance(provenance, dict) or provenance.get("design") != design
            or provenance.get("design_id") != digest(design)
            or report.get("fingerprint") != digest(provenance)):
        errors.append("report provenance integrity mismatch")
    try:
        rows = canonical_rows(report)
    except (ValueError, TypeError, KeyError) as exc:
        return [], errors + [str(exc)], len(expected)
    seen = Counter()
    for row in report.get("rows", []):
        key = row["case_id"], row["arm"]
        if key not in expected:
            errors.append(f"unexpected case/arm {key}")
            continue
        if row.get("phase") != phase or type(row.get("case_index")) is not int or row["case_index"] != expected[key]:
            errors.append(f"case metadata mismatch {key}")
        if "seeds" in row and row["seeds"] != case_spec(phase, expected[key], design)["seeds"]:
            errors.append(f"case seed streams changed {key}")
        if row.get("status") not in ("ok", "error"):
            errors.append(f"nonterminal or invalid status {key}")
        if row.get("status") == "error" and not isinstance(row.get("error"), str):
            errors.append(f"error record needs an explicit reason {key}")
        for name in ("data_identity", "catalogue_identity"):
            if not isinstance(row.get(name), str) or not row[name]:
                errors.append(f"missing {name} {key}")
        if row.get("status") == "ok":
            metrics = row.get("metrics", {})
            if not isinstance(metrics, dict):
                errors.append(f"metrics must be an object {key}")
                continue
            if any(type(metrics.get(name)) is not bool for name in BOOLEAN_METRICS):
                errors.append(f"missing or malformed outcome flags {key}")
            if any(not _finite_number(metrics.get(name)) for name in NUMERIC_METRICS):
                errors.append(f"missing or nonfinite primary/secondary metrics {key}")
            if any(_finite_number(metrics.get(name)) and metrics[name] < 0 for name in NUMERIC_METRICS):
                errors.append(f"negative absolute error or log loss {key}")
        if row["attempt"] == 1:
            seen[key] += 1
    missing = sorted(set(expected) - set(seen))
    if missing:
        errors.append(f"missing {len(missing)} canonical case/arm records")
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    for case_id, group in by_case.items():
        for identity in ("data_identity", "catalogue_identity"):
            if len({digest(r.get(identity)) for r in group}) != 1:
                errors.append(f"{identity} differs across arms for {case_id}")
        matching = [r.get("matching_identity") for r in group]
        if any(value is not None for value in matching) and (
                any(not isinstance(value, str) or not value for value in matching)
                or len(set(matching)) != 1):
            errors.append(f"truth matching differs across arms for {case_id}")
    return rows, errors, len(expected)


def evaluate_gate(report, design):
    """Recompute the full physical gate. Never trust a saved ``passed`` flag."""
    validate_design(design)
    rows, errors, expected = _accounting(report, design)
    decision = {"complete": not errors, "passed": False, "geometry_useful": False,
                "physical_inference_useful": False, "retained_baseline": "B",
                "expected_records": expected, "canonical_records": len(rows),
                "failed_records": sum(row.get("status") == "error" for row in rows),
                "accounting_errors": errors, "criteria": {}, "summary": {}}
    if errors:
        return decision
    pairs = {index: {} for index in design["scope"][report["phase"]]}
    for row in rows:
        pairs[row["case_index"]][row["arm"]] = row
    candidates = [pair[arm] for pair in pairs.values() for arm in ("C", "D")]
    reliable = all(row["status"] == "ok" and all(row["metrics"][name] for name in
                   ("feasible", "probability_valid", "converged", "objective_contract_passed"))
                   for row in candidates)
    decision["criteria"]["all_C_D_numerically_valid"] = reliable
    if not reliable:
        return decision
    c = [pair["C"]["metrics"] for pair in pairs.values()]
    d = [pair["D"]["metrics"] for pair in pairs.values()]
    radius = [left["radius_mare"] - right["radius_mare"] for left, right in zip(c, d)]
    capillary = [left["ell_absolute_relative_error"] - right["ell_absolute_relative_error"]
                 for left, right in zip(c, d)]
    held = [right["full_heldback_log_loss"] - left["full_heldback_log_loss"]
            for left, right in zip(c, d)]
    summary = {"patterns": len(c), "radius_wins": sum(x > 0 for x in radius),
               "median_radius_reduction": float(np.median(radius)),
               "capillary_wins": sum(x > 0 for x in capillary),
               "median_capillary_reduction": float(np.median(capillary)),
               "median_full_heldback_loss_change": float(np.median(held)),
               "C_bound_count": sum(m["search_bound_contact"] for m in c),
               "D_bound_count": sum(m["search_bound_contact"] for m in d),
               "C_ambiguity_count": sum(m["ambiguous"] for m in c),
               "D_ambiguity_count": sum(m["ambiguous"] for m in d),
               "median_xi_error_change": float(np.median([right["xi_absolute_relative_error"]
                   - left["xi_absolute_relative_error"] for left, right in zip(c, d)])),
               "median_matrix_loss_change": float(np.median([right["matrix_heldback_log_loss"]
                   - left["matrix_heldback_log_loss"] for left, right in zip(c, d)]))}
    decision["summary"] = summary
    rules = design["promotion"]
    criteria = decision["criteria"]
    criteria.update({"radius_wins": summary["radius_wins"] >= rules["minimum_radius_wins"],
        "radius_size": summary["median_radius_reduction"] >= rules["minimum_median_radius_reduction"],
        "capillary_wins": summary["capillary_wins"] >= rules["minimum_capillary_wins"],
        "capillary_size": summary["median_capillary_reduction"] >= rules["minimum_median_capillary_reduction"],
        "no_more_bounds": summary["D_bound_count"] <= summary["C_bound_count"],
        "no_more_ambiguity": summary["D_ambiguity_count"] <= summary["C_ambiguity_count"],
        "heldback_no_worse": summary["median_full_heldback_loss_change"] <= rules["maximum_median_full_heldback_loss_change"]})
    eligible = report["phase"] in ("confirmation", "test")
    decision["geometry_useful"] = eligible and criteria["radius_wins"] and criteria["radius_size"]
    decision["passed"] = decision["physical_inference_useful"] = eligible and all(criteria.values())
    decision["retained_baseline"] = "D" if decision["passed"] else "B"
    if not eligible:
        decision["development_only"] = True
    return decision


def require_test_permission(confirmation, design, provenance=None):
    if confirmation.get("phase") != "confirmation":
        raise RuntimeError("only complete confirmation can release the reserved test")
    if provenance is not None and (confirmation.get("provenance") != provenance
            or confirmation.get("fingerprint") != digest(provenance)):
        raise RuntimeError("confirmation provenance mismatch")
    decision = evaluate_gate(confirmation, design)
    if not decision["passed"]:
        raise RuntimeError("full physical-inference confirmation gate failed; test remains unopened")
    return decision


def smoke_gate(report, design):
    """All four smoke cases and all corrected arms must be numerically ready."""
    if report.get("phase") != "development":
        return {"passed": False, "errors": ["smoke requires development"], "runtime": None}
    # The later eight development cases may be absent, but unexpected/malformed
    # records anywhere in this report must not disappear through smoke filtering.
    _, full_errors, _ = _accounting(report, design)
    full_errors = [error for error in full_errors if not (
        error.startswith("missing ") and error.endswith(" canonical case/arm records"))]
    subset = deepcopy(report)
    subset["rows"] = [row for row in report.get("rows", [])
                      if isinstance(row, dict) and row.get("case_index") in design["smoke_cases"]]
    rows, errors, _ = _accounting(subset, design, design["smoke_cases"])
    errors = sorted(set(errors + full_errors))
    corrected = [row for row in rows if row["arm"] in ("B", "C", "D")]
    numerical = len(corrected) == 12 and all(row.get("status") == "ok" and all(
        row.get("metrics", {}).get(name) is True for name in
        ("feasible", "probability_valid", "converged", "objective_contract_passed"))
        for row in corrected)
    elapsed = [row.get("elapsed_seconds") for row in rows]
    cost_valid = len(elapsed) == 16 and all(_finite_number(x) and x > 0 for x in elapsed)
    cost = {"base_fits": len(rows), "sum_seconds": sum(elapsed),
            "median_seconds": float(np.median(elapsed))} if cost_valid else None
    return {"passed": not errors and numerical and cost_valid,
            "errors": errors, "numerically_ready": numerical, "runtime_recorded": cost_valid,
            "runtime": cost, "promotion": False}
