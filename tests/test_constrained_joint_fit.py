"""Physical-domain and numerical-contract tests for the versioned sphere fitter."""

import json

import numpy as np
import pytest
import torch
from scipy.optimize._numdiff import approx_derivative

from geom_mesh_net.fields import constrained_joint_fit as cj
from geom_mesh_net.fields import joint_fit, physics


@pytest.fixture(autouse=True)
def limited_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(3)
    yield
    torch.set_num_threads(before)


def model(**options):
    values = dict(c_eq=0.03, ell=2.0, xi=5.0, c_inf=0.1, box=35.0)
    values.update(options)
    return cj.ConstrainedDiffusionModel([[10., 10., 10.], [22., 10., 10.]], [2., 3.], **values)


def test_projection_preserves_ids_and_centres_and_minimizes_radius_change():
    centres = np.array([[5., 5., 5.], [9., 5., 5.]])
    original = centres.copy()
    radii, record = cj.prepare_feasible_geometry(centres, [3., 3.], candidate_ids=["left", "right"], box=15.)
    np.testing.assert_array_equal(centres, original)
    np.testing.assert_allclose(radii, (4 - 1e-5) / 2, atol=1e-8)
    assert record["status"] == "projected"
    assert record["candidate_ids"] == ["left", "right"]
    assert record["after"]["feasible"]
    assert len(record["changes"]) == 2


def test_feasible_geometry_is_not_silently_shrunk():
    radii, record = cj.prepare_feasible_geometry([[5., 5., 5.], [9., 5., 5.]], [1., 2.], box=15.)
    np.testing.assert_array_equal(radii, [1., 2.])
    assert record["status"] == "unchanged"


@pytest.mark.parametrize("centres", [[[0.3, 5., 5.]], [[5., 5., 5.], [5.8, 5., 5.]]])
def test_impossible_preserved_centres_fail_without_dropping_candidates(centres):
    radii, record = cj.prepare_feasible_geometry(centres, np.ones(len(centres)), box=15.)
    assert radii is None
    assert record["reason"] == "no_feasible_radii_at_preserved_centres"
    assert record["candidate_count_unchanged"]
    assert len(record["candidate_ids"]) == len(centres)


def test_radius_projection_respects_complete_spheres_at_box_boundary():
    radii, record = cj.prepare_feasible_geometry([[1., 5., 5.]], [2.], box=15.)
    np.testing.assert_allclose(radii, [1.])
    assert record["after"]["minimum_box_clearance"] >= -cj.DOMAIN_TOLERANCE


def test_duplicate_candidate_ids_are_refused():
    with pytest.raises(ValueError, match="unique ID"):
        cj.prepare_feasible_geometry([[5., 5., 5.], [9., 5., 5.]], [1., 1.], candidate_ids=["a", "a"])


def test_attempted_overlap_is_refused_before_any_coupling_solve(monkeypatch):
    m = model()
    vector = m.vector.numpy().copy()
    vector[6:8] = np.array([7., 7.]) / cj.GEOMETRY_SCALE
    m.set_vector(vector)
    called = []
    monkeypatch.setattr(torch.linalg, "solve", lambda *args: called.append(True))
    with pytest.raises(cj.DomainError, match="before coupling solve"):
        m.amplitudes()
    assert called == []


def test_nearly_touching_spheres_remain_finite_and_valid():
    m = cj.ConstrainedDiffusionModel([[10., 10., 10.], [14.00002, 10., 10.]], [2., 2.],
                                     ell=0., c_eq=.1, c_inf=.1, box=30.)
    assert m.guard_geometry()["feasible"]
    assert torch.isfinite(m.amplitudes()).all()
    np.testing.assert_allclose(m.raw_matrix([[12., 10., 10.]]), [.1], atol=1e-14)


def test_exterior_matrix_and_coupling_agree_with_archived_equations():
    m = model()
    centres, radii = m.centres().numpy(), m.radii().numpy()
    field = physics.solve_field(centres, radii, **m.constants())
    np.testing.assert_allclose(m.amplitudes().numpy(), field.amplitudes, atol=1e-14)
    rng = np.random.default_rng(9)
    points = rng.uniform(0, 35, (500, 3))
    outside = np.all(np.linalg.norm(points[:, None] - centres[None], axis=-1) >= radii[None], axis=1)
    np.testing.assert_allclose(m.raw_matrix(points[outside]), field(points[outside]), atol=1e-13)
    old = joint_fit.JointDiffusionModel(centres, radii, **m.constants(), width=.3, knots=6,
                                       levels=m.shape()["levels"])
    np.testing.assert_allclose(m.raw_predict(points[outside]), joint_fit.predict(old, points[outside]), atol=1e-13)


def test_interior_source_extension_is_finite_at_centres_and_explicitly_versioned():
    m = model()
    values = m.raw_predict(m.centres().numpy())
    assert np.isfinite(values).all() and np.all((values > 0) & (values < 1))
    assert m.model_version == "screened_spheres_constant_interior_v1"
    # Interior continuation changes only each source's own singular term.
    assert np.isfinite(m.raw_matrix(m.centres().numpy())).all()


def test_fixed_zero_capillarity_is_exact_and_removed_from_active_vector():
    free = model(ell=0.)
    null = model(ell=9., fixed_ell=0.)
    assert null.constants()["ell"] == 0.
    assert free.layout.size == null.layout.size + 1
    np.testing.assert_allclose(free.raw_predict([[12., 10., 10.]]), null.raw_predict([[12., 10., 10.]]))


def test_positive_log_parameters_retain_derivatives_without_parameter_clipping():
    m = model()
    vector = m.vector.clone().requires_grad_(True)
    m.set_vector(vector)
    derivative = torch.autograd.grad(m.value("xi"), vector)[0]
    assert derivative[m.layout.chemistry + 1] == pytest.approx(5.)


def test_ill_conditioned_coupling_is_refused_before_solve(monkeypatch):
    m = model()
    monkeypatch.setattr(torch.linalg, "cond", lambda *args: torch.tensor(1e12))
    called = []
    monkeypatch.setattr(torch.linalg, "solve", lambda *args: called.append(True))
    with pytest.raises(cj.DomainError, match="ill-conditioned"):
        m.amplitudes()
    assert not called


def test_materially_invalid_probabilities_never_receive_likelihood():
    m = model(c_eq=.9, ell=4.)
    points = torch.tensor([[12., 10., 10.], [15., 10., 10.]], dtype=torch.float64)
    trace = []
    evaluator = cj._Evaluation(m, points, torch.ones(2, dtype=torch.float64), m.assign(points), 10,
                               trace, cj._fixed_probes(m.box))
    with pytest.raises(cj.DomainError, match="raw probability"):
        evaluator.fun(m.vector.numpy())
    assert len(trace) == 1 and trace[0]["geometry_feasible"]
    assert trace[0]["likelihood_evaluated"] is False
    assert trace[0]["mean_nll"] is None


def test_cached_evaluation_restores_model_after_rejected_trial():
    m = model()
    points = torch.tensor([[12., 10., 10.], [15., 10., 10.]], dtype=torch.float64)
    evaluator = cj._Evaluation(m, points, torch.ones(2, dtype=torch.float64), m.assign(points), 10,
                               [], cj._fixed_probes(m.box))
    valid = m.vector.numpy().copy()
    original = evaluator.evaluate(valid)
    invalid = valid.copy()
    invalid[6:8] = 8. / cj.GEOMETRY_SCALE
    with pytest.raises(cj.DomainError):
        evaluator.evaluate(invalid)
    repeated = evaluator.evaluate(valid)
    assert repeated[0] == original[0]
    np.testing.assert_array_equal(m.vector.detach().numpy(), valid)


def test_likelihood_and_probability_jacobians_match_finite_differences():
    m = model()
    points = torch.tensor(np.random.default_rng(17).uniform(1, 34, (53, 3)), dtype=torch.float64)
    labels = torch.tensor(np.random.default_rng(18).uniform(.05, .95, len(points)), dtype=torch.float64)
    evaluator = cj._Evaluation(m, points, labels, m.assign(points), 1000, [], cj._fixed_probes(m.box))
    vector = m.vector.numpy().copy()
    actual = evaluator.evaluate(vector)
    finite_objective = approx_derivative(lambda v: evaluator.fun(v), vector, method="3-point", abs_step=1e-5).ravel()
    finite_constraints = approx_derivative(lambda v: evaluator.constraints(v), vector, method="3-point", abs_step=1e-5)
    np.testing.assert_allclose(actual[1], finite_objective, atol=2e-7, rtol=1e-4)
    np.testing.assert_allclose(actual[3].toarray(), finite_constraints, atol=2e-7, rtol=1e-4)


def test_pair_constraint_jacobian_matches_finite_differences():
    m = model()
    constraint = cj.geometry_constraints(m.layout, box=m.box, min_gap=m.min_gap)[1]
    vector = m.vector.numpy()
    finite = approx_derivative(constraint.fun, vector, method="3-point")
    np.testing.assert_allclose(constraint.jac(vector).toarray(), np.atleast_2d(finite), atol=1e-8)


def test_previously_overlapping_e20_geometry_regression_is_projected_without_merging():
    # Exact Fourier pair 53,54 in E20 confirmation cells[3].physics.fourier.
    # The archived result stays unchanged; this tests only feasible preparation.
    centres = np.array([[55.007388143219494, 37.019984994889356, 50.87616894376653],
                        [54.71587203665982, 27.177252710552626, 51.35122692371706]])
    radii = np.array([5.739324483298389, 4.120455011208602])
    before = cj.geometry_diagnostics(centres, radii)
    prepared, record = cj.prepare_feasible_geometry(centres, radii, candidate_ids=["53", "54"])
    assert before["minimum_surface_gap"] == pytest.approx(-0.0012785665858023876)
    assert len(prepared) == 2 and record["after"]["feasible"]


def test_bound_contact_and_identification_are_separate():
    m = model(ell=0., xi=.5)
    contacts = cj._bound_contacts(m)
    assert {r["parameter"]: r["kind"] for r in contacts} == {
        "ell": "physical_null_boundary", "xi": "search_bound"}


def test_small_model_matched_fit_is_valid_and_all_trials_are_auditable():
    truth = model()
    points = np.random.default_rng(12).uniform(0, 35, (1400, 3))
    labels = truth.raw_predict(points)
    fitted, record = cj.fit_constrained(truth.centres().numpy(), [2.1, 2.85], points, labels,
                                        candidate_ids=["a", "b"], box=35., starts=((2., 5.),),
                                        maxiter=120, max_evaluations=250, gtol=1e-4)
    assert fitted is not None, record["starts"]
    assert record["converged"] and record["geometry_feasible"] and record["sampled_probability_valid"]
    assert record["assignment_stable"]
    assert record["identified"] is None
    assert record["candidate_ids"] == ["a", "b"]
    for trace in record["starts"][0]["evaluations"]:
        if trace["likelihood_evaluated"]:
            assert trace["geometry_feasible"] and trace["probability_valid"]
            assert trace["minimum_surface_gap"] >= cj.DOMAIN_TOLERANCE
    assert record["summed_nll"] == pytest.approx(record["train_loss"] * len(labels))
    json.dumps(record, allow_nan=False)


def test_iteration_cap_is_not_reported_as_convergence():
    m = model()
    points = np.random.default_rng(13).uniform(0, 35, (100, 3))
    fitted, record = cj.fit_constrained(m.centres().numpy(), m.radii().numpy(), points, m.predict(points),
                                        box=35., starts=((2., 5.),), maxiter=1, max_evaluations=20)
    assert fitted is None
    assert record["status"] == "no_converged_start"
    assert not record["converged"]
    assert record["starts"][0]["status"] == "not_converged"


def test_adversarial_backend_overlap_is_logged_and_never_scored(monkeypatch):
    m = model()
    points = np.random.default_rng(14).uniform(0, 35, (100, 3))
    def overlap_trial(fun, vector, **kwargs):
        invalid = vector.copy()
        invalid[6:8] = 8. / cj.GEOMETRY_SCALE
        return fun(invalid)
    monkeypatch.setattr(cj, "minimize", overlap_trial)
    fitted, record = cj.fit_constrained(m.centres().numpy(), m.radii().numpy(), points, m.predict(points),
                                        box=35., starts=((2., 5.),), maxiter=30, max_evaluations=20,
                                        backend="trust_constr")
    assert fitted is None
    attempt = record["starts"][0]
    assert attempt["status"] == "domain_or_backend_failure"
    trial = attempt["evaluations"][-1]
    assert trial["minimum_surface_gap"] == pytest.approx(-4.)
    assert not trial["geometry_feasible"] and not trial["likelihood_evaluated"]
    assert trial["mean_nll"] is None


def test_evaluation_budget_has_distinct_failure_status():
    m = model()
    points = np.random.default_rng(14).uniform(0, 35, (100, 3))
    fitted, record = cj.fit_constrained(m.centres().numpy(), m.radii().numpy(), points, m.predict(points),
                                        box=35., starts=((2., 5.),), maxiter=30, max_evaluations=1)
    assert fitted is None
    assert record["starts"][0]["status"] == "evaluation_budget"


def test_validation_positions_do_not_change_start_selection():
    m = model()
    points = np.random.default_rng(12).uniform(0, 35, (1400, 3))
    labels = m.predict(points)
    options = dict(box=35., starts=((2., 5.),), maxiter=120, max_evaluations=250, gtol=1e-4)
    _, one = cj.fit_constrained(m.centres().numpy(), [2.1, 2.85], points, labels, **options)
    _, two = cj.fit_constrained(m.centres().numpy(), [2.1, 2.85], points, labels,
                               validation_points=np.array([[0., 0., 0.], [35., 35., 35.]]), **options)
    assert one["selected_start"] == two["selected_start"] == 0
    assert one["train_loss"] == two["train_loss"]
    assert one["constants"] == two["constants"]


def test_direction_projection_has_a_checkable_kkt_certificate():
    from scipy import sparse
    # Projection of (2,2) onto x+y<=1 is (0.5,0.5).
    matrix = sparse.csr_matrix([[-1., -1.]])
    direction, multipliers, report = cj._project_direction(np.array([2., 2.]), matrix, np.array([1.]))
    np.testing.assert_allclose(direction, [.5, .5], atol=1e-9)
    np.testing.assert_allclose(multipliers, [1.5], atol=1e-9)
    assert report["converged"] and report["primal_violation"] <= 1e-9
    assert report["complementarity"] <= 1e-9


def test_active_set_refinement_repairs_dual_precision_without_relaxing_certificate():
    from scipy import sparse
    from scipy.optimize import LinearConstraint, minimize
    rng = np.random.default_rng(908)
    refined = 0
    for case in range(30):
        matrix = rng.normal(size=(16, 4))
        matrix /= np.linalg.norm(matrix, axis=1)[:, None]
        slack = rng.uniform(0, .5, 16)
        slack[:4] = 0 if case % 2 else .01
        desired = rng.normal(size=4) * 3
        direction, _, report = cj._project_direction(desired, sparse.csr_matrix(matrix), slack)
        reference = minimize(lambda p: .5 * np.sum((p - desired) ** 2), np.zeros(4),
                             jac=lambda p: p - desired, method="SLSQP",
                             constraints=[LinearConstraint(matrix, -slack, np.inf)],
                             options={"ftol": 1e-13, "maxiter": 200})
        assert reference.success and report["converged"]
        np.testing.assert_allclose(direction, reference.x, atol=3e-9)
        assert report["primal_violation"] <= 1e-9
        assert report["complementarity"] <= 1e-8
        refined += report.get("active_set_iterations", 0) > 0
    assert refined > 0


def test_reassigned_objective_requires_its_own_stationarity_certificate(monkeypatch):
    from scipy.optimize import OptimizeResult
    m = model()
    points = np.random.default_rng(12).uniform(0, 35, (1400, 3))
    def stale_optimizer_certificate(evaluator, vector, *args, **kwargs):
        evaluated = evaluator.evaluate(vector)
        return OptimizeResult(x=vector.copy(), fun=evaluated[0], jac=evaluated[1], success=True,
                              status=1, message="stale claimed convergence", nit=1,
                              optimality=0., constr_violation=0., history=[])
    monkeypatch.setattr(cj, "_minimize_feasible", stale_optimizer_certificate)
    fitted, record = cj.fit_constrained(m.centres().numpy(), [2.1, 2.85], points, m.predict(points),
                                        box=35., starts=((2., 5.),), rounds=1, gtol=1e-5)
    assert fitted is None and not record["converged"]
    round_record = record["starts"][0]["rounds"][0]
    assert round_record["optimizer_optimality"] == 0.
    assert round_record["optimality"] > 1e-5


def test_direction_projection_handles_redundant_nearly_duplicate_active_rows():
    from scipy import sparse
    from scipy.optimize import LinearConstraint, minimize
    rng = np.random.default_rng(908)
    for case in range(30):
        matrix = rng.normal(size=(16, 4))
        matrix /= np.linalg.norm(matrix, axis=1)[:, None]
        slack = rng.uniform(0, .5, 16)
        slack[:4] = 0 if case % 2 else .01
        desired = rng.normal(size=4) * 3
        desired /= max(1., np.max(np.abs(desired)))
        matrix = np.vstack([matrix, matrix[:4], matrix[:4] * (1 + 1e-12)])
        slack = np.r_[slack, slack[:4], slack[:4]]
        direction, _, report = cj._project_direction(desired, sparse.csr_matrix(matrix), slack)
        reference = minimize(lambda p: .5 * np.sum((p - desired) ** 2), np.zeros(4),
                             jac=lambda p: p - desired, method="SLSQP",
                             constraints=[LinearConstraint(matrix, -slack, np.inf)],
                             options={"ftol": 1e-13, "maxiter": 200})
        assert report["converged"] and reference.success
        np.testing.assert_allclose(direction, reference.x, atol=3e-9)


class _CurvedDomainEvaluator:
    """A nonlinear feasible domain whose tangent permits prohibited trial steps."""
    def __init__(self, stuck=False):
        self.trace = []
        self.model = self
        self.stuck = stuck

    def set_vector(self, vector):
        self.vector = vector

    def evaluate(self, vector, *, derivatives=True):
        from scipy import sparse
        x = float(vector[0])
        valid = x * x <= 1 + cj.DOMAIN_TOLERANCE and (not self.stuck or x == .1)
        self.trace.append({"x": x, "valid": valid, "likelihood_evaluated": valid})
        if not valid:
            raise cj.DomainError("prohibited raw-probability trial")
        return (float((x - 2) ** 2), np.array([2 * (x - 2)]), np.array([1 - x * x]),
                sparse.csr_matrix([[-2 * x]]))


def test_feasible_backend_backtracks_invalid_trials_without_assigning_a_loss():
    evaluator = _CurvedDomainEvaluator()
    result = cj._minimize_feasible(evaluator, np.array([.1]), [], np.array([-2.]), np.array([2.]),
                                   maxiter=60, gtol=1e-5)
    assert result.success, result.message
    assert result.x[0] == pytest.approx(1., abs=1e-5)
    assert any(not trial["valid"] for trial in evaluator.trace)
    assert all(trial["valid"] for trial in evaluator.trace if trial["likelihood_evaluated"])
    assert result.optimality <= 1e-5


def test_line_search_stall_with_nonzero_stationarity_is_not_convergence():
    evaluator = _CurvedDomainEvaluator(stuck=True)
    result = cj._minimize_feasible(evaluator, np.array([.1]), [], np.array([-2.]), np.array([2.]),
                                   maxiter=60, gtol=1e-5)
    assert not result.success
    assert result.status == 2 and result.optimality > 1e-5
    assert "stalled" in result.message


def test_projected_geometry_has_numerical_interior_clearance_for_solver():
    radii, record = cj.prepare_feasible_geometry([[5., 5., 5.], [9., 5., 5.]], [3., 3.], box=12.)
    assert record["after"]["minimum_surface_gap"] > 1e-5
    assert record["numerical_interior_slack"] == 1e-8
    assert len(radii) == 2


from scipy.optimize import OptimizeResult


def _round_budget_problem():
    model = cj.ConstrainedDiffusionModel([[10., 10., 10.], [22., 10., 10.]], [2., 3.],
                                         c_eq=.03, ell=2., xi=5., c_inf=.1, box=35.)
    points = np.random.default_rng(49).uniform(0, 35, (90, 3))
    labels = model.raw_predict(points)
    kwargs = {"box": 35., "starts": ((2., 5.),), "initial_constants": [model.constants()],
              "maxiter": 2, "max_evaluations": 20, "rounds": 3, "gtol": 1e-5}
    return model, points, labels, kwargs


def _controlled_rounds(monkeypatch, statuses):
    calls = []
    def optimizer(evaluator, vector, constraints, lower, upper, *, maxiter, gtol, progress_callback=None):
        status = statuses[min(len(calls), len(statuses) - 1)]
        calls.append(vector.copy())
        evaluated = evaluator.evaluate(vector)
        return OptimizeResult(x=vector.copy(), fun=evaluated[0], jac=evaluated[1],
                              success=status == 1, status=status, message=f"controlled status {status}",
                              nit=maxiter, optimality=0. if status == 1 else .01,
                              constr_violation=0., history=[{"iteration": 0, "mean_nll": evaluated[0]}])
    monkeypatch.setattr(cj, "_minimize_feasible", optimizer)
    return calls


def test_safe_iteration_cap_uses_next_declared_round_and_can_then_converge(monkeypatch):
    model, points, labels, kwargs = _round_budget_problem()
    calls = _controlled_rounds(monkeypatch, [0, 1])
    fitted, record = cj.fit_constrained(model.centres().numpy(), model.radii().numpy(), points, labels, **kwargs)
    assert fitted is not None and record["converged"]
    assert len(calls) == 2
    rounds = record["starts"][0]["rounds"]
    assert rounds[0]["continuation_decision"] == "continue_safe_iteration_cap"
    assert rounds[0]["numerically_converged"] is False
    assert rounds[1]["continuation_decision"] == "converged"
    assert rounds[1]["optimality"] <= kwargs["gtol"]


@pytest.mark.parametrize("status", [2, 3])
def test_stalled_or_no_descent_result_does_not_receive_another_round(monkeypatch, status):
    model, points, labels, kwargs = _round_budget_problem()
    calls = _controlled_rounds(monkeypatch, [status, 1])
    fitted, record = cj.fit_constrained(model.centres().numpy(), model.radii().numpy(), points, labels, **kwargs)
    assert fitted is None and not record["converged"]
    assert len(calls) == 1
    assert record["starts"][0]["termination_reason"] == "terminal_backend_nonconvergence"


def test_round_budget_exhaustion_remains_nonconvergence(monkeypatch):
    model, points, labels, kwargs = _round_budget_problem()
    kwargs["rounds"] = 2
    calls = _controlled_rounds(monkeypatch, [0])
    fitted, record = cj.fit_constrained(model.centres().numpy(), model.radii().numpy(), points, labels, **kwargs)
    assert fitted is None and not record["converged"]
    assert len(calls) == 2
    attempt = record["starts"][0]
    assert attempt["termination_reason"] == "round_budget_exhausted"
    assert attempt["status"] == "not_converged"
    assert all(not item["numerically_converged"] for item in attempt["rounds"])


def test_exhausted_total_evaluation_budget_prevents_continuation(monkeypatch):
    model, points, labels, kwargs = _round_budget_problem()
    kwargs["max_evaluations"] = 2
    calls = _controlled_rounds(monkeypatch, [0, 1])
    fitted, record = cj.fit_constrained(model.centres().numpy(), model.radii().numpy(), points, labels, **kwargs)
    assert fitted is None and len(calls) == 1
    attempt = record["starts"][0]
    assert len(attempt["evaluations"]) == 2
    assert attempt["termination_reason"] == "evaluation_budget_exhausted"
    assert attempt["status"] == "evaluation_budget"


def test_domain_failure_is_terminal_despite_remaining_rounds(monkeypatch):
    model, points, labels, kwargs = _round_budget_problem()
    calls = []
    def failure(*args, **options):
        calls.append(True)
        raise cj.DomainError("controlled failed projection")
    monkeypatch.setattr(cj, "_minimize_feasible", failure)
    fitted, record = cj.fit_constrained(model.centres().numpy(), model.radii().numpy(), points, labels, **kwargs)
    assert fitted is None and len(calls) == 1
    assert record["starts"][0]["status"] == "domain_or_backend_failure"


def test_live_progress_omits_history_but_saved_report_keeps_it(monkeypatch):
    model, points, labels, kwargs = _round_budget_problem()
    _controlled_rounds(monkeypatch, [0, 1])
    events = []
    _, record = cj.fit_constrained(model.centres().numpy(), model.radii().numpy(), points, labels,
                                   progress_callback=events.append, **kwargs)
    assert all("iteration_history" not in event for event in events)
    assert all(item["iteration_history"] for item in record["starts"][0]["rounds"])
    assert [event["continuation_decision"] for event in events if event["event"] == "round_completed"] == [
        "continue_safe_iteration_cap", "converged"]
    json.dumps(events, allow_nan=False)
    assert record["configuration"]["backend_version"] == "feasible_lbfgs_v3"


def _compact_fixture(kind, count):
    options=dict(c_eq=.03,ell=2.,xi=5.,c_inf=.1,box=35.)
    if kind=='surface_maximum':
        options.update(c_eq=.15,c_inf=.03,levels=np.linspace(.3,.01,7))
    if kind=='surface_minimum':
        options.update(c_eq=.03,ell=0.,c_inf=.1)
    if kind=='flat_ties':
        options.update(c_eq=.1,ell=0.,c_inf=.1)
    model=cj.ConstrainedDiffusionModel([[10.,10.,10.],[22.,10.,10.]],[2.,3.],**options)
    points=torch.tensor(np.random.default_rng(719).uniform(1,34,(count,3)),dtype=torch.float64)
    labels=torch.tensor(np.random.default_rng(720).uniform(.1,.9,count),dtype=torch.float64)
    return model,points,labels


@pytest.mark.parametrize('kind,count,stale_assignment',[
    ('ordinary',20,False),('ordinary',400,False),('ordinary',400,True),
    ('surface_maximum',400,False),('surface_minimum',400,False),('flat_ties',400,False)])
def test_compact_probability_jacobians_match_original_full_graph(kind,count,stale_assignment):
    model,points,labels=_compact_fixture(kind,count)
    assignment=model.assign(points)
    if stale_assignment:
        assignment=1-assignment
    evaluator=cj._Evaluation(model,points,labels,assignment,50,[],cj._fixed_probes(model.box))
    vector=model.vector.numpy().copy()
    values=evaluator.evaluate(vector,derivatives=False)
    # The original method differentiates these full-data expressions. This
    # reference is independent of compact point selection/recomputation.
    reference_gradient=torch.autograd.grad(evaluator.cached_loss,evaluator.cached_parameter,
                                           retain_graph=True)[0].detach().numpy()
    reference_jacobian=np.vstack([torch.autograd.grad(item,evaluator.cached_parameter,
                                   retain_graph=True)[0].detach().numpy()
                                   for item in evaluator.cached_constraints])
    if kind in ('surface_maximum','surface_minimum'):
        moving_start=len(points)+len(evaluator.fixed_probes)
        assert any(bool((indices>=moving_start).any()) for indices in evaluator.cached_extrema_origins)
    result=evaluator.evaluate(vector)
    np.testing.assert_array_equal(result[2],values[2])
    np.testing.assert_array_equal(result[1],reference_gradient)
    np.testing.assert_allclose(result[3].toarray(),reference_jacobian,atol=2e-11,rtol=1e-9)
    repeated=evaluator.evaluate(vector)
    np.testing.assert_array_equal(repeated[3].toarray(),result[3].toarray())
    assert evaluator.cached_loss is None and evaluator.cached_constraints is None


def test_large_exact_ties_reuse_full_graph_without_resampling_the_tied_atoms(monkeypatch):
    model,points,labels=_compact_fixture('flat_ties',400)
    evaluator=cj._Evaluation(model,points,labels,model.assign(points),50,[],cj._fixed_probes(model.box))
    vector=model.vector.numpy().copy()
    evaluator.evaluate(vector,derivatives=False)
    counts=[len(indices) for indices in evaluator.cached_extrema_origins]
    assert counts[0]>256 and counts[2]>256 and counts[3]>256
    # Matrix ties need an average derivative, not the first tied point. Capture
    # the full expression's derivative before the compact method frees it.
    expected=np.vstack([torch.autograd.grad(evaluator.cached_constraints[index],evaluator.cached_parameter,
                             retain_graph=True)[0].detach().numpy() for index in (0,2,3)])
    raw=model._raw
    recomputed_counts=[]
    def track(points,*args,**kwargs):
        recomputed_counts.append(len(points))
        return raw(points,*args,**kwargs)
    monkeypatch.setattr(model,'_raw',track)
    result=evaluator.evaluate(vector)
    np.testing.assert_array_equal(result[3].toarray()[[0,2,3]],expected)
    assert recomputed_counts and max(recomputed_counts)<=256


def test_compact_jacobians_preserve_successful_multiple_iteration_fit(monkeypatch):
    from scipy import sparse

    centres = np.array([[10., 10., 10.], [22., 10., 10.]])
    radii = np.array([2., 3.])
    truth = cj.ConstrainedDiffusionModel(centres, radii, c_eq=.03, ell=2., xi=5., c_inf=.1, box=35.)
    points = np.random.default_rng(12).uniform(0., 35., (1400, 3))
    labels = truth.raw_predict(points)
    options = dict(candidate_ids=['a', 'b'], box=35., starts=((2., 5.),),
                   maxiter=120, max_evaluations=250, gtol=1e-4)
    compact, compact_record = cj.fit_constrained(centres, [2.1, 2.85], points, labels, **options)

    class FullGraphEvaluation(cj._Evaluation):
        def _differentiate(self):
            value, gradient, values, jacobian = self.cached
            if gradient is not None:
                return self.cached
            gradient = torch.autograd.grad(self.cached_loss, self.cached_parameter,
                                           retain_graph=True)[0].detach().numpy()
            jacobian = np.vstack([torch.autograd.grad(item, self.cached_parameter,
                                  retain_graph=True)[0].detach().numpy()
                                  for item in self.cached_constraints])
            self.cached = (value, gradient, values, sparse.csr_matrix(jacobian))
            self.cached_loss = self.cached_constraints = None
            return self.cached

    monkeypatch.setattr(cj, '_Evaluation', FullGraphEvaluation)
    full, full_record = cj.fit_constrained(centres, [2.1, 2.85], points, labels, **options)
    assert compact is not None and full is not None
    assert compact_record['status'] == full_record['status'] == 'ok'
    for record in (compact_record, full_record):
        assert record['converged'] and record['assignment_stable']
        assert record['geometry_feasible'] and record['sampled_probability_valid']
        assert sum(round_['iterations'] for round_ in record['starts'][0]['rounds']) > 1
    np.testing.assert_allclose(compact.vector.detach().numpy(), full.vector.detach().numpy(),
                               atol=2e-9, rtol=1e-8)
    assert abs(compact_record['summed_nll'] - full_record['summed_nll']) < 1e-8
