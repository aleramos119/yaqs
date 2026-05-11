# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for the propagation module's noise characterization functionality."""

from __future__ import annotations

import re

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z
from mqt.yaqs.noise_char import propagation


class Parameters:
    """Container for default test parameters used in a lightweight open-quantum-system propagation test."""

    def __init__(self) -> None:
        """Initialize default test simulation parameters.

        This constructor sets up a collection of attributes used for running a simple
        open-quantum-system propagation test. Attributes and their meanings:
        - sites (int): Number of sites/spins. Default: 1.
        - sim_time (float): Total simulation time. Default: 0.6.
        - dt (float): Time step for propagation. Default: 0.2.
        - order (int): Integration/order parameter for the propagator. Default: 1.
        - threshold (float): Numerical/truncation tolerance used in algorithms. Default: 1e-4.
        - ntraj (int): Number of trajectories to average over (stochastic methods). Default: 1.
        - max_bond_dim (int): Maximum bond dimension for tensor-network representations. Default: 4.
        - j (float): Coupling constant used in the model Hamiltonian. Default: 1.
        - g (float): Local field (e.g., transverse field) parameter. Default: 0.5.
        - times (np.ndarray): 1-D array of time points computed as np.arange(0, sim_time + dt, dt).
        - n_obs (int): Number of observables (3 per site for Pauli x, y, z). Computed as sites * 3.
        - n_jump (int): Number of jump operators (2 per site, e.g., lowering and Pauli-Z). Computed as sites * 2.
        - n_t (int): Number of time points (len(times)).
        - gamma_rel (float): Relaxation (dissipative) rate. Default: 0.1.
        - gamma_deph (float): Dephasing rate. Default: 0.15.
        - d (int): Local Hilbert-space dimension (e.g., spin-1/2 -> 2). Default: 2.

        Notes:
        - The provided defaults are chosen for lightweight tests and can be modified
            on the instance after construction if different test scenarios are required.
        - The 'times' array explicitly includes the final time by using sim_time + dt
            as the stop value in np.arange.
        """
        self.sites = 1
        self.sim_time = 0.6
        self.dt = 0.2
        self.order = 1
        self.threshold = 1e-4
        self.ntraj = 1
        self.max_bond_dim = 4
        self.j = 1
        self.g = 0.5

        self.times = np.arange(0, self.sim_time + self.dt, self.dt)

        self.n_obs = self.sites * 3  # x, y, z for each site
        self.n_jump = self.sites * 2  # lowering and pauli_z for each site
        self.n_t = len(self.times)

        self.gamma_rel = 0.1
        self.gamma_deph = 0.15

        self.d = 2


def create_propagator_instance(
    test: Parameters,
) -> tuple[MPO, MPS, list[Observable], AnalogSimParams, CompactNoiseModel, propagation.Propagator]:
    """Create and initialize a Propagator instance.

    It is configured for an analog open quantum system simulation.
    This helper constructs an Ising Hamiltonian (MPO), a zero-filled initial MPS, a list of single-site
    Pauli observables (X, Y, Z for each site), an AnalogSimParams object with sampling enabled, and a
    CompactNoiseModel containing two noise channels ("lowering" and "pauli_z"). It then instantiates a
    propagation.Propagator using those objects, registers the observable list with the
    propagator, and runs a propagation using the reference noise model.
    Parameters
    ----------
    test : Parameters
        A parameter bundle object required to configure the system. Expected attributes:
          - sites (int): number of lattice sites (spins).
          - j (float): Ising coupling strength used to initialize the MPO Hamiltonian.
          - g (float): transverse field strength used to initialize the MPO Hamiltonian.
          - sim_time (float): total simulation elapsed time.
          - dt (float): simulation time step.
          - ntraj (int): number of stochastic trajectories to sample.
          - max_bond_dim (int): maximum MPS/MPO bond dimension for truncation.
          - threshold (float): singular-value threshold for truncation.
          - order (int): Trotter/order parameter for the simulator.
          - gamma_rel (float): strength of the "lowering" (relaxation) noise channel applied to all sites.
          - gamma_deph (float): strength of the "pauli_z" (dephasing) noise channel applied to all sites.

    Returns:
    -------
    tuple[MPO, MPS, list[Observable], AnalogSimParams, CompactNoiseModel, propagation.Propagator]
        A 6-tuple containing, in order:
          - h_0: MPO
              The initialized Ising Hamiltonian MPO for the given system parameters.
          - init_state: MPS
              The initialized many-body state (all zeros).
          - obs_list: list[Observable]
              The list of single-site Observable objects (X, Y, Z for each site).
          - sim_params: AnalogSimParams
              The simulation parameter object used to configure the propagator (with sample_timesteps=True).
          - ref_noise_model: CompactNoiseModel
              The compact noise model containing the "lowering" and "pauli_z" channels applied to all sites.
          - propagator: propagation.Propagator
              The propagator instance after calling set_observable_list(...) and run(ref_noise_model). The
              propagator therefore has performed the configured propagation at least once.
    """
    h_0 = MPO.ising(test.sites, test.j, test.g)

    # Define the initial state
    init_state = MPS(test.sites, state="zeros")

    obs_list = (
        [Observable(X(), site) for site in range(test.sites)]
        + [Observable(Y(), site) for site in range(test.sites)]
        + [Observable(Z(), site) for site in range(test.sites)]
    )

    sim_params = AnalogSimParams(
        observables=obs_list,
        elapsed_time=test.sim_time,
        dt=test.dt,
        num_traj=test.ntraj,
        max_bond_dim=test.max_bond_dim,
        threshold=test.threshold,
        order=test.order,
        sample_timesteps=True,
    )

    ref_noise_model = CompactNoiseModel([
        {"name": "lowering", "sites": list(range(test.sites)), "strength": test.gamma_rel},
        {"name": "pauli_z", "sites": list(range(test.sites)), "strength": test.gamma_deph},
    ])

    propagator = propagation.Propagator(
        sim_params=sim_params, hamiltonian=h_0, compact_noise_model=ref_noise_model, init_state=init_state
    )

    propagator.set_observable_list(obs_list)

    return h_0, init_state, obs_list, sim_params, ref_noise_model, propagator


def test_propagatorwithgradients_runs() -> None:
    """Test that `propagation.tjm_traj` executes correctly and returns expected output shapes.

    This test verifies that:
    - The function can be called with a valid `SimulationParameters` instance.
    - The returned values `t`, `original_exp_vals`, and `d_on_d_gk` are NumPy arrays.
    - The shapes of the outputs match the expected dimensions based on simulation parameters.
    - The average minimum and maximum trajectory time is returned as a list of None values.
    """
    # Prepare SimulationParameters
    test = Parameters()

    _, _, _obs_list, _, ref_noise_model, propagator = create_propagator_instance(test)

    propagator.run(ref_noise_model)

    assert isinstance(propagator.times, np.ndarray)
    assert isinstance(propagator.obs_array, np.ndarray)

    assert propagator.times.shape == (test.n_t,)
    assert propagator.obs_array.shape == (test.n_obs, test.n_t)


def test_raises_errors() -> None:
    """Test that `Propagator` raises expected ValueErrors.

    Verifies errors for:
    - Noise model referencing sites beyond the Hamiltonian.
    - Observable list referencing sites beyond the Hamiltonian.
    - Running without setting observables.
    - Mismatched compact noise model (names/sites) at run-time.
    """
    test = Parameters()

    h_0, init_state, obs_list, sim_params, ref_noise_model, _ = create_propagator_instance(test)

    # Test that Propagator raises a ValueError when
    # the noise model exceeds the number of sites of the Hamiltonian.
    exceed_ref_noise_model = CompactNoiseModel([
        {"name": "lowering", "sites": list(range(test.sites + 1)), "strength": test.gamma_rel},
        {"name": "pauli_z", "sites": list(range(test.sites)), "strength": test.gamma_deph},
    ])

    msg = "Noise site index exceeds number of sites in the Hamiltonian."
    with pytest.raises(ValueError, match=re.escape(msg)):
        propagator = propagation.Propagator(
            sim_params=sim_params, hamiltonian=h_0, compact_noise_model=exceed_ref_noise_model, init_state=init_state
        )

    # Test that Propagator raises a ValueError when
    # observable list exceeds the number of sites of the Hamiltonian.
    exceed_obs_list = (
        [Observable(X(), site) for site in range(test.sites)]
        + [Observable(Y(), site) for site in range(test.sites)]
        + [Observable(Z(), site) for site in range(test.sites + 1)]
    )
    propagator = propagation.Propagator(
        sim_params=sim_params, hamiltonian=h_0, compact_noise_model=ref_noise_model, init_state=init_state
    )
    msg = "Observable site index exceeds number of sites in the Hamiltonian."
    with pytest.raises(ValueError, match=re.escape(msg)):
        propagator.set_observable_list(exceed_obs_list)

    # Test that Propagator raises a ValueError when
    # observable list is not set.
    propagator = propagation.Propagator(
        sim_params=sim_params, hamiltonian=h_0, compact_noise_model=ref_noise_model, init_state=init_state
    )
    msg = "Observable list not set. Please use the set_observable_list method to set the observables."

    with pytest.raises(ValueError, match=re.escape(msg)):
        propagator.run(ref_noise_model)

    # Test that Propagator raises a ValueError when
    # the provided noise model does not match the initialized noise model.
    wrong_ref_noise_model = CompactNoiseModel([
        {"name": "lowering", "sites": list(range(test.sites)), "strength": test.gamma_rel},
        {"name": "pauli_x", "sites": list(range(test.sites)), "strength": test.gamma_deph},
    ])

    propagator = propagation.Propagator(
        sim_params=sim_params, hamiltonian=h_0, compact_noise_model=ref_noise_model, init_state=init_state
    )
    propagator.set_observable_list(obs_list)

    msg = "Noise model processes or sites do not match the initialized noise model."
    with pytest.raises(ValueError, match=re.escape(msg)):
        propagator.run(wrong_ref_noise_model)


def _make_propagator_2site() -> propagation.Propagator:
    """Build a minimal 2-site Propagator for Neumann-expansion tests.

    2 sites are required so that MPO boundary bonds properly collapse to 1,
    making ``to_sparse_matrix`` give the correct full dense matrix.
    """
    sites = 2
    dt = 0.1
    h_0 = MPO.ising(sites, 1.0, 0.5)
    init_state = MPS(sites, state="zeros")
    ref_noise_model = CompactNoiseModel([
        {"name": "lowering", "sites": list(range(sites)), "strength": 0.1},
        {"name": "pauli_z", "sites": list(range(sites)), "strength": 0.15},
    ])
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=dt,
        dt=dt,
        num_traj=1,
        max_bond_dim=4,
        threshold=1e-4,
        order=1,
    )
    return propagation.Propagator(
        sim_params=sim_params,
        hamiltonian=h_0,
        compact_noise_model=ref_noise_model,
        init_state=init_state,
    )


def test_neumann_expansion_order_zero_is_identity() -> None:
    """Order-0 expansion is the identity MPO."""
    prop = _make_propagator_2site()
    result = prop.neumann_expansion(dt=0.1, n=0)

    assert isinstance(result, MPO)
    assert result.length == prop.sites
    assert result.physical_dimension == 2

    dim = 2**prop.sites
    np.testing.assert_allclose(result.to_sparse_matrix().toarray(), np.eye(dim, dtype=complex), atol=1e-12)


def test_neumann_expansion_first_order() -> None:
    """Order-1 expansion equals I + H_eff * dt as a dense matrix."""
    prop = _make_propagator_2site()
    dt = 0.1
    result = prop.neumann_expansion(dt=dt, n=1)

    h_eff_dense = prop.effective_hamiltonian().to_sparse_matrix().toarray()
    dim = 2**prop.sites
    expected = np.eye(dim, dtype=complex) + dt * h_eff_dense

    np.testing.assert_allclose(result.to_sparse_matrix().toarray(), expected, atol=1e-10)


def test_neumann_expansion_higher_order_matches_dense() -> None:
    """Order-n expansion matches the dense geometric partial sum."""
    prop = _make_propagator_2site()
    dt = 0.1
    n = 3
    result = prop.neumann_expansion(dt=dt, n=n)

    h_eff_dense = prop.effective_hamiltonian().to_sparse_matrix().toarray()
    dim = 2**prop.sites
    a = dt * h_eff_dense
    expected = sum(np.linalg.matrix_power(a, k) for k in range(n + 1))

    np.testing.assert_allclose(result.to_sparse_matrix().toarray(), expected, atol=1e-10)


def test_neumann_expansion_negative_order_raises() -> None:
    """Negative expansion order raises ValueError."""
    test = Parameters()
    _, _, _, _, _, propagator = create_propagator_instance(test)

    with pytest.raises(ValueError, match="non-negative"):
        propagator.neumann_expansion(dt=test.dt, n=-1)


def test_neumann_expansion_with_compress() -> None:
    """Compressed expansion gives the same dense result as without compression."""
    prop = _make_propagator_2site()
    dt = 0.1
    n = 2
    uncompressed = prop.neumann_expansion(dt=dt, n=n)
    compressed = prop.neumann_expansion(dt=dt, n=n, compress=True, tol=1e-14)

    np.testing.assert_allclose(
        compressed.to_sparse_matrix().toarray(),
        uncompressed.to_sparse_matrix().toarray(),
        atol=1e-10,
    )


def test_kraus_operators_count() -> None:
    """kraus_operators returns 1 + n_jump MPOs."""
    prop = _make_propagator_2site()
    kraus = prop.kraus_operators(dt=0.1, n=1)

    # 1 no-jump operator + one per noise process
    assert len(kraus) == 1 + prop.n_jump
    for op in kraus:
        assert isinstance(op, MPO)
        assert op.length == prop.sites
        assert op.physical_dimension == 2


def test_kraus_operators_f0_matches_neumann() -> None:
    """F_0 equals the Neumann expansion of (I - H_eff dt)^{-1}."""
    prop = _make_propagator_2site()
    dt = 0.1
    n = 2
    kraus = prop.kraus_operators(dt=dt, n=n)
    resolvent = prop.neumann_expansion(dt=dt, n=n)

    np.testing.assert_allclose(
        kraus[0].to_sparse_matrix().toarray(),
        resolvent.to_sparse_matrix().toarray(),
        atol=1e-12,
    )


def test_kraus_operators_fm_matches_dense() -> None:
    """Each F_m equals (I - H_eff dt)^{-1} sqrt(gamma_m dt) L_m as a dense matrix."""
    prop = _make_propagator_2site()
    dt = 0.1
    n = 1
    kraus = prop.kraus_operators(dt=dt, n=n)

    resolvent_dense = kraus[0].to_sparse_matrix().toarray()

    for i, proc in enumerate(prop.expanded_noise_model.processes):
        gamma = float(proc["strength"])
        scale = (gamma * dt) ** 0.5

        # Build L_m dense: single-site operator embedded in full Hilbert space
        from mqt.yaqs.noise_char.propagation import Propagator
        op = np.asarray(proc["matrix"], dtype=complex)
        l_mpo = Propagator._single_site_mpo(op, proc["sites"][0], prop.sites, prop.hamiltonian.physical_dimension)
        l_dense = l_mpo.to_sparse_matrix().toarray()

        expected = resolvent_dense @ (scale * l_dense)
        np.testing.assert_allclose(kraus[i + 1].to_sparse_matrix().toarray(), expected, atol=1e-10)


def test_kraus_operators_adjoint_count_and_type() -> None:
    """kraus_operators_adjoint returns the same count and shapes as kraus_operators."""
    prop = _make_propagator_2site()
    adjoints = prop.kraus_operators_adjoint(dt=0.1, n=1)

    assert len(adjoints) == 1 + prop.n_jump
    for op in adjoints:
        assert isinstance(op, MPO)
        assert op.length == prop.sites
        assert op.physical_dimension == 2


def test_kraus_operators_adjoint_matches_conj_transpose() -> None:
    """Each F_i^dagger equals conj(F_i).T as a dense matrix."""
    prop = _make_propagator_2site()
    dt = 0.1
    n = 1
    kraus = prop.kraus_operators(dt=dt, n=n)
    adjoints = prop.kraus_operators_adjoint(dt=dt, n=n)

    for f, fd in zip(kraus, adjoints):
        f_dense = f.to_sparse_matrix().toarray()
        fd_dense = fd.to_sparse_matrix().toarray()
        np.testing.assert_allclose(fd_dense, f_dense.conj().T, atol=1e-12)


def test_kraus_derivative_shape() -> None:
    """dF has shape [n_jump][1+n_jump] with correct MPO metadata."""
    prop = _make_propagator_2site()
    dF = prop.kraus_operators_derivative(dt=0.1, n=1)

    assert len(dF) == prop.n_jump
    for row in dF:
        assert len(row) == 1 + prop.n_jump
        for op in row:
            assert isinstance(op, MPO)
            assert op.length == prop.sites
            assert op.physical_dimension == 2


def test_kraus_derivative_f0_formula() -> None:
    """dF[j][0] matches the exact Neumann recursion result for dR^(n)/dgamma_j.

    For n=1: D^(1) = B_j R^(0) = (-dt/2) P_j I = (-dt/2) P_j.
    """
    prop = _make_propagator_2site()
    dt, n = 0.1, 1
    dF = prop.kraus_operators_derivative(dt=dt, n=n)

    for j, proc in enumerate(prop.expanded_noise_model.processes):
        op = np.asarray(proc["matrix"], dtype=complex)
        ldagl = op.conj().T @ op
        from mqt.yaqs.noise_char.propagation import Propagator
        p_j = Propagator._single_site_mpo(ldagl, proc["sites"][0], prop.sites, prop.hamiltonian.physical_dimension)
        P_j = p_j.to_sparse_matrix().toarray()
        # For n=1: D^(1) = B_j @ R^(0) = (-dt/2) P_j @ I = (-dt/2) P_j
        expected = (-dt / 2.0) * P_j
        np.testing.assert_allclose(dF[j][0].to_sparse_matrix().toarray(), expected, atol=1e-10)


def test_kraus_derivative_matches_finite_difference() -> None:
    """Every dF[j][i] matches a central finite-difference perturbation of gamma_j."""
    prop = _make_propagator_2site()
    dt, n, eps = 0.1, 1, 1e-5

    dF = prop.kraus_operators_derivative(dt=dt, n=n)
    processes = prop.expanded_noise_model.processes

    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
    from mqt.yaqs.core.libraries.gate_library import Z

    def _kraus_at(delta: float, j: int) -> list:
        nm = CompactNoiseModel([
            {
                "name": p["name"],
                "sites": list(p["sites"]),
                "strength": float(p["strength"]) + (delta if k == j else 0.0),
            }
            for k, p in enumerate(processes)
        ])
        sp = AnalogSimParams(
            observables=[Observable(Z(), 0)],
            elapsed_time=dt, dt=dt, num_traj=1,
            max_bond_dim=4, threshold=1e-4, order=1,
        )
        return propagation.Propagator(
            sim_params=sp, hamiltonian=prop.hamiltonian,
            compact_noise_model=nm, init_state=prop.init_state,
        ).kraus_operators(dt=dt, n=n)

    for j in range(len(processes)):
        k_fwd = _kraus_at(+eps, j)
        k_bwd = _kraus_at(-eps, j)
        for i in range(1 + prop.n_jump):
            fd = (k_fwd[i].to_sparse_matrix().toarray() - k_bwd[i].to_sparse_matrix().toarray()) / (2 * eps)
            np.testing.assert_allclose(dF[j][i].to_sparse_matrix().toarray(), fd, atol=1e-8)


def test_kraus_derivative_precomputed_inputs() -> None:
    """Passing pre-computed kraus gives identical results to computing from scratch."""
    prop = _make_propagator_2site()
    dt, n = 0.1, 1

    kraus = prop.kraus_operators(dt=dt, n=n)

    dF_auto = prop.kraus_operators_derivative(dt=dt, n=n)
    dF_pre = prop.kraus_operators_derivative(dt=dt, n=n, kraus=kraus)

    for j in range(prop.n_jump):
        for i in range(1 + prop.n_jump):
            np.testing.assert_allclose(
                dF_pre[j][i].to_sparse_matrix().toarray(),
                dF_auto[j][i].to_sparse_matrix().toarray(),
                atol=1e-12,
            )


def test_kraus_derivative_adjoint_shape() -> None:
    """kraus_operators_derivative_adjoint returns the same shape as kraus_operators_derivative."""
    prop = _make_propagator_2site()
    dF_adj = prop.kraus_operators_derivative_adjoint(dt=0.1, n=1)
    assert len(dF_adj) == prop.n_jump
    for row in dF_adj:
        assert len(row) == 1 + prop.n_jump
        for op in row:
            assert isinstance(op, MPO)
            assert op.length == prop.sites


def test_kraus_derivative_adjoint_matches_conj_transpose() -> None:
    """Every dF_adj[j][i] equals dF[j][i].conj().T in dense form."""
    prop = _make_propagator_2site()
    dt, n = 0.1, 1

    dF = prop.kraus_operators_derivative(dt=dt, n=n)
    dF_adj = prop.kraus_operators_derivative_adjoint(dt=dt, n=n)

    for j in range(prop.n_jump):
        for i in range(1 + prop.n_jump):
            d_dense = dF[j][i].to_sparse_matrix().toarray()
            d_adj_dense = dF_adj[j][i].to_sparse_matrix().toarray()
            np.testing.assert_allclose(d_adj_dense, d_dense.conj().T, atol=1e-12)


def test_kraus_derivative_adjoint_precomputed_kraus() -> None:
    """Passing pre-computed kraus to derivative_adjoint gives identical results."""
    prop = _make_propagator_2site()
    dt, n = 0.1, 1

    kraus = prop.kraus_operators(dt=dt, n=n)
    dF_adj_auto = prop.kraus_operators_derivative_adjoint(dt=dt, n=n)
    dF_adj_pre = prop.kraus_operators_derivative_adjoint(dt=dt, n=n, kraus=kraus)

    for j in range(prop.n_jump):
        for i in range(1 + prop.n_jump):
            np.testing.assert_allclose(
                dF_adj_pre[j][i].to_sparse_matrix().toarray(),
                dF_adj_auto[j][i].to_sparse_matrix().toarray(),
                atol=1e-12,
            )


# ---------------------------------------------------------------------------
# backward_kraus_map
# ---------------------------------------------------------------------------


def test_backward_kraus_map_returns_mpo() -> None:
    """backward_kraus_map returns an MPO of the correct length."""
    prop = _make_propagator_2site()
    obs = MPO()
    obs.identity(prop.sites, prop.hamiltonian.physical_dimension)
    result = prop.backward_kraus_map(obs, dt=0.1, n=1)
    assert isinstance(result, MPO)
    assert result.length == prop.sites


def test_backward_kraus_map_identity_observable() -> None:
    """K(I) = sum_j F_j† F_j; verify against explicit dense sum."""
    prop = _make_propagator_2site()
    dt, n = 0.1, 1

    kraus = prop.kraus_operators(dt=dt, n=n)

    # dense reference: sum_j F_j† F_j
    dim = 2**prop.sites
    expected = np.zeros((dim, dim), dtype=complex)
    for f in kraus:
        f_dense = f.to_sparse_matrix().toarray()
        expected += f_dense.conj().T @ f_dense

    obs = MPO()
    obs.identity(prop.sites, prop.hamiltonian.physical_dimension)
    result = prop.backward_kraus_map(obs, dt=dt, n=n, kraus=kraus)
    result_dense = result.to_sparse_matrix().toarray()

    np.testing.assert_allclose(result_dense, expected, atol=1e-12)


def test_backward_kraus_map_matches_dense() -> None:
    """K(O) matches the explicit dense sum F_j† O F_j for a non-trivial O."""
    prop = _make_propagator_2site()
    dt, n = 0.1, 1

    kraus = prop.kraus_operators(dt=dt, n=n)

    # Use the Ising Hamiltonian MPO as the observable
    obs = MPO.ising(prop.sites, 1.0, 0.5)
    obs_dense = obs.to_sparse_matrix().toarray()

    dim = 2**prop.sites
    expected = np.zeros((dim, dim), dtype=complex)
    for f in kraus:
        f_dense = f.to_sparse_matrix().toarray()
        expected += f_dense.conj().T @ obs_dense @ f_dense

    result = prop.backward_kraus_map(obs, dt=dt, n=n, kraus=kraus)
    result_dense = result.to_sparse_matrix().toarray()

    np.testing.assert_allclose(result_dense, expected, atol=1e-12)


def test_backward_kraus_map_precomputed_kraus() -> None:
    """Passing pre-computed kraus gives identical result to computing from scratch."""
    prop = _make_propagator_2site()
    dt, n = 0.1, 1

    obs = MPO()
    obs.identity(prop.sites, prop.hamiltonian.physical_dimension)

    kraus = prop.kraus_operators(dt=dt, n=n)
    result_auto = prop.backward_kraus_map(obs, dt=dt, n=n)
    result_pre = prop.backward_kraus_map(obs, dt=dt, n=n, kraus=kraus)

    np.testing.assert_allclose(
        result_pre.to_sparse_matrix().toarray(),
        result_auto.to_sparse_matrix().toarray(),
        atol=1e-12,
    )
