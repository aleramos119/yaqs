# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Performs the simulation of the Ising model and returns expectations values and  A_kn trahectories."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import GateLibrary

if TYPE_CHECKING:
    from pathlib import Path

    from mqt.yaqs.core.data_structures.networks import MPO, MPS
    from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel, NoiseModel


def noise_model_to_operator_list(noise_model: NoiseModel) -> list[Observable]:
    """Converts a noise model to a list of observables.

    Args:
        noise_model (NoiseModel): The noise model to convert.

    Returns:
        list[Observable]: A list of observables corresponding to the noise processes in the noise model.
    """
    noise_list: list[Observable] = []

    for proc in noise_model.processes:
        gate_cls = getattr(GateLibrary, proc["name"])
        gate_instance = gate_cls()
        if gate_instance.interaction == 1:
            noise_list.extend(Observable(gate_cls(), site) for site in proc["sites"])
        else:
            # 2-site operator: one Observable for the pair; fix name to registered key
            gate_instance.name = proc["name"]
            noise_list.append(Observable(gate_instance, proc["sites"]))
    return noise_list


class Propagator:
    r"""High-level propagator that runs an MPS-based Lindblad simulation.

    The class wraps simulator inputs, performs
    consistency checks between noise models and the Hamiltonian, augments the
    observable set with Lindblad-derived A_kn operators (sensitivities of
    expectation values w.r.t. jump rates), runs the underlying simulator, and
    post-processes simulator outputs into convenient arrays for analysis.

    Attributes:
    obs_list : list[Observable]
        (Set after set_observable_list) Deep copy of user-provided observables.
    n_obs : int
        (Set after set_observable_list) Number of observables.
    times : array-like
        Time grid used by the most recent run (copied from sim_params.times).
    obs_traj : list[Observable]
        Observables returned by the simulator corresponding to the original
        user-requested observables (populated by run).
    obs_array : numpy.ndarray
        Array of observable trajectories with shape (n_obs, n_timesteps).
    d_on_d_gk : numpy.ndarray
        Object-array of Observable entries (shape [n_jump, n_obs]) corresponding
        to A_kn-like operators (or zero placeholders) computed by the simulator
        and integrated in time.
    d_on_d_gk_array : numpy.ndarray
        Numeric array of integrated A_kn trajectories (shape [n_jump, n_obs]).
    Other internal fields may be set during execution (e.g., temporary lists and
    simulator-specific containers).
    Public methods
    set_observable_list(obs_list: list[Observable]) -> None
        Store and validate a deep copy of obs_list. Validates that every site index
        referenced by the observables is within the range [0, sites-1]. Sets
        n_obs and flips set_observables to True. Raises ValueError for empty lists
        or out-of-range site indices.
    run(noise_model: CompactNoiseModel) -> None
        Execute the propagation. Requires that set_observables has been called.
        Validates that the provided compact noise_model matches the one used to
        construct this propagator (same process names and site assignments).
        Constructs A_kn-like observables for each matching jump/operator pair,
        appends them to the observable list, builds a new AnalogSimParams instance
        for the simulator, and invokes the underlying simulator with the expanded
        noise model. Post-processes results by trapezoidally integrating A_kn
        trajectories, arranging them into object and numeric arrays (d_on_d_gk and
        d_on_d_gk_array) and extracting obs_traj and obs_array for the original
        observables.
        - During initialization: if any site index in compact_noise_model.expanded_noise_model
          exceeds the number of sites in the Hamiltonian.
        - set_observable_list: if obs_list is empty or contains observables that
          reference out-of-range site indices.
        - run: if observables have not been set (set_observables is False) or if the
          provided noise_model does not match the initialized compact_noise_model
          in process names or site indices.
    - All constructor inputs are deep-copied to avoid accidental external mutation.
    - The class expects external types (AnalogSimParams, MPO, MPS, CompactNoiseModel,
      Observable) to expose particular attributes (for example, `times`, `length`,
      `expanded_noise_model`, `compact_processes`, `gate`, `sites`, and `results`).
    - The A_kn operators constructed in run follow the Lindblad derivative form:
      L_k^\dagger O L_k - 0.5 {L_k^\dagger L_k, O}, computed only for observables
      that act on the same site(s) as the corresponding jump operator.
    - The user-facing numeric arrays (obs_array and d_on_d_gk_array) are convenient
      summaries for optimization or analysis tasks (e.g., gradient-based fitting of
      jump rates).
    """

    def __init__(
        self,
        *,
        sim_params: AnalogSimParams,
        hamiltonian: MPO,
        compact_noise_model: CompactNoiseModel,
        init_state: MPS,
    ) -> None:
        """Initialize a Propagation object for simulating open quantum system dynamics.

        This constructor deep-copies the provided inputs and derives internal
        structures needed for propagation of an MPS under a Hamiltonian with
        a compact noise model.
        Parameters.
        ----------
        sim_params : AnalogSimParams
            Simulation parameters container. A deep copy is stored as
            self.sim_params. It is expected to provide a sequence/array
            `times` used to determine the number of time steps.
        hamiltonian : MPO
            Matrix product operator representing the Hamiltonian. A deep copy
            is stored as self.hamiltonian. The MPO must expose a `length`
            attribute indicating the number of sites.
        compact_noise_model : CompactNoiseModel
            Compact representation of the noise model. A deep copy is stored
            as self.compact_noise_model. Its `expanded_noise_model` attribute
            is deep-copied to self.expanded_noise_model and converted into a
            list of jump operators.
        init_state : MPS
            Initial many-body quantum state as a matrix product state.
            A deep copy is stored as self.init_state.
        Attributes set
        --------------
        sim_params : AnalogSimParams
            Deep copy of the provided simulation parameters.
        hamiltonian : MPO
            Deep copy of the provided Hamiltonian MPO.
        compact_noise_model : CompactNoiseModel
            Deep copy of the provided compact noise model.
        init_state : MPS
            Deep copy of the provided initial state.
        expanded_noise_model
            Deep copy of compact_noise_model.expanded_noise_model.
        noise_list : list[Observable]
            List of noise (jump) operators produced by converting the expanded
            noise model via noise_model_to_operator_list.
        n_jump : int
            Number of jump operators (len(self.noise_list)).
        n_t : int
            Number of time steps (len(self.sim_params.times)).
        sites : int
            Number of sites in the chain (self.hamiltonian.length).
        set_observables : bool
            Flag indicating whether observables have been set (initialized to False).

        Raises:
        ------
        ValueError: If any site index referenced in expanded_noise_model.processes is
            greater than or equal to the number of sites in the Hamiltonian,
            a ValueError is raised with the message
            "Noise site index exceeds number of sites in the Hamiltonian."

        Notes:
        -----
        - All provided inputs are deep-copied to avoid accidental external mutation.
        - This method performs basic consistency checking between the noise
          model and the Hamiltonian site count.
        """
        self.sim_params: AnalogSimParams = copy.deepcopy(sim_params)
        self.hamiltonian: MPO = copy.deepcopy(hamiltonian)
        self.compact_noise_model: CompactNoiseModel = copy.deepcopy(compact_noise_model)
        self.init_state: MPS = copy.deepcopy(init_state)

        self.expanded_noise_model = copy.deepcopy(self.compact_noise_model.expanded_noise_model)

        self.noise_list: list[Observable] = noise_model_to_operator_list(self.expanded_noise_model)

        self.n_jump: int = len(self.noise_list)  # number of jump operators

        self.n_t: int = len(self.sim_params.times)  # number of time steps

        self.sites: int = self.hamiltonian.length  # number of sites in the chain

        self.set_observables: bool = False

        if max(proc["sites"][0] for proc in self.expanded_noise_model.processes) >= self.sites:
            msg = "Noise site index exceeds number of sites in the Hamiltonian."
            raise ValueError(msg)

    def set_observable_list(self, obs_list: list[Observable]) -> None:
        """Set the list of observables to be used for propagation.

        This method stores a deep copy of the provided observable list on the instance,
        validates that all referenced site indices lie within the allowed range of the
        Hamiltonian, and updates bookkeeping attributes.

        Args:
            obs_list (list[Observable]): Sequence of Observable objects. Each Observable
                must expose a `sites` attribute that is either an int (single site) or
                a list of ints (multiple sites).
        Side effects:
            - self.obs_list is set to a deep copy of obs_list.
            - self.n_obs is set to the number of observables (len(self.obs_list)).
            - self.set_observables is set to True.

        Raises:
            ValueError: If any site index in the observables is greater than or equal
                to self.sites (i.e., outside the range of available sites).
            ValueError: If obs_list is empty (which makes site-index validation via max()
                impossible) or if observables do not provide valid site information.
        """
        self.obs_list = copy.deepcopy(obs_list)

        all_obs_sites = [
            site for obs in obs_list for site in (obs.sites if isinstance(obs.sites, list) else [obs.sites])
        ]

        if max(all_obs_sites) >= self.sites:
            msg = "Observable site index exceeds number of sites in the Hamiltonian."
            raise ValueError(msg)

        self.n_obs = len(self.obs_list)  # number of measurement operators

        self.set_observables = True

    @staticmethod
    def _single_site_mpo(op: np.ndarray, site: int, length: int, d: int) -> MPO:
        """Build an MPO with ``op`` at ``site`` and identity on all other sites.

        Args:
            op: Local operator matrix of shape ``(d, d)``.
            site: Site index where ``op`` is placed.
            length: Total chain length.
            d: Physical (local Hilbert-space) dimension.

        Returns:
            MPO with bond dimension 1 throughout.
        """
        from mqt.yaqs.core.data_structures.networks import MPO

        identity = np.eye(d, dtype=complex).reshape(d, d, 1, 1)
        tensors = [identity.copy() if i != site else op.astype(complex).reshape(d, d, 1, 1) for i in range(length)]
        result = MPO()
        result.tensors = tensors
        result.length = length
        result.physical_dimension = d
        return result

    @staticmethod
    def _adjacent_two_site_mpo(op: np.ndarray, site_a: int, site_b: int, length: int, d: int) -> MPO:
        """Build an MPO for a two-site operator on adjacent sites ``site_a`` and ``site_b = site_a + 1``.

        Factorizes ``op`` (shape ``(d^2, d^2)``) via SVD into a pair of tensors and
        embeds them in the chain with identity on all other sites.

        Args:
            op: Two-site operator matrix of shape ``(d^2, d^2)`` with row index
                ``(out_a, out_b)`` and column index ``(in_a, in_b)`` in row-major order.
            site_a: Left site index.
            site_b: Right site index; must equal ``site_a + 1``.
            length: Total chain length.
            d: Physical dimension per site.

        Returns:
            MPO representing ``op`` embedded in the full chain.
        """
        from mqt.yaqs.core.data_structures.networks import MPO

        # Reshape: op[out_a*d+out_b, in_a*d+in_b] → op4[out_a, out_b, in_a, in_b]
        # Transpose → m[out_a*d+in_a, out_b*d+in_b], then SVD.
        m = op.astype(complex).reshape(d, d, d, d).transpose(0, 2, 1, 3).reshape(d * d, d * d)
        u, s, vh = np.linalg.svd(m, full_matrices=False)
        chi = len(s)
        sqrt_s = np.sqrt(s)

        a_mat = (u * sqrt_s).reshape(d, d, 1, chi)
        b_mat = (sqrt_s[:, np.newaxis] * vh).reshape(chi, d, d).transpose(1, 2, 0)[:, :, :, np.newaxis]

        identity = np.eye(d, dtype=complex).reshape(d, d, 1, 1)
        tensors = []
        for i in range(length):
            if i == site_a:
                tensors.append(a_mat)
            elif i == site_b:
                tensors.append(b_mat)
            else:
                tensors.append(identity.copy())
        result = MPO()
        result.tensors = tensors
        result.length = length
        result.physical_dimension = d
        return result

    @staticmethod
    def _product_two_site_mpo(
        op1: np.ndarray, site1: int, op2: np.ndarray, site2: int, length: int, d: int
    ) -> MPO:
        """Build an MPO for the product operator ``op1 ⊗ op2`` at non-adjacent sites.

        Both ``op1`` and ``op2`` are single-site operators; identity is placed on all
        other sites. The resulting MPO has bond dimension 1 throughout.

        Args:
            op1: Single-site operator of shape ``(d, d)`` placed at ``site1``.
            site1: Site index for ``op1``.
            op2: Single-site operator of shape ``(d, d)`` placed at ``site2``.
            site2: Site index for ``op2``; must differ from ``site1``.
            length: Total chain length.
            d: Physical dimension per site.

        Returns:
            MPO representing ``op1 ⊗ I ⊗ ... ⊗ I ⊗ op2`` embedded in the full chain.
        """
        from mqt.yaqs.core.data_structures.networks import MPO

        identity = np.eye(d, dtype=complex).reshape(d, d, 1, 1)
        tensors = []
        for i in range(length):
            if i == site1:
                tensors.append(op1.astype(complex).reshape(d, d, 1, 1))
            elif i == site2:
                tensors.append(op2.astype(complex).reshape(d, d, 1, 1))
            else:
                tensors.append(identity.copy())
        result = MPO()
        result.tensors = tensors
        result.length = length
        result.physical_dimension = d
        return result

    def effective_hamiltonian(self) -> MPO:
        r"""Return the effective non-Hermitian Hamiltonian as an MPO.

        Computes

        .. math::
            H_{\mathrm{eff}} = -i H - \frac{1}{2} \sum_m \gamma_m L_m^\dagger L_m

        where :math:`H` is the system Hamiltonian and :math:`L_m` are the jump
        operators from the expanded noise model with rates :math:`\gamma_m`.

        Returns:
            MPO: :math:`H_{\mathrm{eff}}` with the same length and physical
            dimension as ``self.hamiltonian``.

        Notes:
            All process strengths must be concrete floats. If the noise model
            uses distribution-valued strengths, resolve them with
            ``CompactNoiseModel.sample()`` before constructing this
            ``Propagator``.
        """
        d = self.hamiltonian.physical_dimension
        n = self.sites

        h_eff: MPO = (-1j) * self.hamiltonian

        for proc in self.expanded_noise_model.processes:
            gamma = float(proc["strength"])
            sites = proc["sites"]

            if len(sites) == 1:
                op = np.asarray(proc["matrix"], dtype=complex)
                ldagl = op.conj().T @ op
                term_mpo = self._single_site_mpo(ldagl, sites[0], n, d)
            elif "matrix" in proc:
                # Adjacent two-site operator
                op = np.asarray(proc["matrix"], dtype=complex)
                ldagl = op.conj().T @ op
                term_mpo = self._adjacent_two_site_mpo(ldagl, sites[0], sites[1], n, d)
            else:
                # Long-range Crosstalk: L = L1 ⊗ L2, so L†L = (L1†L1) ⊗ (L2†L2)
                mat1, mat2 = (np.asarray(f, dtype=complex) for f in proc["factors"])
                ldagl1 = mat1.conj().T @ mat1
                ldagl2 = mat2.conj().T @ mat2
                term_mpo = self._product_two_site_mpo(ldagl1, sites[0], ldagl2, sites[1], n, d)

            h_eff = h_eff + ((-0.5 * gamma) * term_mpo)

        return h_eff

    def neumann_expansion(
        self,
        dt: float,
        n: int,
        *,
        compress: bool = False,
        tol: float = 1e-12,
        max_bond_dim: int | None = None,
    ) -> MPO:
        r"""Return the n-th order Neumann expansion of :math:`(I - H_{\mathrm{eff}}\,dt)^{-1}`.

        Computes

        .. math::

            \sum_{k=0}^{n} (H_{\mathrm{eff}}\,dt)^k
            = I + H_{\mathrm{eff}}\,dt + (H_{\mathrm{eff}}\,dt)^2 + \cdots
              + (H_{\mathrm{eff}}\,dt)^n

        entirely in MPO format.  Each additional term multiplies the bond
        dimension of the running power by that of :math:`H_{\mathrm{eff}}`, so
        for large ``n`` pass ``compress=True`` to keep the representation
        compact.

        Args:
            dt: Time step :math:`dt`.
            n: Expansion order (number of terms beyond the identity, so the
               result contains :math:`n+1` terms total).
            compress: If ``True``, compress the running sum after each
               accumulation step using SVD sweeps.
            tol: SVD truncation threshold used when ``compress=True``.
            max_bond_dim: Hard cap on the bond dimension when
               ``compress=True``; ``None`` means no cap.

        Returns:
            MPO: :math:`\sum_{k=0}^{n}(H_{\mathrm{eff}}\,dt)^k`.

        Raises:
            ValueError: If ``n`` is negative.
        """
        if n < 0:
            msg = "Expansion order n must be non-negative."
            raise ValueError(msg)

        from mqt.yaqs.core.data_structures.networks import MPO

        d = self.hamiltonian.physical_dimension

        identity = MPO()
        identity.identity(self.sites, d)

        a = dt * self.effective_hamiltonian()

        result = copy.deepcopy(identity)
        power = copy.deepcopy(identity)

        for _ in range(n):
            power = a @ power
            result = result + power
            if compress:
                result.compress(tol=tol, max_bond_dim=max_bond_dim)

        return result

    def kraus_operators(
        self,
        dt: float,
        n: int,
        *,
        compress: bool = False,
        tol: float = 1e-12,
        max_bond_dim: int | None = None,
    ) -> list[MPO]:
        r"""Return the Kraus operators for one no-jump / jump time step.

        Computes

        .. math::

            F_0 &= (I - H_{\mathrm{eff}}\,dt)^{-1} \\
            F_m &= (I - H_{\mathrm{eff}}\,dt)^{-1}\,\sqrt{\gamma_m\,dt}\;L_m
                   \quad m = 1,\ldots,M

        where :math:`(I - H_{\mathrm{eff}}\,dt)^{-1}` is approximated by the
        ``n``-th order Neumann expansion and each :math:`L_m` is the jump
        operator of the :math:`m`-th noise process with rate :math:`\gamma_m`.
        All operators are returned as MPOs.

        Args:
            dt: Time step :math:`dt`.
            n: Neumann expansion order used to approximate
               :math:`(I - H_{\mathrm{eff}}\,dt)^{-1}`.
            compress: If ``True``, compress every MPO product using SVD sweeps.
            tol: SVD truncation threshold used when ``compress=True``.
            max_bond_dim: Hard cap on the bond dimension when
               ``compress=True``; ``None`` means no cap.

        Returns:
            list[MPO]: ``[F_0, F_1, ..., F_M]`` — the no-jump operator
            followed by one jump operator per noise process, in the order
            they appear in ``expanded_noise_model``.

        Raises:
            ValueError: If ``n`` is negative (propagated from
               :meth:`neumann_expansion`).
        """
        d = self.hamiltonian.physical_dimension
        n_sites = self.sites

        resolvent = self.neumann_expansion(dt, n, compress=compress, tol=tol, max_bond_dim=max_bond_dim)
        kraus: list[MPO] = [resolvent]

        for proc in self.expanded_noise_model.processes:
            gamma = float(proc["strength"])
            sites = proc["sites"]
            scale = (gamma * dt) ** 0.5

            if len(sites) == 1:
                op = np.asarray(proc["matrix"], dtype=complex)
                l_mpo = self._single_site_mpo(op, sites[0], n_sites, d)
            elif "matrix" in proc:
                op = np.asarray(proc["matrix"], dtype=complex)
                l_mpo = self._adjacent_two_site_mpo(op, sites[0], sites[1], n_sites, d)
            else:
                mat1, mat2 = (np.asarray(f, dtype=complex) for f in proc["factors"])
                l_mpo = self._product_two_site_mpo(mat1, sites[0], mat2, sites[1], n_sites, d)

            f_m = resolvent @ (scale * l_mpo)
            if compress:
                f_m.compress(tol=tol, max_bond_dim=max_bond_dim)
            kraus.append(f_m)

        return kraus

    def kraus_operators_adjoint(
        self,
        dt: float,
        n: int,
        *,
        compress: bool = False,
        tol: float = 1e-12,
        max_bond_dim: int | None = None,
    ) -> list[MPO]:
        r"""Return the adjoints of the Kraus operators.

        Computes :math:`[F_0^\dagger, F_1^\dagger, \ldots, F_M^\dagger]` by
        calling :meth:`kraus_operators` and taking the Hermitian adjoint of
        each element via :meth:`~mqt.yaqs.core.data_structures.networks.MPO.adjoint`.

        Args:
            dt: Time step :math:`dt`.
            n: Neumann expansion order used to approximate
               :math:`(I - H_{\mathrm{eff}}\,dt)^{-1}`.
            compress: If ``True``, compress every MPO using SVD sweeps.
            tol: SVD truncation threshold used when ``compress=True``.
            max_bond_dim: Hard cap on the bond dimension when
               ``compress=True``; ``None`` means no cap.

        Returns:
            list[MPO]: ``[F_0^\\dagger, F_1^\\dagger, ..., F_M^\\dagger]``.

        Raises:
            ValueError: If ``n`` is negative (propagated from
               :meth:`neumann_expansion`).
        """
        return [f.adjoint() for f in self.kraus_operators(dt, n, compress=compress, tol=tol, max_bond_dim=max_bond_dim)]

    def kraus_operators_derivative(
        self,
        dt: float,
        n: int,
        *,
        resolvent: MPO | None = None,
        kraus: list[MPO] | None = None,
        compress: bool = False,
        tol: float = 1e-12,
        max_bond_dim: int | None = None,
    ) -> list[list[MPO]]:
        r"""Return derivatives of all Kraus operators w.r.t. all jump rates.

        Using 1-based jump index :math:`k` (maps to ``processes[k-1]`` in code)
        and differentiation index :math:`j`, the formulas are:

        Let :math:`A = H_{\mathrm{eff}}\,dt`, :math:`R^{(n)} = \sum_{p=0}^n A^p`,
        :math:`B_j = -\tfrac{dt}{2} P_j` where :math:`P_j = L_j^\dagger L_j`.
        Differentiating :math:`R^{(p)} = I + A\,R^{(p-1)}` term-by-term gives the
        recursion

        .. math::

            D_j^{(0)} = 0, \qquad
            D_j^{(p)} = B_j\,R^{(p-1)} + A\,D_j^{(p-1)}

        so that :math:`D_j^{(n)} = \partial R^{(n)} / \partial\gamma_j`.  The
        Kraus derivatives then follow from the product rule:

        .. math::

            \frac{\partial F_0}{\partial\gamma_j} &= D_j^{(n)} \\[4pt]
            \frac{\partial F_k}{\partial\gamma_j} &= D_j^{(n)}\,\sqrt{\gamma_k\,dt}\,L_k
              + \delta_{jk}\,\frac{F_k}{2\gamma_k}

        Args:
            dt: Time step :math:`dt`.
            n: Neumann expansion order.
            resolvent: Ignored (kept for API compatibility); the resolvent is
                recomputed internally from the Neumann sequence required for the
                recursion.
            kraus: Pre-computed list ``[F_0, F_1, ..., F_M]`` from
                :meth:`kraus_operators`. If ``None``, rebuilt internally.
            compress: If ``True``, compress each intermediate and output MPO.
            tol: SVD truncation threshold used when ``compress=True``.
            max_bond_dim: Hard cap on the bond dimension when
                ``compress=True``; ``None`` means no cap.

        Returns:
            list[list[MPO]]: ``dF[j][i]`` =
            :math:`\partial F_i / \partial \gamma_j` where ``j`` indexes
            processes (0-based, matching ``expanded_noise_model.processes``)
            and ``i`` indexes Kraus operators (0 = no-jump, ``k`` = jump
            operator for process ``k-1``).

        Raises:
            ValueError: If ``n`` is negative (propagated from
                :meth:`neumann_expansion`).

        Note:
            Bond dimensions grow as :math:`\chi_R^2` per derivative entry.
            Use ``compress=True`` for large ``n``.
        """
        from mqt.yaqs.core.data_structures.networks import MPO as _MPO

        d = self.hamiltonian.physical_dimension
        n_sites = self.sites

        # Build A = dt * H_eff and the Neumann sequence R^(0), ..., R^(n).
        # The exact derivative of R^(n) w.r.t. gamma_j satisfies the recursion
        #   D^(0) = 0
        #   D^(p) = B_j @ R^(p-1) + A @ D^(p-1),   B_j = -dt/2 * P_j
        # which is derived by differentiating R^(p) = I + A R^(p-1) term-by-term.
        h_eff = self.effective_hamiltonian()
        a_mpo = dt * h_eff

        identity = _MPO()
        identity.identity(n_sites, d)

        neumann_seq: list[MPO] = [copy.deepcopy(identity)]  # neumann_seq[p] = R^(p)
        for _ in range(n):
            r_next = copy.deepcopy(identity) + (a_mpo @ neumann_seq[-1])
            if compress:
                r_next.compress(tol=tol, max_bond_dim=max_bond_dim)
            neumann_seq.append(r_next)

        resolvent = neumann_seq[-1]  # R^(n)

        # Build scaled jump MPOs  sqrt(gamma_k dt) L_k  and Kraus operators F_k.
        processes = self.expanded_noise_model.processes
        scaled_l_mpos: list[MPO] = []
        for proc in processes:
            gamma = float(proc["strength"])
            sites = proc["sites"]
            scale = (gamma * dt) ** 0.5
            if len(sites) == 1:
                op = np.asarray(proc["matrix"], dtype=complex)
                l_mpo = self._single_site_mpo(op, sites[0], n_sites, d)
            elif "matrix" in proc:
                op = np.asarray(proc["matrix"], dtype=complex)
                l_mpo = self._adjacent_two_site_mpo(op, sites[0], sites[1], n_sites, d)
            else:
                mat1, mat2 = (np.asarray(f, dtype=complex) for f in proc["factors"])
                l_mpo = self._product_two_site_mpo(mat1, sites[0], mat2, sites[1], n_sites, d)
            scaled_l_mpos.append(scale * l_mpo)

        if kraus is None:
            kraus = [resolvent]
            for sl in scaled_l_mpos:
                f_k = resolvent @ sl
                if compress:
                    f_k.compress(tol=tol, max_bond_dim=max_bond_dim)
                kraus.append(f_k)

        dF: list[list[MPO]] = []

        for j, proc_j in enumerate(processes):
            sites_j = proc_j["sites"]

            if len(sites_j) == 1:
                op_j = np.asarray(proc_j["matrix"], dtype=complex)
                p_j = self._single_site_mpo(op_j.conj().T @ op_j, sites_j[0], n_sites, d)
            elif "matrix" in proc_j:
                op_j = np.asarray(proc_j["matrix"], dtype=complex)
                p_j = self._adjacent_two_site_mpo(op_j.conj().T @ op_j, sites_j[0], sites_j[1], n_sites, d)
            else:
                mat1, mat2 = (np.asarray(f, dtype=complex) for f in proc_j["factors"])
                p_j = self._product_two_site_mpo(
                    mat1.conj().T @ mat1, sites_j[0], mat2.conj().T @ mat2, sites_j[1], n_sites, d
                )

            b_j = (-dt / 2.0) * p_j  # B_j = dA/d gamma_j = -dt/2 * P_j

            # Recursion:  D^(p) = B_j @ R^(p-1) + A @ D^(p-1),  D^(0) = 0
            d_curr: MPO | None = None
            for p in range(1, n + 1):
                term1 = b_j @ neumann_seq[p - 1]
                d_curr = term1 if d_curr is None else term1 + (a_mpo @ d_curr)
                if compress:
                    d_curr.compress(tol=tol, max_bond_dim=max_bond_dim)

            dF_j: list[MPO] = []

            # dF[j][0] = D^(n)  (zero when n=0 since R^(0)=I has no gamma dependence)
            zero = 0.0 * copy.deepcopy(identity)
            dF_j.append(zero if d_curr is None else d_curr)

            # dF[j][k] = D^(n) @ (sqrt(gamma_k dt) L_k) + delta_{jk} F_k/(2 gamma_k)
            for k, (proc_k, sl_k, f_k) in enumerate(zip(processes, scaled_l_mpos, kraus[1:]), start=1):
                df_k = (0.0 * copy.deepcopy(f_k)) if d_curr is None else (d_curr @ sl_k)
                if k - 1 == j:  # delta_{jk}: k is 1-based, j is 0-based
                    gamma_k = float(proc_k["strength"])
                    df_k = df_k + ((1.0 / (2.0 * gamma_k)) * f_k)
                if compress:
                    df_k.compress(tol=tol, max_bond_dim=max_bond_dim)
                dF_j.append(df_k)

            dF.append(dF_j)

        return dF

    def kraus_operators_derivative_adjoint(
        self,
        dt: float,
        n: int,
        *,
        kraus: list[MPO] | None = None,
        compress: bool = False,
        tol: float = 1e-12,
        max_bond_dim: int | None = None,
    ) -> list[list[MPO]]:
        r"""Return the adjoints of the Kraus operator derivatives.

        Computes ``dF_adj[j][i]`` =
        :math:`(\partial F_i / \partial\gamma_j)^\dagger` by calling
        :meth:`kraus_operators_derivative` and applying
        :meth:`~mqt.yaqs.core.data_structures.networks.MPO.adjoint` to every
        element.

        Args:
            dt: Time step :math:`dt`.
            n: Neumann expansion order.
            kraus: Pre-computed list ``[F_0, F_1, ..., F_M]`` passed through
                to :meth:`kraus_operators_derivative`.
            compress: If ``True``, compress every MPO using SVD sweeps.
            tol: SVD truncation threshold used when ``compress=True``.
            max_bond_dim: Hard cap on the bond dimension when
                ``compress=True``; ``None`` means no cap.

        Returns:
            list[list[MPO]]: ``dF_adj[j][i]`` =
            :math:`(\partial F_i/\partial\gamma_j)^\dagger`.

        Raises:
            ValueError: If ``n`` is negative (propagated from
                :meth:`neumann_expansion`).
        """
        return [
            [op.adjoint() for op in row]
            for row in self.kraus_operators_derivative(
                dt, n, kraus=kraus, compress=compress, tol=tol, max_bond_dim=max_bond_dim
            )
        ]

    def write_traj(self, output_file: Path) -> None:
        """Saves the optimized trajectory of expectation values to a text file.

        This method reshapes the `exp_vals_traj` array, concatenates the time array `self.t` as the first row,
        and writes the resulting data to a file named `opt_traj_{self.n_eval}.txt` in the working directory.
        The file includes a header with time and observable labels.
        The output file format:
            - Each column corresponds to a time point or an observable at a specific site.
            - The first column is time (`t`).
            - Subsequent columns are labeled as `x0`, `y0`, `z0`, ..., up to the number of observed
            sites and system size.
        Attributes used:
            exp_vals_traj (np.ndarray): Array of expectation values with shape (n_obs_site, sites, n_t).
            t (np.ndarray): Array of time points.
            work_dir (str): Directory where the output file will be saved.
            n_eval (int): Evaluation index used in the output filename.
        File saved:
            {work_dir}/opt_traj_{n_eval}.txt.
        """
        n_obs, _n_t = np.shape(self.obs_array)
        exp_vals_traj_with_t = np.concatenate([np.array([self.times]), self.obs_array], axis=0)

        header = "t  " + "  ".join(["obs_" + str(i) for i in range(n_obs)])

        np.savetxt(output_file, exp_vals_traj_with_t.T, header=header, fmt="%.6f")

    def run(self, noise_model: CompactNoiseModel) -> None:
        """Run the propagation routine with augmented Lindblad-derived operators.

        Parameters
        ----------
        noise_model : CompactNoiseModel
            The compact representation of the noise model to use for propagation.
            The method verifies that the list of compact processes and their sites
            in `noise_model` match the model used to initialize this propagator
            (self.compact_noise_model). The expanded form of this model is passed
            to the underlying simulator.

        Side effects / State changes
        ----------------------------
        On successful completion, several attributes of self are set or updated:
        - self.obs_traj : list[Observable]
            The list of original observables (with their computed time trajectories)
            extracted from the simulator results.
        - self.d_on_d_gk : numpy.ndarray of shape (n_jump, n_obs) with Observable entries
            A matrix of the A_kn-like operators (or zero placeholders) corresponding
            to each jump operator / observable pair; entries are Observable objects
            whose .results have been integrated (trapezoidally) over time.
        - self.d_on_d_gk_array : numpy.ndarray
            2D array of numeric trajectories corresponding to d_on_d_gk (shape
            [n_jump, n_obs, n_timesteps]).
        - self.obs_array : numpy.ndarray
            2D array of numeric trajectories for the original observables
            (shape [n_obs, n_timesteps]).
        - self.times : array-like
            Time grid used by the simulation (copied from self.sim_params.times).

        Raises:
            ValueError: If the observable list has not been initialized (self.set_observables is False).
            ValueError: If any process name or site in the provided noise_model does not match
              the corresponding entry in self.compact_noise_model.

        Notes:
        -----
        - The purpose of the added A_kn observables is to provide sensitivity-like
          quantities (derivatives of observable expectations with respect to
          jump rates) that are computed by the same underlying simulator and then
          post-processed into arrays suitable for analysis or parameter updates.
        """
        if not self.set_observables:
            msg = "Observable list not set. Please use the set_observable_list method to set the observables."
            raise ValueError(msg)

        for i, proc in enumerate(noise_model.compact_processes):
            for j, site in enumerate(proc["sites"]):
                if (
                    proc["name"] != self.compact_noise_model.compact_processes[i]["name"]
                    or site != self.compact_noise_model.compact_processes[i]["sites"][j]
                ):
                    msg = "Noise model processes or sites do not match the initialized noise model."
                    raise ValueError(msg)

        sim_params = AnalogSimParams(
            observables=self.obs_list,
            elapsed_time=self.sim_params.elapsed_time,
            dt=self.sim_params.dt,
            num_traj=self.sim_params.num_traj,
            max_bond_dim=self.sim_params.max_bond_dim,
            threshold=self.sim_params.threshold,
            order=self.sim_params.order,
            sample_timesteps=True,
        )

        simulator.run(self.init_state, self.hamiltonian, sim_params, noise_model.expanded_noise_model)

        # Separate original and new expectation values from result_lindblad.
        self.obs_traj = sim_params.observables

        self.times = self.sim_params.times

        self.obs_array = np.array([obs.results for obs in self.obs_traj])
