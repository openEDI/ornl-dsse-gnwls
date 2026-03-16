"""Weighted Least Squares (WLS) state estimator for distribution systems.

Uses polar coordinates (V, delta) as state variables.
Measurement types: power injections (P, Q) and voltage magnitudes (V).
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.linalg
import scipy.sparse
from oedisi.types.data_types import (
    AdmittanceMatrix,
    AdmittanceSparse,
    Complex,
)


def matrix_to_numpy(admittance: List[List[Complex]]) -> np.ndarray:
    """Convert list-of-list Complex type into a numpy matrix."""
    return np.array([[x[0] + 1j * x[1] for x in row] for row in admittance])


def get_y(
    admittance: Union[AdmittanceMatrix, AdmittanceSparse], ids: List[str]
) -> np.ndarray:
    """Extract Y-bus matrix from topology admittance data."""
    if isinstance(admittance, AdmittanceMatrix):
        assert ids == admittance.ids
        return matrix_to_numpy(admittance.admittance_matrix)
    elif isinstance(admittance, AdmittanceSparse):
        node_map = {name: i for (i, name) in enumerate(ids)}
        return scipy.sparse.coo_matrix(
            (
                [v[0] + 1j * v[1] for v in admittance.admittance_list],
                (
                    [node_map[r] for r in admittance.from_equipment],
                    [node_map[c] for c in admittance.to_equipment],
                ),
            ),
            shape=(len(ids), len(ids)),
        ).toarray()
    else:
        raise ValueError(
            f"Unsupported admittance type: {type(admittance)}. "
            "Expected AdmittanceMatrix or AdmittanceSparse."
        )


def _compute_power_injections(
    V: np.ndarray,
    delta: np.ndarray,
    G: np.ndarray,
    B: np.ndarray,
    bus_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute P and Q injection equations h(x) for given bus indices."""
    nbus = len(V)
    n_meas = len(bus_indices)
    h_p = np.zeros(n_meas)
    h_q = np.zeros(n_meas)

    for i in range(n_meas):
        m = bus_indices[i]
        for k in range(nbus):
            angle_diff = delta[m] - delta[k]
            h_p[i] += (
                V[m]
                * V[k]
                * (G[m, k] * np.cos(angle_diff) + B[m, k] * np.sin(angle_diff))
            )
            h_q[i] += (
                V[m]
                * V[k]
                * (G[m, k] * np.sin(angle_diff) - B[m, k] * np.cos(angle_diff))
            )

    return h_p, h_q


def _compute_power_jacobians(
    V: np.ndarray,
    delta: np.ndarray,
    G: np.ndarray,
    B: np.ndarray,
    p_indices: np.ndarray,
    q_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Jacobian sub-matrices for power injection measurements.

    Returns (joc_pn, joc_qn) where each is [n_meas x 2*nbus].
    Column layout: [dh/dV | dh/ddelta].
    """
    nbus = len(V)
    npi = len(p_indices)
    nqi = len(q_indices)

    # dP/ddelta (H21)
    H21 = np.zeros((npi, nbus))
    for i in range(npi):
        m = p_indices[i]
        for k in range(nbus):
            angle_diff = delta[m] - delta[k]
            if k == m:
                for n in range(nbus):
                    ad = delta[m] - delta[n]
                    H21[i, k] += (
                        V[m] * V[n] * (-G[m, n] * np.sin(ad) + B[m, n] * np.cos(ad))
                    )
                H21[i, k] -= V[m] ** 2 * B[m, m]
            else:
                H21[i, k] = (
                    V[m]
                    * V[k]
                    * (G[m, k] * np.sin(angle_diff) - B[m, k] * np.cos(angle_diff))
                )

    # dP/dV (H22)
    H22 = np.zeros((npi, nbus))
    for i in range(npi):
        m = p_indices[i]
        for k in range(nbus):
            angle_diff = delta[m] - delta[k]
            if k == m:
                for n in range(nbus):
                    ad = delta[m] - delta[n]
                    H22[i, k] += V[n] * (G[m, n] * np.cos(ad) + B[m, n] * np.sin(ad))
                H22[i, k] += V[m] * G[m, m]
            else:
                H22[i, k] = V[m] * (
                    G[m, k] * np.cos(angle_diff) + B[m, k] * np.sin(angle_diff)
                )

    # dQ/ddelta (H31)
    H31 = np.zeros((nqi, nbus))
    for i in range(nqi):
        m = q_indices[i]
        for k in range(nbus):
            angle_diff = delta[m] - delta[k]
            if k == m:
                for n in range(nbus):
                    ad = delta[m] - delta[n]
                    H31[i, k] += (
                        V[m] * V[n] * (G[m, n] * np.cos(ad) + B[m, n] * np.sin(ad))
                    )
                H31[i, k] -= V[m] ** 2 * G[m, m]
            else:
                H31[i, k] = (
                    V[m]
                    * V[k]
                    * (-G[m, k] * np.cos(angle_diff) - B[m, k] * np.sin(angle_diff))
                )

    # dQ/dV (H32)
    H32 = np.zeros((nqi, nbus))
    for i in range(nqi):
        m = q_indices[i]
        for k in range(nbus):
            angle_diff = delta[m] - delta[k]
            if k == m:
                for n in range(nbus):
                    ad = delta[m] - delta[n]
                    H32[i, k] += V[n] * (G[m, n] * np.sin(ad) - B[m, n] * np.cos(ad))
                H32[i, k] -= V[m] * B[m, m]
            else:
                H32[i, k] = V[m] * (
                    G[m, k] * np.sin(angle_diff) - B[m, k] * np.cos(angle_diff)
                )

    # Stack as [dh/dV | dh/ddelta]
    joc_pn = np.hstack((H22, H21))
    joc_qn = np.hstack((H32, H31))
    return joc_pn, joc_qn


def _compute_voltage_jacobian(voltage_indices: np.ndarray, nbus: int) -> np.ndarray:
    """Compute Jacobian for voltage magnitude measurements.

    dV/dV = 1 at the measured bus, 0 elsewhere. dV/ddelta = 0.
    Returns [n_voltage_meas x 2*nbus].
    """
    n_voltage_meas = len(voltage_indices)
    H_dv = np.zeros((n_voltage_meas, nbus))
    for i, bus_idx in enumerate(voltage_indices):
        H_dv[i, bus_idx] = 1.0
    H_ddelta = np.zeros((n_voltage_meas, nbus))
    return np.hstack((H_dv, H_ddelta))


def _find_zero_rows_cols(A: np.ndarray) -> Tuple[List[int], List[int]]:
    """Find rows and columns that are all zeros (within tolerance)."""
    N = A.shape[0]
    zero_rows = []
    zero_cols = []
    for i in range(N):
        if np.all(np.abs(A[i, :]) < 1e-6):
            zero_rows.append(i)
        if np.all(np.abs(A[:, i]) < 1e-6):
            zero_cols.append(i)
    return zero_rows, zero_cols


def wls_estimate(
    Y_bus: np.ndarray,
    base_voltages: np.ndarray,
    power_p: np.ndarray,
    power_q: np.ndarray,
    voltage_meas: np.ndarray,
    voltage_indices: np.ndarray,
    slack_index: int,
    base_power: float = 100.0,
    tol: float = 1e-4,
    max_iter: int = 100,
    sensor_error_pq: float = 0.1,
    sensor_error_v: float = 0.001,
    initial_v: Optional[float] = None,
    initial_angles: Optional[np.ndarray] = None,
    p_indices: Optional[np.ndarray] = None,
    q_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:

    nbus = Y_bus.shape[0]

    # Convert Y-bus to per-unit: Y_pu = Y * V_base_i * V_base_j / S_base_VA
    S_base_VA = base_power * 1e6
    Y_pu = (
        base_voltages.reshape(1, -1) * Y_bus * base_voltages.reshape(-1, 1) / S_base_VA
    )
    G = np.real(Y_pu)
    B = np.imag(Y_pu)

    # Convert power to per-unit: P_pu = P_kW / S_base_kW
    S_base_kW = base_power * 1000.0
    P_pu = power_p / S_base_kW
    Q_pu = power_q / S_base_kW

    # Convert voltage measurements to per-unit
    V_meas_pu = voltage_meas / base_voltages[voltage_indices]

    # Use provided measurement indices, or default to all buses
    if p_indices is None:
        p_indices = np.arange(nbus, dtype=int)
    if q_indices is None:
        q_indices = np.arange(nbus, dtype=int)

    z = np.concatenate([P_pu, Q_pu, V_meas_pu])

    # Build weight matrix W = diag(1/sigma^2) for proper WLS weighting
    n_p = len(p_indices)
    n_q = len(q_indices)
    n_v = len(voltage_indices)
    n_meas = n_p + n_q + n_v
    w_pq = 1.0 / (sensor_error_pq**2) if sensor_error_pq > 0 else 1.0
    w_v = 1.0 / (sensor_error_v**2) if sensor_error_v > 0 else 1.0
    W = np.diag(
        np.concatenate(
            [
                np.full(n_p, w_pq),
                np.full(n_q, w_pq),
                np.full(n_v, w_v),
            ]
        )
    )

    # Initial state: [V_1..V_n, delta_1..delta_n] in per-unit/radians
    if initial_v is None:
        initial_v = 1.0
    V = np.full(nbus, initial_v)
    if initial_angles is not None:
        delta = initial_angles.copy()
    else:
        delta = np.zeros(nbus)

    n_iterations = 0
    for iteration in range(max_iter):
        state = np.concatenate([V, delta])

        h_p, h_q = _compute_power_injections(V, delta, G, B, p_indices)
        h_v = V[voltage_indices]
        h_x = np.concatenate([h_p, h_q, h_v])

        joc_pn, joc_qn = _compute_power_jacobians(V, delta, G, B, p_indices, q_indices)
        joc_vn = _compute_voltage_jacobian(voltage_indices, nbus)

        H = np.vstack([joc_pn, joc_qn, joc_vn])

        H[:, nbus + slack_index] = 0.0

        delta_z = z - h_x

        # Normal equations: (H^T W H) dx = H^T W dz
        G2 = H.T @ W @ delta_z
        G3 = H.T @ W @ H

        # Handle zero rows/cols (unobservable states)
        size_G3 = G3.shape[0]
        zero_rows, zero_cols = _find_zero_rows_cols(G3)

        G3_reduced = np.delete(G3, zero_rows, axis=0)
        G3_reduced = np.delete(G3_reduced, zero_cols, axis=1)

        G3_inv = scipy.linalg.pinv(G3_reduced)

        non_zero_idx = np.setdiff1d(np.arange(size_G3), zero_rows)
        G3_full_inv = np.zeros((size_G3, size_G3))
        G3_full_inv[np.ix_(non_zero_idx, non_zero_idx)] = G3_inv

        delta_x = G3_full_inv @ G2

        tol_current = np.max(np.abs(delta_x))
        n_iterations = iteration + 1

        if tol_current < tol:
            break

        V = V + delta_x[:nbus]
        delta = delta + delta_x[nbus:]

    voltage_magnitudes = V * base_voltages
    voltage_angles_deg = np.degrees(delta)
    voltage_angles_deg[V < 1e-4] = 0.0

    return voltage_magnitudes, voltage_angles_deg, n_iterations
