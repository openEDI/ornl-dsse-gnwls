"""Tests for WLS state estimator."""

import numpy as np
import pytest
from dsse_federate.wls_estimator import (
    _compute_power_injections,
    _compute_power_jacobians,
    _compute_voltage_jacobian,
    _find_zero_rows_cols,
    get_y,
    wls_estimate,
)
from oedisi.types.data_types import AdmittanceMatrix, AdmittanceSparse


def _make_3bus_ybus():
    """Create a simple 3-bus Y-bus matrix for testing.

    Topology: bus 0 (slack) -- bus 1 -- bus 2
    Line impedances: z01 = 0.01 + j0.05, z12 = 0.02 + j0.06
    """
    z01 = 0.01 + 0.05j
    z12 = 0.02 + 0.06j
    y01 = 1.0 / z01
    y12 = 1.0 / z12

    Y = np.zeros((3, 3), dtype=complex)
    Y[0, 0] = y01
    Y[0, 1] = -y01
    Y[1, 0] = -y01
    Y[1, 1] = y01 + y12
    Y[1, 2] = -y12
    Y[2, 1] = -y12
    Y[2, 2] = y12
    return Y


def _solve_power_flow_3bus(Y, base_voltages, P_load, Q_load, slack_index=0):
    """Simple Newton-Raphson power flow for 3-bus test case.

    Returns voltage magnitudes (V) and angles (rad) as ground truth.
    """
    nbus = 3
    V = np.ones(nbus)
    delta = np.zeros(nbus)

    # Convert to per-unit
    base_power = 100.0
    Y_pu = (
        base_voltages.reshape(1, -1)
        * Y
        * base_voltages.reshape(-1, 1)
        / (base_power * 1000)
    )
    G = np.real(Y_pu)
    B = np.imag(Y_pu)
    P_pu = P_load / base_power
    Q_pu = Q_load / base_power

    pq_buses = [i for i in range(nbus) if i != slack_index]

    for _ in range(50):
        # Compute mismatches
        dP = np.zeros(len(pq_buses))
        dQ = np.zeros(len(pq_buses))
        for idx, m in enumerate(pq_buses):
            p_calc = 0.0
            q_calc = 0.0
            for k in range(nbus):
                ad = delta[m] - delta[k]
                p_calc += V[m] * V[k] * (G[m, k] * np.cos(ad) + B[m, k] * np.sin(ad))
                q_calc += V[m] * V[k] * (G[m, k] * np.sin(ad) - B[m, k] * np.cos(ad))
            dP[idx] = P_pu[m] - p_calc
            dQ[idx] = Q_pu[m] - q_calc

        if np.max(np.abs(np.concatenate([dP, dQ]))) < 1e-10:
            break

        # Build Jacobian (simplified for PQ buses only)
        n_pq = len(pq_buses)
        J = np.zeros((2 * n_pq, 2 * n_pq))

        for i, m in enumerate(pq_buses):
            for j, k in enumerate(pq_buses):
                ad = delta[m] - delta[k]
                if m == k:
                    # dP/ddelta
                    s = 0.0
                    for n in range(nbus):
                        if n != m:
                            adn = delta[m] - delta[n]
                            s += (
                                V[m]
                                * V[n]
                                * (-G[m, n] * np.sin(adn) + B[m, n] * np.cos(adn))
                            )
                    J[i, j] = s
                    # dP/dV
                    s2 = 0.0
                    for n in range(nbus):
                        adn = delta[m] - delta[n]
                        s2 += V[n] * (G[m, n] * np.cos(adn) + B[m, n] * np.sin(adn))
                    J[i, n_pq + j] = s2 + V[m] * G[m, m]
                    # dQ/ddelta
                    s3 = 0.0
                    for n in range(nbus):
                        if n != m:
                            adn = delta[m] - delta[n]
                            s3 += (
                                V[m]
                                * V[n]
                                * (G[m, n] * np.cos(adn) + B[m, n] * np.sin(adn))
                            )
                    J[n_pq + i, j] = s3
                    # dQ/dV
                    s4 = 0.0
                    for n in range(nbus):
                        adn = delta[m] - delta[n]
                        s4 += V[n] * (G[m, n] * np.sin(adn) - B[m, n] * np.cos(adn))
                    J[n_pq + i, n_pq + j] = s4 - V[m] * B[m, m]
                else:
                    J[i, j] = (
                        V[m] * V[k] * (G[m, k] * np.sin(ad) - B[m, k] * np.cos(ad))
                    )
                    J[i, n_pq + j] = V[m] * (
                        G[m, k] * np.cos(ad) + B[m, k] * np.sin(ad)
                    )
                    J[n_pq + i, j] = (
                        V[m] * V[k] * (-G[m, k] * np.cos(ad) - B[m, k] * np.sin(ad))
                    )
                    J[n_pq + i, n_pq + j] = V[m] * (
                        G[m, k] * np.sin(ad) - B[m, k] * np.cos(ad)
                    )

        dx = np.linalg.solve(J, np.concatenate([dP, dQ]))
        for idx, m in enumerate(pq_buses):
            delta[m] += dx[idx]
            V[m] += dx[n_pq + idx]

    return V * base_voltages, delta


class TestWLSConvergence:
    """Test WLS convergence with a known 3-bus system."""

    def setup_method(self):
        self.Y = _make_3bus_ybus()
        self.base_voltages = np.array([2401.7, 2401.7, 2401.7])
        self.slack_index = 0

    def test_flat_start_convergence(self):
        """WLS should converge from flat start with perfect measurements."""
        P_load = np.array([0.0, -50.0, -30.0])
        Q_load = np.array([0.0, -20.0, -15.0])

        # Get ground truth from power flow
        V_true, delta_true = _solve_power_flow_3bus(
            self.Y, self.base_voltages, P_load, Q_load, self.slack_index
        )

        # Use all buses as voltage measurements
        voltage_indices = np.array([0, 1, 2], dtype=int)
        voltage_meas = V_true.copy()

        V_est, angle_est, n_iter = wls_estimate(
            Y_bus=self.Y,
            base_voltages=self.base_voltages,
            power_p=P_load,
            power_q=Q_load,
            voltage_meas=voltage_meas,
            voltage_indices=voltage_indices,
            slack_index=self.slack_index,
            tol=1e-6,
        )

        assert n_iter < 100, "WLS did not converge"
        np.testing.assert_allclose(V_est, V_true, rtol=1e-3)

    def test_partial_voltage_measurements(self):
        """WLS should work with voltage measurements at a subset of buses."""
        P_load = np.array([0.0, -50.0, -30.0])
        Q_load = np.array([0.0, -20.0, -15.0])

        V_true, delta_true = _solve_power_flow_3bus(
            self.Y, self.base_voltages, P_load, Q_load, self.slack_index
        )

        # Only measure voltage at slack bus
        voltage_indices = np.array([0], dtype=int)
        voltage_meas = V_true[[0]]

        V_est, angle_est, n_iter = wls_estimate(
            Y_bus=self.Y,
            base_voltages=self.base_voltages,
            power_p=P_load,
            power_q=Q_load,
            voltage_meas=voltage_meas,
            voltage_indices=voltage_indices,
            slack_index=self.slack_index,
            tol=1e-6,
        )

        assert n_iter < 100, "WLS did not converge"
        np.testing.assert_allclose(V_est, V_true, rtol=1e-2)

    def test_convergence_tolerance(self):
        """Tighter tolerance should require more iterations."""
        P_load = np.array([0.0, -50.0, -30.0])
        Q_load = np.array([0.0, -20.0, -15.0])

        V_true, _ = _solve_power_flow_3bus(
            self.Y, self.base_voltages, P_load, Q_load, self.slack_index
        )

        voltage_indices = np.array([0, 1, 2], dtype=int)
        voltage_meas = V_true.copy()

        _, _, n_iter_loose = wls_estimate(
            Y_bus=self.Y,
            base_voltages=self.base_voltages,
            power_p=P_load,
            power_q=Q_load,
            voltage_meas=voltage_meas,
            voltage_indices=voltage_indices,
            slack_index=self.slack_index,
            tol=1e-2,
        )

        _, _, n_iter_tight = wls_estimate(
            Y_bus=self.Y,
            base_voltages=self.base_voltages,
            power_p=P_load,
            power_q=Q_load,
            voltage_meas=voltage_meas,
            voltage_indices=voltage_indices,
            slack_index=self.slack_index,
            tol=1e-8,
        )

        assert n_iter_loose <= n_iter_tight


class TestSlackBusConstraint:
    """Test that slack bus angle is held fixed."""

    def test_slack_angle_zero(self):
        """Slack bus angle should remain at zero."""
        Y = _make_3bus_ybus()
        base_voltages = np.array([2401.7, 2401.7, 2401.7])
        P_load = np.array([0.0, -50.0, -30.0])
        Q_load = np.array([0.0, -20.0, -15.0])

        V_true, _ = _solve_power_flow_3bus(
            Y, base_voltages, P_load, Q_load, slack_index=0
        )

        voltage_indices = np.array([0, 1, 2], dtype=int)

        V_est, angle_est, _ = wls_estimate(
            Y_bus=Y,
            base_voltages=base_voltages,
            power_p=P_load,
            power_q=Q_load,
            voltage_meas=V_true,
            voltage_indices=voltage_indices,
            slack_index=0,
            tol=1e-6,
        )

        assert abs(angle_est[0]) < 1e-6, "Slack bus angle should be ~0"

    def test_different_slack_bus(self):
        """WLS should work with a non-zero slack bus index."""
        Y = _make_3bus_ybus()
        base_voltages = np.array([2401.7, 2401.7, 2401.7])
        P_load = np.array([-30.0, 0.0, -50.0])
        Q_load = np.array([-15.0, 0.0, -20.0])

        V_true, _ = _solve_power_flow_3bus(
            Y, base_voltages, P_load, Q_load, slack_index=1
        )

        voltage_indices = np.array([0, 1, 2], dtype=int)

        V_est, angle_est, n_iter = wls_estimate(
            Y_bus=Y,
            base_voltages=base_voltages,
            power_p=P_load,
            power_q=Q_load,
            voltage_meas=V_true,
            voltage_indices=voltage_indices,
            slack_index=1,
            tol=1e-6,
        )

        assert n_iter < 100, "WLS did not converge with slack at bus 1"
        assert abs(angle_est[1]) < 1e-6, "Slack bus 1 angle should be ~0"


class TestYBusExtraction:
    """Test Y-bus extraction from topology admittance types."""

    def test_admittance_matrix_extraction(self):
        """Extract Y-bus from AdmittanceMatrix format."""
        Y_expected = _make_3bus_ybus()
        ids = ["bus0", "bus1", "bus2"]

        admittance_list = []
        for row in Y_expected:
            row_list = []
            for val in row:
                row_list.append([float(val.real), float(val.imag)])
            admittance_list.append(row_list)

        adm = AdmittanceMatrix(admittance_matrix=admittance_list, ids=ids)

        Y_result = get_y(adm, ids)
        np.testing.assert_allclose(Y_result, Y_expected, atol=1e-10)

    def test_admittance_sparse_extraction(self):
        """Extract Y-bus from AdmittanceSparse format."""
        Y_expected = _make_3bus_ybus()
        ids = ["bus0", "bus1", "bus2"]

        from_eq = []
        to_eq = []
        adm_list = []
        for i in range(3):
            for j in range(3):
                if abs(Y_expected[i, j]) > 1e-10:
                    from_eq.append(ids[i])
                    to_eq.append(ids[j])
                    adm_list.append(
                        [float(Y_expected[i, j].real), float(Y_expected[i, j].imag)]
                    )

        adm = AdmittanceSparse(
            from_equipment=from_eq,
            to_equipment=to_eq,
            admittance_list=adm_list,
        )

        Y_result = get_y(adm, ids)
        np.testing.assert_allclose(Y_result, Y_expected, atol=1e-10)


class TestVoltageAngleEstimation:
    """Test voltage angle estimation accuracy."""

    def test_nonzero_angles(self):
        """WLS should estimate nonzero angles for loaded buses."""
        Y = _make_3bus_ybus()
        base_voltages = np.array([2401.7, 2401.7, 2401.7])
        P_load = np.array([0.0, -80.0, -60.0])
        Q_load = np.array([0.0, -40.0, -30.0])

        V_true, delta_true = _solve_power_flow_3bus(
            Y, base_voltages, P_load, Q_load, slack_index=0
        )

        voltage_indices = np.array([0, 1, 2], dtype=int)

        V_est, angle_est, _ = wls_estimate(
            Y_bus=Y,
            base_voltages=base_voltages,
            power_p=P_load,
            power_q=Q_load,
            voltage_meas=V_true,
            voltage_indices=voltage_indices,
            slack_index=0,
            tol=1e-6,
        )

        angle_true_deg = np.degrees(delta_true)
        np.testing.assert_allclose(angle_est, angle_true_deg, atol=0.5)


class TestHelperFunctions:
    """Test internal helper functions."""

    def test_find_zero_rows_cols(self):
        """Correctly identify zero rows and columns."""
        A = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        )
        zero_rows, zero_cols = _find_zero_rows_cols(A)
        assert 1 in zero_rows
        assert 1 in zero_cols

    def test_voltage_jacobian_shape(self):
        """Voltage Jacobian should have correct dimensions and map to correct buses."""
        voltage_indices = np.array([0, 2, 4])
        jac = _compute_voltage_jacobian(voltage_indices, nbus=5)
        assert jac.shape == (3, 10)
        assert jac[0, 0] == 1.0
        assert jac[1, 2] == 1.0
        assert jac[2, 4] == 1.0
        assert jac[0, 1] == 0.0

    def test_power_injections_zero_load(self):
        """Zero load should produce zero power injections."""
        nbus = 3
        V = np.ones(nbus)
        delta = np.zeros(nbus)
        G = np.zeros((nbus, nbus))
        B = np.zeros((nbus, nbus))
        indices = np.array([0, 1, 2])

        h_p, h_q = _compute_power_injections(V, delta, G, B, indices)
        np.testing.assert_allclose(h_p, 0.0, atol=1e-10)
        np.testing.assert_allclose(h_q, 0.0, atol=1e-10)
