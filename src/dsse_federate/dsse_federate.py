"""DSSE federate for OEDISI co-simulation.

Subscribes to voltage, power, and topology measurements from a feeder,
runs WLS state estimation, and publishes estimated voltages.
"""

import json
import logging
from datetime import datetime
from typing import Optional

import helics as h
import numpy as np
from oedisi.types.common import BrokerConfig
from oedisi.types.data_types import (
    AdmittanceMatrix,
    AdmittanceSparse,
    PowersImaginary,
    PowersReal,
    Topology,
    VoltagesAngle,
    VoltagesMagnitude,
)
from pydantic import BaseModel

from .wls_estimator import get_y, wls_estimate

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class AlgorithmParameters(BaseModel):
    """WLS algorithm configuration."""

    tol: float = 1e-4
    base_power: float = 100.0
    max_iter: int = 100


def get_indices(topology, measurement):
    """Get list of indices in the topology for each id of the input measurement."""
    inv_map = {v: i for i, v in enumerate(topology.base_voltage_magnitudes.ids)}
    return [inv_map[v] for v in measurement.ids]


class DSSEFederate:
    """DSSE federate. Wraps WLS estimator with HELICS pub/sub."""

    def __init__(
        self,
        federate_name: str,
        algorithm_parameters: AlgorithmParameters,
        input_mapping: dict,
        broker_config: BrokerConfig,
    ):
        """Initialize federate with HELICS subscriptions and publications."""
        deltat = 0.1
        self.algorithm_parameters = algorithm_parameters

        fedinfo = h.helicsCreateFederateInfo()
        h.helicsFederateInfoSetBroker(fedinfo, broker_config.broker_ip)
        h.helicsFederateInfoSetBrokerPort(fedinfo, broker_config.broker_port)
        fedinfo.core_name = federate_name
        fedinfo.core_type = h.HELICS_CORE_TYPE_ZMQ
        fedinfo.core_init = "--federates=1"
        h.helicsFederateInfoSetTimeProperty(
            fedinfo, h.helics_property_time_delta, deltat
        )

        self.vfed = h.helicsCreateValueFederate(federate_name, fedinfo)
        logger.info("Value federate created")

        self.sub_voltages_magnitude = self.vfed.register_subscription(
            input_mapping["voltages_magnitude"], "V"
        )
        self.sub_power_P = self.vfed.register_subscription(
            input_mapping["powers_real"], "W"
        )
        self.sub_power_Q = self.vfed.register_subscription(
            input_mapping["powers_imaginary"], "W"
        )
        self.sub_topology = self.vfed.register_subscription(
            input_mapping["topology"], ""
        )
        self.pub_voltage_mag = self.vfed.register_publication(
            "voltage_mag", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_voltage_angle = self.vfed.register_publication(
            "voltage_angle", h.HELICS_DATA_TYPE_STRING, ""
        )

    def run(self):
        """Enter execution and exchange data."""
        self.vfed.enter_executing_mode()
        logger.info("Entering execution mode")

        granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)

        topology = Topology.model_validate(self.sub_topology.json)
        ids = topology.base_voltage_magnitudes.ids
        base_voltages = np.array(topology.base_voltage_magnitudes.values)
        logger.info("Topology received: %d buses", len(ids))

        if not isinstance(topology.admittance, (AdmittanceMatrix, AdmittanceSparse)):
            raise ValueError(
                "WLS algorithm expects AdmittanceMatrix or AdmittanceSparse"
            )

        slack_index = None
        for i, bus_id in enumerate(ids):
            if bus_id == topology.slack_bus[0]:
                slack_index = i
                break
        if slack_index is None:
            logger.warning(
                "Slack bus '%s' not found in topology ids, defaulting to 0",
                topology.slack_bus[0] if topology.slack_bus else "N/A",
            )
            slack_index = 0

        Y_bus = get_y(topology.admittance, ids)

        initial_angles: Optional[np.ndarray] = None
        if topology.base_voltage_angles is not None:
            initial_angles = np.array(topology.base_voltage_angles.values)

        while granted_time < h.HELICS_TIME_MAXTIME:
            if not self.sub_voltages_magnitude.is_updated():
                granted_time = h.helicsFederateRequestTime(
                    self.vfed, h.HELICS_TIME_MAXTIME
                )
                continue

            logger.info("Timestep start: %s", str(datetime.now()))

            voltages = VoltagesMagnitude.model_validate(
                self.sub_voltages_magnitude.json
            )
            power_P = PowersReal.model_validate(self.sub_power_P.json)
            power_Q = PowersImaginary.model_validate(self.sub_power_Q.json)

            voltage_indices = np.array(get_indices(topology, voltages), dtype=int)

            # Map power measurements to bus indices (partial, not all buses)
            inv_map = {v: i for i, v in enumerate(ids)}
            p_idx_list = []
            p_values = []
            for j, pid in enumerate(power_P.ids):
                if pid in inv_map:
                    p_idx_list.append(inv_map[pid])
                    p_values.append(power_P.values[j])
            q_idx_list = []
            q_values = []
            for j, qid in enumerate(power_Q.ids):
                if qid in inv_map:
                    q_idx_list.append(inv_map[qid])
                    q_values.append(power_Q.values[j])

            p_indices = np.array(p_idx_list, dtype=int)
            q_indices = np.array(q_idx_list, dtype=int)

            voltage_magnitudes, voltage_angles, n_iter = wls_estimate(
                Y_bus=Y_bus,
                base_voltages=base_voltages,
                power_p=np.array(p_values),
                power_q=np.array(q_values),
                voltage_meas=np.array(voltages.values),
                voltage_indices=voltage_indices,
                slack_index=slack_index,
                base_power=self.algorithm_parameters.base_power,
                tol=self.algorithm_parameters.tol,
                max_iter=self.algorithm_parameters.max_iter,
                initial_angles=initial_angles,
                p_indices=p_indices,
                q_indices=q_indices,
            )

            logger.info("WLS converged in %d iterations", n_iter)

            self.pub_voltage_mag.publish(
                VoltagesMagnitude(
                    values=voltage_magnitudes.flatten().tolist(),
                    ids=ids,
                    time=voltages.time,
                ).model_dump_json()
            )
            self.pub_voltage_angle.publish(
                VoltagesAngle(
                    values=voltage_angles.flatten().tolist(),
                    ids=ids,
                    time=voltages.time,
                ).model_dump_json()
            )

            logger.info("Timestep end: %s", str(datetime.now()))

        self.destroy()

    def destroy(self):
        """Finalize and destroy the federate."""
        h.helicsFederateDisconnect(self.vfed)
        logger.info("Federate disconnected")
        h.helicsFederateFree(self.vfed)
        h.helicsCloseLibrary()


def run_simulator(broker_config: BrokerConfig):
    """Entry point called by server or CLI."""
    with open("static_inputs.json") as f:
        config = json.load(f)
        federate_name = config["name"]
        if "algorithm_parameters" in config:
            parameters = AlgorithmParameters.model_validate(
                config["algorithm_parameters"]
            )
        else:
            parameters = AlgorithmParameters.model_validate({})

    with open("input_mapping.json") as f:
        input_mapping = json.load(f)

    sfed = DSSEFederate(federate_name, parameters, input_mapping, broker_config)
    sfed.run()


if __name__ == "__main__":
    run_simulator(BrokerConfig(broker_ip="127.0.0.1"))
