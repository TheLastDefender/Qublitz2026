"""Qublitz 2026 - Quantum Circuit Golf participant starter.

Install:  python -m pip install numpy
Try it:   python submission.py

Organizers will import `optimize(circuit, config)` from this file.
Change the body of `optimize` to implement your own algorithm.

Circuit format:
    {"n_qubits": 2, "gates": [{"gate": "H", "qubits": [0]}, ...]}

Supported gates: H, X, Z, S, SDG, T, TDG, CX.
CX qubit order: [control, target]. Qubit 0 is the top/MSB wire.
Gates execute in listed order. One through six qubits are supported.

Config format:
    {"w1": 1.0, "w2": 6.0, "wd": 0.2}

This includes a LOCAL checker, not the organizers' private tests.
"""

import copy
import math

import numpy as np


SINGLE_QUBIT_GATES = frozenset(("H", "X", "Z", "S", "SDG", "T", "TDG"))
ALLOWED_GATES = SINGLE_QUBIT_GATES | {"CX"}


def optimize(circuit, config):
    """Optimizes a quantum circuit to reduce its cost according to the given weights.

    Inputs:
        circuit: A dictionary representing the quantum circuit to be optimized.
        config: A dictionary containing the weights for one-qubit gates, two-qubit 
        gates, and circuit depth.
    """
    validate_circuit(circuit)
    validate_config(config)

    # TODO: Implement your optimization algorithm here.
    return copy.deepcopy(circuit)


def validate_circuit(circuit):
    """Raise ValueError if a circuit violates the public format."""
    if not isinstance(circuit, dict) or set(circuit) != {"n_qubits", "gates"}:
        raise ValueError("Circuit must have exactly 'n_qubits' and 'gates'.")

    n = circuit["n_qubits"]
    if type(n) is not int or not 1 <= n <= 6:
        raise ValueError("n_qubits must be an integer from 1 through 6.")

    gates = circuit["gates"]
    if not isinstance(gates, list):
        raise ValueError("gates must be a list.")

    for i, operation in enumerate(gates):
        if not isinstance(operation, dict) or set(operation) != {"gate", "qubits"}:
            raise ValueError(f"Gate {i}: expected exactly 'gate' and 'qubits'.")

        name = operation["gate"]
        qubits = operation["qubits"]
        if not isinstance(name, str) or name not in ALLOWED_GATES:
            raise ValueError(f"Gate {i}: unsupported gate {name!r}.")
        if not isinstance(qubits, list):
            raise ValueError(f"Gate {i}: qubits must be a list.")

        arity = 2 if name == "CX" else 1
        if len(qubits) != arity:
            raise ValueError(f"Gate {i}: {name} requires {arity} qubit(s).")
        if any(type(q) is not int or not 0 <= q < n for q in qubits):
            raise ValueError(f"Gate {i}: invalid qubit index.")
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"Gate {i}: qubit indices must be distinct.")

    return True


def validate_config(config):
    """Check nonnegative finite resource weights w1, w2, wd."""
    if not isinstance(config, dict) or set(config) != {"w1", "w2", "wd"}:
        raise ValueError("config must have exactly 'w1', 'w2', and 'wd'.")

    for key in ("w1", "w2", "wd"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a real number.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")

    if not any(config[key] > 0 for key in ("w1", "w2", "wd")):
        raise ValueError("At least one weight must be positive.")

    return True


_ROOT2 = math.sqrt(2)
_ONE_QUBIT_MATRICES = {
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / _ROOT2,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "S": np.diag([1, 1j]).astype(complex),
    "SDG": np.diag([1, -1j]).astype(complex),
    "T": np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex),
    "TDG": np.diag([1, np.exp(-1j * np.pi / 4)]).astype(complex),
}


def _full_gate_matrix(n, operation):
    """Matrix for one gate, with qubit 0 as the most significant bit."""
    name = operation["gate"]
    qubits = operation["qubits"]
    if name != "CX":
        result = np.array([[1]], dtype=complex)
        for q in range(n):
            factor = (_ONE_QUBIT_MATRICES[name]
                      if q == qubits[0] else np.eye(2, dtype=complex))
            result = np.kron(result, factor)
        return result

    control, target = qubits
    dimension = 1 << n
    result = np.zeros((dimension, dimension), dtype=complex)
    control_bit = 1 << (n - 1 - control)
    target_bit = 1 << (n - 1 - target)
    for input_state in range(dimension):
        output_state = (input_state ^ target_bit
                        if input_state & control_bit else input_state)
        result[output_state, input_state] = 1
    return result


def circuit_unitary(circuit):
    """Calculate the complete unitary for a supported 1-6 qubit circuit.

    Earlier listed gates act first: [A, B] has unitary U_B @ U_A.
    """
    validate_circuit(circuit)
    n = circuit["n_qubits"]
    result = np.eye(1 << n, dtype=complex)
    for operation in circuit["gates"]:
        result = _full_gate_matrix(n, operation) @ result
    return result


def equivalent_up_to_global_phase(first, second, atol=1e-9):
    """Return True if both circuits implement the same unitary up to phase."""
    validate_circuit(first)
    validate_circuit(second)
    if first["n_qubits"] != second["n_qubits"]:
        return False

    a = circuit_unitary(first)
    b = circuit_unitary(second)
    pivot = np.unravel_index(np.argmax(np.abs(a)), a.shape)
    phase = b[pivot] / a[pivot]
    return (np.isclose(abs(phase), 1.0, atol=atol, rtol=0.0)
            and np.allclose(b, phase * a, atol=atol, rtol=0.0))


def circuit_metrics(circuit):
    """Count gates and minimum depth, scheduling disjoint gates in parallel.

    Every gate takes one time unit. Gates sharing a qubit are serialized.
    """
    validate_circuit(circuit)
    available = [0] * circuit["n_qubits"]
    one_qubit = 0
    two_qubit = 0

    for operation in circuit["gates"]:
        qubits = operation["qubits"]
        end = 1 + max(available[q] for q in qubits)
        for q in qubits:
            available[q] = end
        if operation["gate"] == "CX":
            two_qubit += 1
        else:
            one_qubit += 1

    return {"n_1q": one_qubit, "n_2q": two_qubit, "depth": max(available)}


def circuit_cost(circuit, config):
    """Weighted cost C = w1*N_1q + w2*N_2q + wd*depth."""
    validate_config(config)
    metrics = circuit_metrics(circuit)
    return (config["w1"] * metrics["n_1q"]
            + config["w2"] * metrics["n_2q"]
            + config["wd"] * metrics["depth"])


def evaluate_optimization(original, optimized, config):
    """Return local correctness and resource comparisons for one input."""
    equivalent = equivalent_up_to_global_phase(original, optimized)
    before = circuit_cost(original, config)
    after = circuit_cost(optimized, config)
    return {
        "equivalent": equivalent,
        "input_metrics": circuit_metrics(original),
        "output_metrics": circuit_metrics(optimized),
        "input_cost": before,
        "output_cost": after,
        "cost_reduction": before - after,
    }


if __name__ == "__main__":
    example = {
        "n_qubits": 2,
        "gates": [
            {"gate": "CX", "qubits": [0, 1]},
            {"gate": "Z", "qubits": [0]},
            {"gate": "CX", "qubits": [0, 1]},
        ],
    }
    weights = {"w1": 1.0, "w2": 6.0, "wd": 0.2}
    result = optimize(example, weights)
    print("Your optimizer (initially an unchanged copy):")
    print(result)
    print("Local assessment:")
    print(evaluate_optimization(example, result, weights))

    # A known equivalent circuit to demonstrate the checker:
    expected = {"n_qubits": 2, "gates": [{"gate": "Z", "qubits": [0]}]}
    print("Known simplification (CX-Z-CX -> Z):")
    print(evaluate_optimization(example, expected, weights))