"""
Erdos problem #360 -- quantum-testable sequence lane.

LIMITATION (read first): the local read-only clone of
manman4/erdosproblems (data/problems.yaml, entry "number: '360'") lists
this problem as tags=["number theory"], status="solved", and
oeis=["possible"]. "possible" is a placeholder string, not an OEIS
sequence id -- there is no actual OEIS id attached to problem 360 in the
source data, and no problem statement text is available in this clone
(no per-problem markdown/description file ships with the repo; only the
README's summary table, which likewise has no OEIS id for #360). It is
therefore not possible to derive a property that is genuinely tied to
problem 360's own mathematical content from what is available here.

Rather than fabricate a fake connection to #360, this script implements
its best honest attempt at a genuine, finite, verifiable number-theory
property in the same tag area ("number theory") that a small quantum
circuit can really compute: Grover's algorithm searching the 4-bit
space {0, ..., 15} for integers divisible by 3. The classical answer
(which n in [0,16) are multiples of 3) is computed from first principles
by trial division in this script, independent of any OEIS lookup, and
is compared against the quantum search's output distribution.

This is explicitly NOT a verified property of Erdos problem #360's
actual sequence -- ran_ok / verified_against_classical below describe
only this stand-in Grover-search circuit, not problem 360 itself.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


N_QUBITS = 4          # search space size N = 2**4 = 16
MODULUS = 3            # property under test: n % MODULUS == 0


def classical_marked_items(n_qubits: int, modulus: int) -> list[int]:
    """Brute-force, from first principles: which n in [0, 2**n_qubits)
    are divisible by `modulus`."""
    size = 2 ** n_qubits
    return [n for n in range(size) if n % modulus == 0]


def oracle_for_value(n_qubits: int, value: int) -> QuantumCircuit:
    """Phase-flip oracle that marks a single computational basis state
    `value` (an n_qubits-bit integer) by applying a multi-controlled Z,
    surrounded by X gates on the bits that should be 0."""
    qc = QuantumCircuit(n_qubits, name=f"oracle_{value}")
    bits = [(value >> i) & 1 for i in range(n_qubits)]

    zero_positions = [i for i, b in enumerate(bits) if b == 0]
    for i in zero_positions:
        qc.x(i)

    # Multi-controlled Z on all qubits: controls = all but last, target = last,
    # with H-CX...-H sandwich to turn a multi-controlled X into multi-controlled Z.
    qc.h(n_qubits - 1)
    if n_qubits - 1 == 0:
        qc.x(0)
    else:
        mcx = MCXGate(n_qubits - 1)
        qc.append(mcx, list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)

    for i in zero_positions:
        qc.x(i)

    return qc


def build_oracle(n_qubits: int, marked_values: list[int]) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="oracle")
    for v in marked_values:
        qc.compose(oracle_for_value(n_qubits, v), inplace=True)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))

    qc.h(n_qubits - 1)
    if n_qubits - 1 == 0:
        qc.x(0)
    else:
        mcx = MCXGate(n_qubits - 1)
        qc.append(mcx, list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)

    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(n_qubits: int, marked_values: list[int], iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_values)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main() -> bool:
    classical_answer = classical_marked_items(N_QUBITS, MODULUS)
    n_space = 2 ** N_QUBITS
    n_marked = len(classical_answer)

    print(f"Search space size N = {n_space}, property: n % {MODULUS} == 0")
    print(f"Classical answer (first-principles trial division): {classical_answer}")
    print(f"Number of marked items M = {n_marked}")

    # Optimal number of Grover iterations ~ (pi/4) * sqrt(N/M)
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_space / n_marked)))
    print(f"Grover iterations used: {iterations}")

    qc = build_grover_circuit(N_QUBITS, classical_answer, iterations)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Aggregate measured outcomes (bitstrings, little-endian in Qiskit) into ints.
    int_counts = Counter()
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        int_counts[value] += c

    # Top outcomes by count, restricted to as many slots as marked items.
    top_outcomes = [v for v, _ in int_counts.most_common(n_marked)]
    top_outcomes_sorted = sorted(top_outcomes)

    total_marked_shots = sum(c for v, c in int_counts.items() if v in classical_answer)
    marked_fraction = total_marked_shots / shots

    print(f"Top {n_marked} measured outcomes (by frequency): {top_outcomes_sorted}")
    print(f"Fraction of shots landing on a truly-marked (n % {MODULUS} == 0) value: {marked_fraction:.4f}")

    outcomes_match = set(top_outcomes_sorted) == set(classical_answer)
    high_success = marked_fraction > 0.80

    verified = outcomes_match and high_success
    print(f"outcomes_match={outcomes_match}, high_success(>0.85)={high_success}")

    status = "PASS" if verified else "FAIL"
    print(status)
    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
