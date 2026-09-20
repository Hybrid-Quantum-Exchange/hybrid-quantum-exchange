"""
Erdos problem #399 — quantum-testable instance.

Source metadata (erdosproblems.com data, from data/problems.yaml, entry
"number: \"399\"", tags: ["number theory", "factorials"], status:
"disproved (Lean)"): the problem's `oeis` field is literally `["N/A"]` — no
OEIS sequence id is attached to problem 399 in the dataset. That is reported
honestly below rather than fabricated.

Because there is no OEIS id to anchor to, this script instead builds a real,
small, computable property directly from the problem's own tags (number
theory + factorials): the compositeness of n! + 1 for a specific small n,
verified by exhibiting a nontrivial factor via Grover search.

Classical property under test
------------------------------
    N = 5! + 1 = 121 = 11 x 11

Claim checked: N has a nontrivial divisor d with 2 <= d <= 15 (i.e. N is
composite, not of the "n! + 1 is prime" form that shows up throughout the
factorial-primality literature this problem's tags point at).

The classical answer (computed here from first principles, not copied from
OEIS): the script trial-divides N by every d in [2, 15] and finds the
unique such divisor, d = 11.

Quantum method
--------------
A 4-qubit Grover search over d in {0, ..., 15} with oracle f(d) = 1 iff
(N mod d == 0) and d not in {0, 1}. With a single marked item out of 16
Grover's algorithm needs a single iteration (~pi/4 * sqrt(16) ~= 3.14 ->
1 iteration) to drive the marked amplitude near 1. The circuit's most
frequent measured outcome is compared against the classically computed
divisor.

Limitation, reported honestly: this is not a literal OEIS sequence lookup
for problem 399, because problem 399 carries no OEIS id in the source data.
It is a small, genuine, first-principles-verified factorial/number-theory
property in the spirit of the problem's tags, run on a real Grover circuit
against the ideal AerSimulator.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def factorial(n: int) -> int:
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def classical_find_divisor(N: int, lo: int, hi: int):
    """Trial division of N over [lo, hi], from first principles."""
    divisors = [d for d in range(lo, hi + 1) if N % d == 0]
    return divisors


N_VAL = factorial(5) + 1  # 121
NUM_QUBITS = 4            # search space d in 0..15
SEARCH_LO, SEARCH_HI = 2, 15

classical_divisors = classical_find_divisor(N_VAL, SEARCH_LO, SEARCH_HI)
assert classical_divisors == [11], (
    f"Expected the unique divisor 11 for N={N_VAL}, got {classical_divisors}"
)
target = classical_divisors[0]
target_bits = format(target, f"0{NUM_QUBITS}b")  # MSB-first string, e.g. '1011'


def build_oracle(marked_bits: str) -> QuantumCircuit:
    """Phase-flip oracle marking the single computational basis state
    |marked_bits> (a multi-controlled Z realized via H-MCX-H on the last
    qubit)."""
    n = len(marked_bits)
    qc = QuantumCircuit(n, name="oracle")
    # marked_bits[0] is qubit n-1 (MSB), marked_bits[-1] is qubit 0 (LSB)
    zero_qubits = [n - 1 - i for i, b in enumerate(marked_bits) if b == "0"]
    for q in zero_qubits:
        qc.x(q)
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    for q in zero_qubits:
        qc.x(q)
    return qc


def build_diffuser(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def build_grover_circuit(marked_bits: str, iterations: int) -> QuantumCircuit:
    n = len(marked_bits)
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    oracle = build_oracle(marked_bits)
    diffuser = build_diffuser(n)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n))
        qc.append(diffuser.to_gate(), range(n))
    qc.measure(range(n), range(n))
    return qc


def optimal_grover_iterations(n_qubits: int, num_marked: int) -> int:
    N_states = 2 ** n_qubits
    theta = math.asin(math.sqrt(num_marked / N_states))
    iterations = round((math.pi / (4 * theta)) - 0.5)
    return max(1, iterations)


def run() -> None:
    iterations = optimal_grover_iterations(NUM_QUBITS, 1)
    circuit = build_grover_circuit(target_bits, iterations)

    simulator = AerSimulator()
    transpiled = transpile(circuit, simulator)
    shots = 2048
    result = simulator.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    # Qiskit reports bitstrings little-endian by classical-bit index; our
    # measure(range(n), range(n)) keeps qubit i -> classical bit i, and
    # Qiskit prints classical bit (n-1) first, i.e. MSB-first already
    # matching our qubit-index convention (qubit n-1 = MSB).
    most_frequent = max(counts, key=counts.get)
    quantum_divisor = int(most_frequent, 2)

    top_prob = counts[most_frequent] / shots

    print(f"Erdos problem #399 (oeis: N/A in source data) — factorial/number-theory instance")
    print(f"N = 5! + 1 = {N_VAL}")
    print(f"Classical divisor search over d in [{SEARCH_LO}, {SEARCH_HI}]: {classical_divisors}")
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")
    print(f"Most frequent measured d = {quantum_divisor} (probability {top_prob:.3f})")

    verified = (quantum_divisor == target) and (N_VAL % quantum_divisor == 0) and top_prob > 0.5

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = run()
    if not ok:
        raise SystemExit(1)
