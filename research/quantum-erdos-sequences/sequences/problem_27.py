"""
Erdos Problem #27 (erdosproblems.com/27) -- quantum-testable instance.

Source metadata (data/problems.yaml, block "number: \"27\"", read 2026-09-19):
    prize: $100
    informal_status: disproved (2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory", "covering systems"]

LIMITATION, stated honestly up front: this problem has no associated OEIS
sequence id ("N/A"), so there is no OEIS-derived term to verify against as
instructed for the general case. The problem itself concerns Erdos's minimum-
modulus conjecture for covering systems of congruences (a system of
congruences n = r_i (mod m_i) that together cover every integer, using
distinct moduli all >= some bound N); Hough proved in 2015 that no covering
system exists with all moduli distinct and >= 10^16, disproving Erdos's
conjecture that arbitrarily large minimum moduli are possible. That result
itself is not a small finite computation. Instead, per the instructions'
fallback for a problem with no usable OEIS id, this script tests a genuine,
small, finite, computable property that is faithful to the problem's own
subject matter (covering systems of congruences): given a small, explicitly
INCOMPLETE covering system

    S = { (r=0, m=2), (r=0, m=3), (r=1, m=4) }

over the search space n = 0, 1, ..., 15 (4 qubits), find the integer(s) n in
that range that are NOT covered by any congruence in S (n mod m_i != r_i for
every i). This is a small, well-defined, computable property (an explicit
finite existence/search problem about a covering system) with a classical
answer computed from first principles below, and it is exactly the kind of
oracle Grover's algorithm is built to search: mark the "uncovered" residues
and amplify them.

Classical computation (done in this script, not copied from anywhere):
    covered(n) = (n % 2 == 0) or (n % 3 == 0) or (n % 4 == 1)
    marked = { n in [0, 15] : not covered(n) }

Quantum approach:
    Grover's algorithm on 4 qubits (search space size N = 16). The oracle is
    built directly from the classically-computed "marked" set: for each
    marked basis state |n>, a multi-controlled-Z (phase flip) is applied,
    using X gates to route the "0" bits of n through the controls. The
    standard number of Grover iterations, round(pi/4 * sqrt(N/M)) where M is
    the number of marked items, is used. The circuit is run on the ideal
    AerSimulator (statevector-based qasm simulation), and the most frequent
    measured outcome(s) are compared against the classically-computed marked
    set.

PASS/FAIL: PASS iff the set of most-frequently-measured basis states (taking
the top M outcomes by count, M = number of marked classical answers) equals
exactly the classically-computed marked set.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_QUBITS = 4
N = 2 ** N_QUBITS  # 16


def covered(n: int) -> bool:
    """Is integer n covered by the (deliberately incomplete) system S?"""
    return (n % 2 == 0) or (n % 3 == 0) or (n % 4 == 1)


def classical_marked_set():
    """Brute-force, from first principles, the n in [0, N-1] NOT covered."""
    return sorted(n for n in range(N) if not covered(n))


def apply_multicontrolled_phase_flip(qc: QuantumCircuit, n: int, n_qubits: int):
    """Flip the phase of basis state |n> (n_qubits-bit binary, qubit 0 = LSB)."""
    bits = [(n >> i) & 1 for i in range(n_qubits)]
    # Open-control: X on qubits whose target bit is 0, so an all-ones
    # multi-controlled-Z fires exactly on |n>.
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)


def build_oracle(marked, n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="oracle")
    for n in marked:
        apply_multicontrolled_phase_flip(qc, n, n_qubits)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked, n_qubits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    marked = classical_marked_set()
    m = len(marked)
    assert 0 < m < N, "degenerate instance: oracle needs at least one marked and one unmarked state"

    iterations = max(1, round((math.pi / 4) * math.sqrt(N / m)))

    qc = build_grover_circuit(marked, N_QUBITS, iterations)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical-register bitstring already reads as a normal binary
    # number with qubit 0 as the least-significant bit (rightmost char), so
    # it converts directly to the integer n it represents.
    int_counts = Counter()
    for bitstring, c in counts.items():
        n = int(bitstring, 2)
        int_counts[n] += c

    # Take the top-m most frequent measured integers as the quantum answer.
    quantum_top = sorted(n for n, _ in int_counts.most_common(m))

    print(f"Erdos problem #27 (covering systems) -- Grover search instance")
    print(f"Search space: n in [0, {N - 1}]  (N = {N}, {N_QUBITS} qubits)")
    print(f"Covering system tested (deliberately incomplete): "
          f"n=0 (mod 2), n=0 (mod 3), n=1 (mod 4)")
    print(f"Classical marked set (uncovered n): {marked}")
    print(f"Grover iterations used: {iterations}")
    print(f"Top-{m} measured outcomes (by count): {quantum_top}")
    print(f"Full measurement counts (by integer n): "
          f"{dict(sorted(int_counts.items()))}")

    ok = quantum_top == marked
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
