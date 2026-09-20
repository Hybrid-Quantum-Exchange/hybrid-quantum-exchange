"""
Erdos problem #633 -- quantum-testable-sequence lane.

Source metadata (data/problems.yaml in manman4/erdosproblems, entry
`number: "633"`):
    prize: $25
    status: solved (2026-04-06), formalized: yes
    oeis: ["N/A"]
    tags: ["geometry"]

HONEST LIMITATION, stated up front: problem 633's entry carries no OEIS
sequence id at all ("N/A"). There is therefore no OEIS-derived integer
sequence for this lane to build a membership/divisibility/counting
property from, and this script does NOT fabricate one. Per the task's own
fallback instructions ("if no OEIS id ... write the script anyway with
your best honest attempt, note the limitation clearly, and report
ran_ok/verified_against_classical accurately"), this script instead builds
a genuine, self-contained finite/computable problem in the spirit of the
problem's only real piece of metadata -- its tag, "geometry" -- and runs a
real Grover search circuit against it. This is NOT a test of Erdos problem
633 itself; it is the best-honest-effort substitute the task explicitly
allows when no OEIS id exists.

Classical property under test (computed from first principles below, not
copied from anywhere):
    Take the 4 integer lattice points of the unit square,
        P = [(0,0), (0,1), (1,0), (1,1)],
    indexed 0..3 by two bits (b1 b0) = binary(index). Define the "extremal
    corner" as the unique point maximizing x + y (a basic geometric
    extremal-point query). Brute-force classical evaluation below finds
    this is point index 3 = (1,1), the unique maximizer, with value 2.

Quantum circuit:
    A standard 2-qubit Grover search (oracle + diffuser, N=4, 1 marked
    item, 1 Grover iteration is exact/optimal for N=4) that searches for
    the index of the extremal corner. The oracle is derived programmatically
    from the classical brute-force result -- it is not hard-coded to "11"
    by fiat; the script computes the extremal index first and then builds
    the oracle to mark whatever index that turns out to be, so the circuit
    is actually solving the geometric search, not just performing a fixed
    party trick.

Pass/fail: run the circuit on the ideal AerSimulator, take the most
frequent measured basis state, and PASS iff it equals the classically
computed extremal-corner index.
"""

import sys
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_extremal_corner_index():
    """Brute-force, from first principles: among the 4 unit-square lattice
    points, find the index (0..3, in the order below) of the unique point
    maximizing x + y. Returns (index, points, value)."""
    points = [(0, 0), (0, 1), (1, 0), (1, 1)]
    values = [x + y for (x, y) in points]
    best_value = max(values)
    winners = [i for i, v in enumerate(values) if v == best_value]
    if len(winners) != 1:
        raise RuntimeError(
            f"expected a unique maximizer, got ties among indices {winners}"
        )
    return winners[0], points, best_value


def build_oracle(marked_index: int) -> QuantumCircuit:
    """2-qubit phase oracle flipping the sign of |marked_index> (0..3),
    qubit ordering q1 q0 = binary(marked_index) with q0 the least
    significant bit (Qiskit little-endian convention)."""
    if not (0 <= marked_index <= 3):
        raise ValueError("marked_index must be in 0..3 for a 2-qubit oracle")
    bits = format(marked_index, "02b")  # bits[0] = q1 (MSB), bits[1] = q0 (LSB)
    qc = QuantumCircuit(2, name="oracle")
    # Put any 0-bits into the "control on 1" convention via X-sandwiching.
    if bits[0] == "0":
        qc.x(1)
    if bits[1] == "0":
        qc.x(0)
    qc.cz(0, 1)
    if bits[0] == "0":
        qc.x(1)
    if bits[1] == "0":
        qc.x(0)
    return qc


def build_diffuser() -> QuantumCircuit:
    """Standard 2-qubit Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(2, name="diffuser")
    qc.h([0, 1])
    qc.x([0, 1])
    qc.cz(0, 1)
    qc.x([0, 1])
    qc.h([0, 1])
    return qc


def build_grover_circuit(marked_index: int, iterations: int = 1) -> QuantumCircuit:
    oracle = build_oracle(marked_index)
    diffuser = build_diffuser()

    qc = QuantumCircuit(2, 2)
    qc.h([0, 1])
    for _ in range(iterations):
        qc.compose(oracle, [0, 1], inplace=True)
        qc.compose(diffuser, [0, 1], inplace=True)
    qc.measure([0, 1], [0, 1])
    return qc


def run_and_verify():
    marked_index, points, best_value = classical_extremal_corner_index()
    print(
        f"Classical brute force: points={points}, x+y values={[x + y for x, y in points]}"
    )
    print(
        f"Classical answer: extremal-corner index = {marked_index} "
        f"(point {points[marked_index]}, value {best_value})"
    )

    # Sanity-check the oracle against a full classical truth table before
    # trusting it inside the circuit: it must flip the sign of exactly the
    # marked basis state and leave every other amplitude's sign alone.
    oracle = build_oracle(marked_index)
    from qiskit.quantum_info import Statevector

    for idx in range(4):
        bits = format(idx, "02b")
        prep = QuantumCircuit(2)
        if bits[1] == "1":
            prep.x(0)
        if bits[0] == "1":
            prep.x(1)
        sv_before = Statevector.from_instruction(prep)
        sv_after = sv_before.evolve(oracle)
        sign_before = np.sign(sv_before.data[sv_before.data.nonzero()][0].real)
        sign_after = np.sign(sv_after.data[sv_after.data.nonzero()][0].real)
        expected_flip = idx == marked_index
        actual_flip = sign_before != sign_after
        if actual_flip != expected_flip:
            raise RuntimeError(
                f"oracle sanity check failed at index {idx}: "
                f"expected_flip={expected_flip}, actual_flip={actual_flip}"
            )
    print("Oracle sanity check against full truth table: OK")

    qc = build_grover_circuit(marked_index, iterations=1)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    print(f"Measurement counts (shots={shots}): {counts}")

    # Qiskit's bit string is "c1c0" (q1 q0), matching our little-endian
    # oracle convention, so int(bitstring, 2) recovers the index directly.
    best_outcome = max(counts, key=counts.get)
    measured_index = int(best_outcome, 2)
    measured_prob = counts[best_outcome] / shots

    print(
        f"Most frequent measured index = {measured_index} "
        f"(probability ~{measured_prob:.3f})"
    )

    verified = measured_index == marked_index and measured_prob > 0.5
    return verified, marked_index, measured_index, measured_prob


def main():
    try:
        verified, expected, measured, prob = run_and_verify()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR while running circuit: {exc}")
        print("FAIL")
        sys.exit(1)

    if verified:
        print(
            f"PASS: Grover search recovered classical answer "
            f"(index {expected}) with probability ~{prob:.3f}"
        )
        sys.exit(0)
    else:
        print(
            f"FAIL: Grover search returned index {measured}, "
            f"classical answer was {expected}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
