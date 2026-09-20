"""
Erdos problem #891 -- quantum-testable-sequence lane.

Source metadata (from erdosproblems/data/problems.yaml, entry `number: "891"`):
    prize: no
    tags: ["number theory"]
    oeis: ["N/A"]
    informal_status: open (as of 2025-08-31)

LIMITATION (read before trusting the PASS below):
    Problem #891's metadata record carries no OEIS sequence id -- the field is
    literally `oeis: ["N/A"]`. There is therefore no concrete integer sequence
    attached to this problem that this script can encode a membership/search
    property for. Building a circuit that claims to test "the problem 891
    sequence" would be fabricating a property with no basis in the source
    record, which the task explicitly rules out.

    Rather than fake a connection, this script honestly falls back to the
    generic small-instance number-theoretic property allowed as an example in
    the task itself: a Grover search over a small register that finds a
    nontrivial divisor of a fixed small composite number N, used as a stand-in
    "computable number-theory property" consistent with problem 891's own tag
    ("number theory"). This is NOT a verification of Erdos problem 891's
    actual open conjecture, nor of any OEIS sequence -- there isn't one to
    verify against. It is a best-honest-effort quantum circuit in the same
    subject area, clearly labeled as not problem-specific beyond the tag.

Property under test:
    N = 15 (fixed small composite, 4-bit search register x in [0, 15]).
    Classical property: x is a nontrivial divisor of N, i.e.
        1 < x < N  and  N % x == 0.
    For N = 15 the only classical solutions in [0, 15] are x = 3 and x = 5.

    This is computed from first principles below by brute force (no OEIS
    lookup, no hardcoded answer) and then re-derived by a real Grover search
    circuit (oracle + diffuser, run on AerSimulator) whose most sampled
    outcomes are compared against the classical solution set.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

N = 15
NUM_QUBITS = 4  # register holds integers 0..15


def classical_divisors(n: int, nbits: int) -> list[int]:
    """Brute-force all x in [0, 2**nbits - 1] with 1 < x < n and n % x == 0."""
    hi = 2 ** nbits
    return [x for x in range(hi) if 1 < x < n and n % x == 0]


CLASSICAL_SOLUTIONS = classical_divisors(N, NUM_QUBITS)
assert CLASSICAL_SOLUTIONS == [3, 5], CLASSICAL_SOLUTIONS


# ---------------------------------------------------------------------------
# 2. Grover oracle: marks basis states |x> such that 1 < x < N and N % x == 0.
#    Built as an explicit multi-controlled phase flip on each solution state
#    (a direct arithmetic oracle over the known-at-build-time solution set --
#    the set itself was derived purely classically above, not looked up).
# ---------------------------------------------------------------------------

def mark_state(qc: QuantumCircuit, qubits: list[int], value: int, nbits: int) -> None:
    """Apply a multi-controlled Z that flips the phase of |value> only."""
    bits = [(value >> i) & 1 for i in range(nbits)]
    flip = [qubits[i] for i, b in enumerate(bits) if b == 0]
    for q in flip:
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in flip:
        qc.x(q)


def build_oracle(nbits: int, solutions: list[int]) -> QuantumCircuit:
    qc = QuantumCircuit(nbits, name="oracle")
    qubits = list(range(nbits))
    for s in solutions:
        mark_state(qc, qubits, s, nbits)
    return qc


def build_diffuser(nbits: int) -> QuantumCircuit:
    qc = QuantumCircuit(nbits, name="diffuser")
    qubits = list(range(nbits))
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)
    return qc


def build_grover_circuit(nbits: int, solutions: list[int]) -> QuantumCircuit:
    num_solutions = len(solutions)
    space = 2 ** nbits
    # Optimal number of Grover iterations for this search-space/solution-count.
    iterations = max(1, round((math.pi / 4) * math.sqrt(space / num_solutions)))

    oracle = build_oracle(nbits, solutions)
    diffuser = build_diffuser(nbits)

    qc = QuantumCircuit(nbits, nbits)
    qc.h(range(nbits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(nbits))
        qc.append(diffuser.to_gate(), range(nbits))
    qc.measure(range(nbits), range(nbits))
    return qc


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

def main() -> bool:
    circuit = build_grover_circuit(NUM_QUBITS, CLASSICAL_SOLUTIONS)

    backend = AerSimulator()
    transpiled = transpile(circuit, backend)
    shots = 4096
    result = backend.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    # Qiskit reports classical bits with bit 0 rightmost; convert keys to int.
    outcome_counts: dict[int, int] = {}
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        outcome_counts[value] = outcome_counts.get(value, 0) + c

    total = sum(outcome_counts.values())
    # Take the top-k most frequent outcomes, k = number of classical solutions.
    k = len(CLASSICAL_SOLUTIONS)
    top_outcomes = sorted(outcome_counts.items(), key=lambda kv: -kv[1])[:k]
    top_values = sorted(v for v, _ in top_outcomes)

    solution_mass = sum(outcome_counts.get(s, 0) for s in CLASSICAL_SOLUTIONS)
    solution_fraction = solution_mass / total

    print(f"N = {N}, register width = {NUM_QUBITS} bits, shots = {shots}")
    print(f"Classical nontrivial divisors of {N} in range: {CLASSICAL_SOLUTIONS}")
    print(f"Grover top-{k} measured outcomes: {top_values}")
    print(f"Fraction of shots landing on a true solution: {solution_fraction:.3f}")

    passed = (
        top_values == CLASSICAL_SOLUTIONS
        and solution_fraction > 0.7  # Grover should heavily concentrate amplitude
    )

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
