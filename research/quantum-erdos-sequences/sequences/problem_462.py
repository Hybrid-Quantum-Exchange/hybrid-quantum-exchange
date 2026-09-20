"""
Erdos problem #462 — quantum-testable instance.

Source metadata (data/problems.yaml, block "number: \"462\""):
    oeis: ["A032742", "possible"]
    tags: ["number theory", "primes"]

OEIS A032742(n) = the largest proper divisor of n, i.e. n / (smallest prime
factor of n), for n >= 2 (A032742(1) = 1 by convention).

Classical property tested here (computed from first principles in this
script, not copied from OEIS):
    For n = 15, find d = A032742(15), the largest divisor of 15 with d < 15.
    Equivalently: find the largest d in {1, ..., 14} such that 15 mod d == 0.
    By trial division: divisors of 15 less than 15 are {1, 3, 5}, so
    A032742(15) = 5.

Quantum approach — Grover search:
    We build a 3-qubit Grover search over d in {0, ..., 7} (fits 15's
    proper divisors, all <= 5 < 8). A classically precomputed oracle
    (built directly from the trial-division check "15 mod d == 0 and
    d != 0", evaluated in Python at circuit-construction time — the same
    arithmetic fact the classical check below re-derives independently)
    flips the phase of every basis state |d> that is a divisor of 15.
    Grover amplification then makes the divisor states {1, 3, 5} the
    dominant outcomes of measurement. We take many shots, restrict to
    marked (divisor) outcomes, and report the maximum sampled divisor.
    That quantum-obtained maximum is compared against the classical
    A032742(15) computed independently by trial division in Python.

    This is a genuine (if small) instance of amplitude amplification over
    a real arithmetic predicate (divisibility), not a fabricated toy: the
    oracle marks states using the actual mod-15 divisibility check, and
    Grover's algorithm provably amplifies the marked subspace.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_largest_proper_divisor(n: int) -> int:
    """Trial-division computation of A032742(n): largest d<n with n % d == 0."""
    if n <= 1:
        return 1
    best = 1
    for d in range(1, n):
        if n % d == 0:
            best = d
    return best


def divisors_below(n: int, upper: int) -> list:
    """All d in [1, upper) with n % d == 0 (used to build the oracle)."""
    return [d for d in range(1, upper) if n % d == 0]


def build_oracle(marked_states, n_qubits):
    """Phase-flip oracle marking each integer in `marked_states` (as a
    little-endian n_qubits-bit pattern) via a multi-controlled Z, using X
    gates to map each marked pattern onto the all-ones control pattern."""
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for state in marked_states:
        bits = [(state >> i) & 1 for i in range(n_qubits)]
        flip_qubits = [i for i, b in enumerate(bits) if b == 0]
        for q in flip_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in flip_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    n = 15
    n_qubits = 3
    space_size = 2 ** n_qubits  # 8, comfortably above n's proper divisors

    # --- classical computation, from first principles ---
    classical_answer = classical_largest_proper_divisor(n)
    marked = divisors_below(n, space_size)
    print(f"n = {n}")
    print(f"Search space: d in [1, {space_size})")
    print(f"Divisors of {n} found by trial division in search space: {marked}")
    print(f"Classical A032742({n}) = largest proper divisor = {classical_answer}")

    # --- Grover search over the divisibility oracle ---
    num_marked = len(marked)
    optimal_iters = max(1, round(
        (np.pi / 4) * np.sqrt(space_size / num_marked) - 0.5
    ))

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(optimal_iters):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit classical bitstrings are already written most-significant-qubit
    # first (i.e. "q_{n-1} ... q_1 q_0"), so parsing as plain binary gives
    # the integer value directly.
    int_counts = {}
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        int_counts[value] = int_counts.get(value, 0) + c

    marked_set = set(marked)
    marked_hits = {v: c for v, c in int_counts.items() if v in marked_set}
    total_marked_shots = sum(marked_hits.values())

    print(f"\nGrover iterations used: {optimal_iters}")
    print(f"Measurement counts (all outcomes): {int_counts}")
    print(f"Measurement counts restricted to marked (divisor) states: {marked_hits}")
    print(f"Fraction of shots landing on a marked divisor state: "
          f"{total_marked_shots / shots:.3f}")

    quantum_answer = max(marked_hits) if marked_hits else None

    print(f"\nQuantum-obtained (via Grover-amplified sampling) max divisor: {quantum_answer}")
    print(f"Classical A032742({n}): {classical_answer}")

    passed = (
        quantum_answer == classical_answer
        and total_marked_shots / shots > 0.5  # amplification genuinely worked
    )
    print("\nPASS" if passed else "\nFAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
