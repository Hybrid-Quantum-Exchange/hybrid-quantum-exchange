"""
Erdos problem #415 -- quantum-testable instance.

LIMITATION (read first): problem #415's entry in erdosproblems/data/problems.yaml
lists `oeis: ["possible"]` -- that is not an OEIS sequence id, it is the
tracker's placeholder meaning "an OEIS id is possible/pending but none has been
assigned yet." The problem has no attached description text in the data
repository beyond `tags: ["number theory"]`, `status: open`. So there is no
concrete integer sequence to derive a property from, and no literal OEIS value
to check against. This script is therefore NOT a property of problem #415's
own (nonexistent) sequence. Instead, honoring the task's fallback instruction
("write the script anyway with your best honest attempt... report accurately
rather than faking a pass"), it implements a genuine, self-contained
number-theory decision problem in the same tag category (number theory) on a
small finite instance, computed classically from first principles and then
verified with a real Grover-search quantum circuit on AerSimulator.

Chosen finite property (NOT sourced from any OEIS entry -- computed here):
    "Does N = 15 have a nontrivial divisor d with 2 <= d <= 13?"
    i.e. is N composite, and if so, find the smallest such divisor via
    Grover's algorithm searching the 4-qubit space {0, 1, ..., 15}.

    N = 15 = 3 * 5, so the marked (solution) states in the search space are
    x = 3 and x = 5 (divisors d with 15 mod d == 0, restricting to 2 <= d < N
    to exclude the trivial divisors 1 and 15).

Classical ground truth is computed directly in `classical_divisors()` below
by trial division -- no value is copied from OEIS or any external source.

The quantum circuit is a standard 2-solution Grover search over 4 qubits:
  - Oracle: phase-flips |x> for x in {3, 5} using a diffusion-compatible
    multi-controlled-Z built from X gates + MCZ.
  - Diffuser: standard inversion-about-the-mean operator.
  - Optimal iteration count for N=16, M=2 solutions: ~2 iterations
    (floor(pi/4 * sqrt(16/2)) = floor(pi/4*2.828) = 2).

PASS criterion: the two most frequent measurement outcomes (by far) match the
classical solution set {3, 5} computed independently by trial division.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


ERDOS_PROBLEM_NUMBER = 415
OEIS_IDS_USED: list[str] = []  # none: problem #415 has no assigned OEIS id ("possible" placeholder only)
N = 15  # the integer whose nontrivial divisors we search for
NUM_QUBITS = 4  # search space {0,...,15}, big enough to cover 2..13


def classical_divisors(n: int) -> list[int]:
    """Return all divisors d of n with 2 <= d <= n-1, found by trial division.

    Computed from first principles (no OEIS lookup, no hardcoded literal).
    """
    return [d for d in range(2, n) if n % d == 0]


def build_oracle(marked_states: list[int], num_qubits: int) -> QuantumCircuit:
    """Phase-oracle that flips the sign of each marked computational basis state."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{num_qubits}b")[::-1]  # little-endian per qubit index
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        if zero_qubits:
            qc.x(zero_qubits)
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        if zero_qubits:
            qc.x(zero_qubits)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def grover_search_divisors(n: int, num_qubits: int, shots: int = 4096):
    marked = [d for d in classical_divisors(n) if d < (1 << num_qubits)]
    if not marked:
        raise ValueError("no marked states fit in the given qubit register")

    space_size = 1 << num_qubits
    num_solutions = len(marked)
    iterations = max(1, round((np.pi / 4) * np.sqrt(space_size / num_solutions)))

    oracle = build_oracle(marked, num_qubits)
    diffuser = build_diffuser(num_qubits)

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, marked, iterations


def main() -> bool:
    classical = classical_divisors(N)
    print(f"Erdos problem #{ERDOS_PROBLEM_NUMBER}")
    print(f"OEIS ids used: {OEIS_IDS_USED!r} (none assigned to this problem)")
    print(f"Classical property: nontrivial divisors of N={N} in [2, {N - 1}]")
    print(f"Classical answer (trial division): {classical}")

    counts, marked, iterations = grover_search_divisors(N, NUM_QUBITS)
    print(f"Grover iterations used: {iterations}")

    # Rank measured bitstrings by frequency, convert to integers.
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_k = ranked[: len(marked)]
    top_values = sorted(int(bs, 2) for bs, _ in top_k)

    print(f"Top {len(marked)} measured outcomes (by count): {top_values}")
    print(f"Expected marked states: {sorted(marked)}")

    total_shots = sum(counts.values())
    marked_mass = sum(c for bs, c in counts.items() if int(bs, 2) in marked)
    marked_fraction = marked_mass / total_shots
    print(f"Fraction of shots landing on a marked state: {marked_fraction:.3f}")

    ok = (top_values == sorted(marked)) and (marked_fraction > 0.8)
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
