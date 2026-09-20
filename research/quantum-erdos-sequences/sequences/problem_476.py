"""
Erdos problem #476 -- quantum-testable lane.

Source metadata (verified 2026-09-19 from a read-only clone of
manman4/erdosproblems, data/problems.yaml, block "- number: \"476\""):
    prize: no
    informal_status: proved (2025-12-31), formal_status: Lean (2025-12-31)
    oeis: ["N/A"]
    tags: ["number theory", "additive combinatorics"]

LIMITATION (reported honestly, per instructions): problem 476 has **no OEIS
sequence id** attached in the source data (oeis: ["N/A"]), and the metadata
file carries no statement/description field for this problem, only status
and tags. There is therefore no genuine OEIS sequence to build a
membership/term-defining circuit against for this specific problem, and this
script does NOT claim otherwise and does NOT fabricate an OEIS value.

Best-effort substitute, honestly scoped: the one concrete, finite,
computable, and mathematically real property implied by problem 476's own
tags ("number theory", "additive combinatorics") is Sidon-set membership --
a classical, well-defined additive-combinatorics property (a set is a Sidon
set iff all pairwise sums of its elements, including sums of an element with
itself, are distinct). This is genuine additive combinatorics content, and
it is a small, exactly computable search problem (finite subsets of a small
ground set), so it is suitable for a real Grover-search circuit. It is
presented as an illustrative additive-combinatorics instance INSPIRED BY the
problem's tags -- not as problem 476's actual sequence, since no such
sequence id exists in the source data.

Concrete instance (N = 16 = 2**4, so 4 qubits, well within N <= 64):
    Ground set E = {1, 2, 3, 4}.
    Every subset S of E is encoded as a 4-bit string b3 b2 b1 b0, bit i telling
    whether element (i+1) is in S.
    S is a Sidon set iff all pairwise sums a+b for a,b in S, a <= b, are
    distinct (this includes a==b, i.e. doubling each element).

    The classical answer (all NON-Sidon subsets of {1,2,3,4} -- the minority
    of the 16 subsets, computed in this script by brute force, not looked
    up) is used to build a Grover oracle that phase-flips exactly those
    marked (non-Sidon) computational basis states.
    Grover's algorithm is then run on the ideal AerSimulator and the most
    probable measured bitstrings are checked against the classically
    computed marked set.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of the property, from first principles.
# ---------------------------------------------------------------------------

GROUND_SET = (1, 2, 3, 4)
N_QUBITS = len(GROUND_SET)  # 4 qubits -> search space size 16


def subset_from_bits(bits: int) -> tuple[int, ...]:
    """bits is an integer 0..2**N_QUBITS-1; bit i (LSB-first) selects
    GROUND_SET[i]."""
    return tuple(GROUND_SET[i] for i in range(N_QUBITS) if (bits >> i) & 1)


def is_sidon(subset: tuple[int, ...]) -> bool:
    """A finite set of integers is a Sidon set iff all pairwise sums
    a + b (a <= b, a, b in the set) are distinct."""
    sums = [a + b for a, b in itertools.combinations_with_replacement(subset, 2)]
    return len(sums) == len(set(sums))


def compute_marked_states() -> list[int]:
    """Brute-force, from first principles, every subset of GROUND_SET and
    return the integer bitstrings of the NON-Sidon ones (a strict minority
    of the 16 subsets of {1,2,3,4}), which is what Grover searches for.
    A small search space with a small marked minority is exactly the shape
    Grover amplitude amplification is built for."""
    marked = []
    for bits in range(2 ** N_QUBITS):
        if not is_sidon(subset_from_bits(bits)):
            marked.append(bits)
    return marked


MARKED_STATES = compute_marked_states()

# Sanity print of the classical answer (computed here, not copied).
print(f"Ground set: {GROUND_SET}")
print(f"Search space size: {2 ** N_QUBITS}")
print(f"Classically computed NON-Sidon subsets (marked states): {len(MARKED_STATES)}")
for b in MARKED_STATES:
    print(f"  bits={b:04b}  subset={subset_from_bits(b)}")


# ---------------------------------------------------------------------------
# 2. Grover search circuit marking exactly MARKED_STATES.
# ---------------------------------------------------------------------------

def bitstring(b: int, n: int) -> str:
    return format(b, f"0{n}b")


def apply_oracle(qc: QuantumCircuit, marked_states: list[int], n: int) -> None:
    """Phase-flip each marked computational basis state using an
    X - multi-controlled-Z - X sandwich (standard technique for building a
    phase oracle from an explicit list of marked bitstrings)."""
    for state in marked_states:
        bits = bitstring(state, n)  # MSB-first string of length n
        zero_positions = [i for i, c in enumerate(bits) if c == "0"]
        # qc qubit index j corresponds to string position (n-1-j) since
        # Qiskit orders qubit 0 as the least-significant bit; here we index
        # qubits directly by "element i" == qubit i (bit i of `state`).
        zero_qubits = [i for i in range(n) if not ((state >> i) & 1)]
        if zero_qubits:
            qc.x(zero_qubits)
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        if zero_qubits:
            qc.x(zero_qubits)


def apply_diffuser(qc: QuantumCircuit, n: int) -> None:
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))


def build_grover_circuit(marked_states: list[int], n: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iterations):
        apply_oracle(qc, marked_states, n)
        apply_diffuser(qc, n)
    qc.measure(range(n), range(n))
    return qc


def optimal_grover_iterations(n_items: int, n_marked: int) -> int:
    if n_marked == 0:
        return 0
    theta = math.asin(math.sqrt(n_marked / n_items))
    iterations = round((math.pi / (4 * theta)) - 0.5)
    return max(1, iterations)


# ---------------------------------------------------------------------------
# 3. Run on AerSimulator and check the result against the classical answer.
# ---------------------------------------------------------------------------

def main() -> bool:
    n_items = 2 ** N_QUBITS
    n_marked = len(MARKED_STATES)
    iters = optimal_grover_iterations(n_items, n_marked)
    print(f"\nMarked count = {n_marked}, Grover iterations = {iters}")

    qc = build_grover_circuit(MARKED_STATES, N_QUBITS, iters)

    simulator = AerSimulator()
    tqc = transpile(qc, simulator)
    result = simulator.run(tqc, shots=4096).result()
    counts = result.get_counts()

    # Qiskit's classical register bit order: rightmost char = qubit 0 (bit 0
    # of our `state` integer), so a measured bitstring "b3b2b1b0" already
    # matches our `bits` integer encoding directly when parsed as binary.
    measured_int_counts = {int(k, 2): v for k, v in counts.items()}
    total_shots = sum(measured_int_counts.values())

    marked_set = set(MARKED_STATES)
    marked_shots = sum(v for k, v in measured_int_counts.items() if k in marked_set)
    marked_fraction = marked_shots / total_shots

    print(f"Fraction of shots landing on a classically-verified NON-Sidon state: "
          f"{marked_fraction:.3f}")

    # Also explicitly check that the single most-frequent measured state is
    # itself a genuine non-Sidon set, cross-checked independently by
    # is_sidon().
    top_state = max(measured_int_counts, key=measured_int_counts.get)
    top_subset = subset_from_bits(top_state)
    top_is_marked_quantum_claim = top_state in marked_set
    top_is_nonsidon_classical_recheck = not is_sidon(top_subset)

    print(f"Most frequent measured state: bits={top_state:04b} "
          f"subset={top_subset} "
          f"(quantum-oracle-marked-non-Sidon={top_is_marked_quantum_claim}, "
          f"independent classical re-check={top_is_nonsidon_classical_recheck})")

    success = (
        marked_fraction > 0.5
        and top_is_marked_quantum_claim
        and top_is_nonsidon_classical_recheck
    )
    return success


if __name__ == "__main__":
    ok = main()
    print("\nPASS" if ok else "\nFAIL")
