"""
Erdos problem #905 -- quantum-testable companion script.

Source metadata (from erdosproblems.com data, data/problems.yaml, entry
"number: \"905\""):
    prize: no
    informal_status: proved (Lean formal_status too, as of 2026-04-07)
    oeis: ["N/A"]
    tags: ["graph theory"]

LIMITATION, stated honestly up front: problem #905 carries no OEIS sequence
id in the source data (oeis: ["N/A"]). There is therefore no literal OEIS
term this script can derive or check. Per the task's fallback instruction
("if no OEIS id ... write the script anyway with your best honest attempt,
note the limitation clearly"), this script instead builds a genuine small
quantum circuit around the one substantive piece of metadata the problem
does carry: its tag, "graph theory". It does NOT claim to verify any term
of any OEIS sequence, and does NOT claim to resolve or touch the actual
mathematical content of Erdos problem #905 itself -- only that the chosen
graph-theoretic search property is real, finite, and correctly computed
both classically and via Grover's algorithm.

Chosen property (finite, computable, small search space):
    Graph G = 4-cycle C4 on vertices {0,1,2,3} with edges
        (0,1), (1,2), (2,3), (3,0).
    Search space: all 2^4 = 16 subsets S of {0,1,2,3}, encoded as 4-bit
    strings b3 b2 b1 b0 (bit i = 1 iff vertex i in S).
    Property being tested: S is an INDEPENDENT SET of size exactly 2
    (no edge of G has both endpoints in S, and |S| = 2).

Classical ground truth (brute force over all 16 subsets, computed in this
script from first principles, not copied from anywhere):
    The independent sets of size 2 in C4 are exactly {0,2} and {1,3}
    (the two diagonals) -- i.e. bitstrings 0101 and 1010 in b3b2b1b0 order.
    All other 14 subsets either have the wrong size or contain an edge.

Quantum method: Grover's algorithm over the 4-qubit subset register.
An oracle built from elementary gates flips the phase of exactly the
marked bitstrings (found classically above); the standard Grover diffuser
amplifies them. With 2 marked states out of 16, the optimal number of
Grover iterations is floor(pi/4 * sqrt(16/2)) = 2. The circuit is run on
the ideal AerSimulator with many shots; PASS requires that the two most
frequent measured bitstrings are exactly the two classically-marked
independent sets, each with high enough probability to show real
amplification (not a uniform-guess result).
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation of the ground truth (first principles).
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # the 4-cycle C4


def subset_from_bits(bits: tuple[int, ...]) -> set[int]:
    """bits[i] == 1 means vertex i is in the subset."""
    return {i for i, b in enumerate(bits) if b == 1}


def is_independent_set(vertices: set[int], edges: list[tuple[int, int]]) -> bool:
    for u, v in edges:
        if u in vertices and v in vertices:
            return False
    return True


def bitstring_from_index(index: int, n: int) -> tuple[int, ...]:
    """Return bits (bit0 first) of `index` over `n` bits."""
    return tuple((index >> i) & 1 for i in range(n))


classical_marked_indices = []
for idx in range(2 ** N_VERTICES):
    bits = bitstring_from_index(idx, N_VERTICES)
    verts = subset_from_bits(bits)
    if len(verts) == 2 and is_independent_set(verts, EDGES):
        classical_marked_indices.append(idx)

classical_marked_indices.sort()

# Independently re-derive via itertools.combinations as a cross-check.
brute_force_pairs = [
    frozenset(pair)
    for pair in combinations(range(N_VERTICES), 2)
    if is_independent_set(set(pair), EDGES)
]
brute_force_indices = sorted(
    sum(1 << v for v in pair) for pair in brute_force_pairs
)

assert classical_marked_indices == brute_force_indices, (
    "Two independent classical derivations disagree -- bug in the script."
)

print("Classical ground truth:")
print(f"  Graph: C4 on vertices {list(range(N_VERTICES))}, edges {EDGES}")
print(
    "  Independent sets of size 2 (as vertex sets):",
    [set(pair) for pair in brute_force_pairs],
)
print(
    "  As 4-bit strings (b3b2b1b0):",
    [format(i, "04b") for i in classical_marked_indices],
)

assert classical_marked_indices == [5, 10], (
    "Expected exactly the two diagonals {0,2}=0b0101=5 and {1,3}=0b1010=10; "
    f"got {classical_marked_indices}"
)

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for these marked bitstrings.
# ---------------------------------------------------------------------------

N_QUBITS = N_VERTICES  # 4 qubits, one per vertex/bit


def build_oracle(marked_indices: list[int], n: int) -> QuantumCircuit:
    """Phase-flip oracle marking each index in `marked_indices` (little-endian
    qubit order: qubit i <-> bit i, matching bitstring_from_index)."""
    qc = QuantumCircuit(n, name="oracle")
    for idx in marked_indices:
        bits = bitstring_from_index(idx, n)
        zero_qubits = [i for i, b in enumerate(bits) if b == 0]
        # Map |idx> -> |11...1> by flipping the zero-bits, apply a
        # multi-controlled Z (via H-MCX-H on the last qubit), then undo.
        for q in zero_qubits:
            qc.x(q)
        if n == 1:
            qc.z(0)
        else:
            qc.h(n - 1)
            qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
            qc.h(n - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


n_marked = len(classical_marked_indices)
n_total = 2 ** N_QUBITS
optimal_iterations = max(
    1, round((math.pi / 4) * math.sqrt(n_total / n_marked))
)
print(f"\nGrover iterations used: {optimal_iterations} "
      f"(marked={n_marked}, total={n_total})")

oracle = build_oracle(classical_marked_indices, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

grover = QuantumCircuit(N_QUBITS, N_QUBITS)
grover.h(range(N_QUBITS))
for _ in range(optimal_iterations):
    grover.append(oracle.to_instruction(), range(N_QUBITS))
    grover.append(diffuser.to_instruction(), range(N_QUBITS))
grover.measure(range(N_QUBITS), range(N_QUBITS))
grover = grover.decompose()

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

SHOTS = 8192
simulator = AerSimulator()
job = simulator.run(grover, shots=SHOTS)
result = job.result()
counts = result.get_counts()

# Qiskit reports classical bits as a string c[n-1]...c[0]; our register bit i
# is classical bit i, so reverse the printed string to read b0 first, then
# convert to the same little-endian integer convention used above.
def counts_key_to_index(key: str) -> int:
    # key is 'c3c2c1c0' (Qiskit prints MSB first == highest qubit index first)
    bits = key[::-1]  # now bits[0] = c0, bits[1] = c1, ...
    return int(bits, 2) if False else sum(
        (1 << i) for i, b in enumerate(bits) if b == "1"
    )


index_counts: dict[int, int] = {}
for key, c in counts.items():
    idx = counts_key_to_index(key)
    index_counts[idx] = index_counts.get(idx, 0) + c

sorted_by_count = sorted(index_counts.items(), key=lambda kv: -kv[1])
top_two_indices = sorted(idx for idx, _ in sorted_by_count[:2])
top_two_prob = sum(c for _, c in sorted_by_count[:2]) / SHOTS

print("\nMeasurement summary (top results):")
for idx, c in sorted_by_count[:6]:
    print(f"  {format(idx, '04b')}: {c} ({c / SHOTS:.3f})")

print(f"\nTop-2 measured indices: {[format(i, '04b') for i in top_two_indices]}")
print(f"Classical marked indices: {[format(i, '04b') for i in classical_marked_indices]}")
print(f"Combined probability mass on top-2: {top_two_prob:.3f}")

# ---------------------------------------------------------------------------
# 4. Verdict.
# ---------------------------------------------------------------------------

quantum_matches_classical = (
    top_two_indices == sorted(classical_marked_indices) and top_two_prob > 0.7
)

if quantum_matches_classical:
    print("\nPASS: Grover search on the ideal simulator recovered exactly the "
          "classically-computed independent sets of size 2 in C4, with "
          f"{top_two_prob:.3f} combined probability mass (>> the 2/16=0.125 "
          "uniform baseline), confirming real amplitude amplification.")
else:
    print("\nFAIL: quantum measurement did not match the classical ground "
          "truth with sufficient confidence.")

print(
    "\nNote on scope: this PASS/FAIL is about the graph-theoretic Grover "
    "search constructed here, chosen because Erdos problem #905 is tagged "
    "'graph theory' but has no OEIS id (oeis: [\"N/A\"]) to derive a "
    "sequence-term property from. It is not a verification of problem #905 "
    "itself."
)
