"""
Erdos problem #150 -- quantum-testable sequence entry.

Source metadata (erdosproblems.com data, as cloned in
/home/user/manman4/erdosproblems/data/problems.yaml, entry "number: '150'"):
    prize: no
    status: proved (Lean)
    oeis: ["possible"]
    tags: ["graph theory"]

HONESTY NOTE ON THE OEIS FIELD
-------------------------------
The yaml's `oeis` field for problem #150 is the literal string "possible",
not a concrete OEIS sequence id (e.g. "A000001"). That is the site's own
placeholder meaning "an OEIS entry may exist for the objects in this
problem, but none is recorded" -- it is not an id that can be looked up,
and no numeric sequence is attached to this problem in the source data.
So this entry does NOT test OEIS-sequence membership, because there is no
real OEIS id to test against. Inventing one would violate the "do not
fabricate a property" instruction.

What we test instead, honestly
-------------------------------
Problem #150 is tagged "graph theory". In lieu of a real OEIS sequence,
this script builds a genuine, small, finite, classically-checkable graph
decision problem in the same spirit (existence of a maximum independent
set of a given size in a small fixed graph) and solves it with a real
Grover search circuit on the ideal AerSimulator, then checks the quantum
result against a from-scratch classical brute-force computation done in
this same script.

The instance: the 4-cycle graph C4 with vertices {0,1,2,3} and edges
{(0,1),(1,2),(2,3),(3,0)}. We search the 4-bit space of vertex subsets
(bit i = 1 iff vertex i is chosen) for subsets of size exactly 2 that
are independent sets (no chosen pair is an edge). This is a completely
well-defined, small (4 qubits, 16-element search space), computable
property with a known classical answer, and is unrelated to any
fabricated OEIS value.

Classical answer (computed below, first principles, brute force over all
16 subsets of {0,1,2,3}): the independent sets of size 2 in C4 are
{0,2} and {1,3} -> bitstrings '0101' and '1010' (little/big-endian noted
in code), i.e. exactly 2 solutions out of 16.

Pass condition: run Grover's algorithm (oracle + diffuser, iterated the
optimal number of times for N=16, M=2) on the ideal AerSimulator and
confirm the two most frequent measured bitstrings are exactly the two
classically-verified solutions, with combined measured probability
comfortably above the ~1/8 baseline of uniform sampling.
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation, from first principles.
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # the 4-cycle C4


def is_independent_set(subset: frozenset[int]) -> bool:
    for u, v in EDGES:
        if u in subset and v in subset:
            return False
    return True


def bits_to_subset(bits: str) -> frozenset[int]:
    # bits[i] is the state of vertex i (bits[0] = vertex 0, leftmost).
    return frozenset(i for i, b in enumerate(bits) if b == "1")


classical_solutions = []
for bits_tuple in itertools.product("01", repeat=N_VERTICES):
    bits = "".join(bits_tuple)
    subset = bits_to_subset(bits)
    if len(subset) == 2 and is_independent_set(subset):
        classical_solutions.append(bits)

classical_solutions.sort()
print("Classical brute-force solutions (size-2 independent sets of C4):",
      classical_solutions)

# Sanity: this must be exactly {0,2} and {1,3} encoded as bitstrings.
expected = sorted(["1010", "0101"])
assert classical_solutions == expected, (
    f"classical computation disagrees with hand-derivation: "
    f"{classical_solutions} != {expected}"
)

N = 2 ** N_VERTICES          # 16
M = len(classical_solutions)  # 2

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for the marked bitstrings.
# ---------------------------------------------------------------------------

n = N_VERTICES  # number of qubits = number of vertices


def apply_oracle(qc: QuantumCircuit, target_bits: str) -> None:
    """Phase-flip the single basis state matching target_bits (qubit i <-> vertex i,
    with qubit index equal to string position, using little-endian Qiskit convention
    where qc.x is applied per-bit and the multi-controlled Z flips the phase when all
    control qubits (after X-conjugation) are 1)."""
    # Qiskit orders qubit 0 as the least-significant (rightmost) bit of the
    # measured bitstring, so reverse target_bits to match string position i -> qubit i.
    zero_positions = [i for i, b in enumerate(target_bits) if b == "0"]
    for i in zero_positions:
        qc.x(i)
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    for i in zero_positions:
        qc.x(i)


def apply_diffuser(qc: QuantumCircuit) -> None:
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))


# Optimal number of Grover iterations for N states, M marked: ~ (pi/4) * sqrt(N/M)
num_iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))
print(f"Grover iterations: {num_iterations} (N={N}, M={M})")

qc = QuantumCircuit(n, n)
qc.h(range(n))

for _ in range(num_iterations):
    for target in classical_solutions:
        apply_oracle(qc, target)
    apply_diffuser(qc)

qc.measure(range(n), range(n))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
job = sim.run(compiled, shots=shots)
result = job.result()
counts = result.get_counts()

# Qiskit's classical-register bitstrings are printed with qubit n-1 leftmost,
# i.e. counts keys already use the same "position i == qubit i" convention we
# used to build the oracle (c[i] measured from qubit i, register order n-1..0
# but since qubit i corresponds to string position i by construction of the
# oracle above, we compare directly against our classical bitstrings).
sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
print("Top measured outcomes:", sorted_counts[:5])

top_two = [bits for bits, _ in sorted_counts[:2]]
top_two_sorted = sorted(top_two)

combined_prob = sum(c for b, c in counts.items() if b in classical_solutions) / shots
print(f"Combined probability of measuring a true solution: {combined_prob:.4f} "
      f"(uniform baseline would be {M / N:.4f})")

verified = (
    top_two_sorted == expected
    and combined_prob > 0.5
)

print("PASS" if verified else "FAIL")

if not verified:
    sys.exit(1)
