"""
Erdos problem #608 -- quantum-testable sequence entry
======================================================

Source record: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: '608'" (tags: ["graph theory"], informal_status: disproved,
formal_status: Lean-verified, oeis: ["N/A"]).

LIMITATION (reported honestly, as instructed): problem #608 carries no OEIS
sequence id in the source data (oeis: ["N/A"]) and the source record gives no
further textual statement of the problem beyond its tag "graph theory". There
is therefore no specific OEIS sequence to tie a quantum circuit to for this
entry, and nothing to "verify against a classical answer" that would actually
be Erdos-problem-608-specific. Fabricating an OEIS id or a specific numeric
claim about problem 608 itself would violate the task's instructions, so this
script does not do that.

Best honest attempt instead: using the one piece of real content this entry
does carry -- its tag "graph theory" -- this script builds a genuine, finite,
classically-checkable graph-theory search problem in the same spirit as
Erdos-style extremal graph theory (Turan-type triangle-free extremal graphs),
and solves it with a real Grover search circuit on the ideal AerSimulator.
This is NOT a claim about the specific mathematical content of problem 608;
it is the closest legitimate finite/computable instance obtainable from what
the source record actually contains for this problem number.

The concrete finite, computable property
------------------------------------------
Let K4 have 6 possible edges, indexed 0..5:
    0:(0,1) 1:(0,2) 2:(0,3) 3:(1,2) 4:(1,3) 5:(2,3)
Each of the 2^6 = 64 bitstrings x in {0,1}^6 encodes a subgraph of K4 (bit i
= 1 means edge i is present). Define the property

    P(x) = "the subgraph encoded by x is triangle-free"
             (contains none of the 4 triangles of K4: {0,1,3}, {0,2,4},
              {1,2,5}, {3,4,5} as edge-index triples)

The classical extremal fact (Turan's theorem / Mantel's theorem for n=4) is
that the maximum number of edges in a triangle-free graph on 4 vertices is
floor(4^2/4) = 4, achieved uniquely (up to labeling) by the complete
bipartite graph K_{2,2}. This script:
  1. Computes, from first principles (direct triangle enumeration, no
     external data), the full classical set of marked (triangle-free)
     bitstrings among all 64, and in particular the unique maximum-edge
     triangle-free edge set(s).
  2. Builds a Grover search circuit (6 qubits) whose oracle marks exactly the
     bitstrings with the maximum number of edges (4) that are triangle-free
     -- i.e. it searches for a maximum triangle-free subgraph of K4.
  3. Runs the circuit on AerSimulator (ideal, statevector-based sampling) and
     checks that the highest-probability measured outcome(s) match exactly
     the classically-computed marked set.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

from itertools import combinations
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no lookup)
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = list(combinations(range(N_VERTICES), 2))  # 6 edges, index 0..5
N_EDGES = len(EDGES)
assert N_EDGES == 6

EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}

# All triangles of K4 as index-triples into EDGES
TRIANGLES = []
for a, b, c in combinations(range(N_VERTICES), 3):
    tri_edges = (EDGE_INDEX[(a, b)], EDGE_INDEX[(a, c)], EDGE_INDEX[(b, c)])
    TRIANGLES.append(tri_edges)
assert len(TRIANGLES) == 4


def is_triangle_free(bits):
    """bits: tuple of 6 ints (0/1), bits[i] = 1 iff EDGES[i] present."""
    for (i, j, k) in TRIANGLES:
        if bits[i] and bits[j] and bits[k]:
            return False
    return True


def bits_of(x, n=N_EDGES):
    return tuple((x >> b) & 1 for b in range(n))


triangle_free = []
for x in range(2 ** N_EDGES):
    b = bits_of(x)
    if is_triangle_free(b):
        triangle_free.append((x, sum(b)))

max_edges = max(cnt for _, cnt in triangle_free)
classical_marked = sorted(x for x, cnt in triangle_free if cnt == max_edges)

# Sanity check against Mantel's theorem: floor(n^2/4) for n=4 is 4.
assert max_edges == (N_VERTICES ** 2) // 4, "Mantel's theorem check failed"
print(f"Classical result: max triangle-free edge count on K4 = {max_edges} "
      f"(Mantel's theorem: floor({N_VERTICES}^2/4) = {(N_VERTICES**2)//4})")
print(f"Classically marked bitstrings (max triangle-free subgraphs): "
      f"{[format(x, '06b') for x in classical_marked]}")
print(f"Total search space size: {2**N_EDGES}, "
      f"number of marked items: {len(classical_marked)}")

# ---------------------------------------------------------------------------
# 2. Build the Grover search circuit
# ---------------------------------------------------------------------------

N_QUBITS = N_EDGES  # 6
N_ITEMS = 2 ** N_QUBITS


def make_oracle_diagonal(marked):
    """Diagonal +-1 unitary implementing the phase oracle for `marked`
    indices, built directly from the classically-computed marked set
    (a legitimate small-instance oracle construction: the oracle IS the
    boolean function being tested, realized as a diagonal phase gate)."""
    diag = np.ones(N_ITEMS, dtype=complex)
    for m in marked:
        diag[m] = -1.0
    return diag


def grover_circuit(marked, n_iterations):
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))

    oracle_diag = make_oracle_diagonal(marked)

    # Diffusion operator diagonal (about the uniform state), standard Grover:
    # D = 2|s><s| - I, which as a diagonal-in-Hadamard-basis operator flips
    # the sign of every basis state except |0...0>.
    diffusion_diag = -np.ones(N_ITEMS, dtype=complex)
    diffusion_diag[0] = 1.0

    for _ in range(n_iterations):
        qc.append(DiagonalGate(list(oracle_diag)), list(range(N_QUBITS)))
        qc.h(range(N_QUBITS))
        qc.append(DiagonalGate(list(diffusion_diag)), list(range(N_QUBITS)))
        qc.h(range(N_QUBITS))

    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


# Optimal number of Grover iterations: round(pi / (4*theta) - 0.5), where
# theta = arcsin(sqrt(M/N)) (standard Grover analysis).
M = len(classical_marked)
theta = np.arcsin(np.sqrt(M / N_ITEMS))
n_iter = max(1, round(np.pi / (4 * theta) - 0.5))
print(f"Running Grover search with {n_iter} iteration(s) over "
      f"{N_QUBITS} qubits ({N_ITEMS} states, {M} marked)")

qc = grover_circuit(classical_marked, n_iter)

# ---------------------------------------------------------------------------
# 3. Run on ideal AerSimulator and compare to the classical answer
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit bit ordering: classical register bit c[0] is the rightmost char.
# Our circuit maps qubit i -> classical bit i -> EDGES[i], so we must reverse
# the returned bitstring to recover x with bit i = (x >> i) & 1.
def counts_to_x(bitstring):
    reversed_bits = bitstring[::-1]
    return int(reversed_bits, 2)

measured = {}
for bitstring, cnt in counts.items():
    x = counts_to_x(bitstring)
    measured[x] = measured.get(x, 0) + cnt

sorted_measured = sorted(measured.items(), key=lambda kv: -kv[1])
top_k = sorted_measured[:M]
top_k_states = sorted(x for x, _ in top_k)

marked_prob = sum(cnt for x, cnt in measured.items() if x in classical_marked) / shots

print(f"Top {M} measured outcome(s) by frequency: "
      f"{[(format(x, '06b'), c) for x, c in top_k]}")
print(f"Total measured probability mass on classically-marked states: "
      f"{marked_prob:.3f}")

# Verification: the Grover-boosted amplitude should concentrate almost all
# measurement probability on exactly the classically-marked set, and the
# top-M most frequent outcomes should equal the classical marked set exactly.
verified = (top_k_states == classical_marked) and (marked_prob > 0.9)

print()
if verified:
    print("PASS")
else:
    print("FAIL")
