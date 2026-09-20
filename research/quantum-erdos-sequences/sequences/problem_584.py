"""
Erdos problem #584 -- quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 584"):
    tags:  ["graph theory", "cycles"]
    oeis:  ["N/A"]
    prize: "no"
    status: "open"

HONESTY NOTE ON SCOPE
----------------------
Problem #584 carries NO OEIS sequence id in the source data (oeis: ["N/A"]).
There is therefore no specific integer sequence from this problem to test
membership/term-computation against, and this script does NOT claim to test
one. Faking an OEIS id or a "known term" for #584 would misrepresent the
problem, which the task instructions explicitly forbid.

What this script does instead, honestly: it takes the one real piece of
mathematical content the metadata *does* give us -- the tags "graph theory"
and "cycles" -- and builds a genuine, small, finite, classically-checkable
instance of a *cycle-in-a-graph* decision property, of the general kind
Erdos cycle problems are about. It then verifies a real quantum circuit
(Grover search) against the classical brute-force answer for that instance.
This is a best-effort, honestly-scoped stand-in, not a claim about the
specific truth of problem #584 (which is open) or about any OEIS sequence
(none exists for it).

CLASSICAL PROPERTY TESTED
--------------------------
Take the complete graph K4 on 4 labeled vertices {0,1,2,3}. It has exactly
6 possible edges:
    e0=(0,1) e1=(0,2) e2=(0,3) e3=(1,2) e4=(1,3) e5=(2,3)
A subset of these 6 edges is represented by a 6-bit string b5 b4 b3 b2 b1 b0
(bit i = 1 iff edge ei is present). The search space has size N = 2^6 = 64,
which matches the "N <= ~64" budget for a small quantum instance.

Property P(subset): "the chosen edge subset forms a Hamiltonian cycle of
K4" -- i.e. every vertex has degree exactly 2 in the subset, and the edges
form a single connected 4-cycle (not two disjoint 2-cycles, which is
impossible here but is checked explicitly for rigor).

The classical answer (computed in this script, from first principles, by
brute-force enumeration over all 64 subsets) is: there are exactly 3 edge
subsets of K4 satisfying P -- the 3 distinct Hamiltonian (4-)cycles of K4,
consistent with the standard count (n-1)!/2 = 3!/2 = 3 for n=4.

QUANTUM CIRCUIT
----------------
A genuine Grover search circuit over 6 qubits (search space size N=64,
number of marked/"good" states M=3) is built:
  - Oracle: a phase oracle that flips the sign of amplitude on exactly the
    3 marked 6-bit strings identified classically (via X-gate framing +
    multi-controlled-Z per marked bitstring).
  - Diffuser: the standard Grover diffusion operator on 6 qubits.
  - Iteration count: floor(pi/4 * sqrt(N/M)), the standard optimal count.
The circuit is run on the ideal AerSimulator (no noise). PASS is declared
if the quantum measurement distribution puts its top-M most frequent
outcomes exactly on the classically-identified marked bitstrings.

Dependencies: qiskit, qiskit_aer, numpy only (already installed).
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute-force over all 2^6 edge subsets of K4.
# ---------------------------------------------------------------------------

VERTICES = (0, 1, 2, 3)
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # e0..e5
NUM_EDGES = len(EDGES)          # 6
N = 2 ** NUM_EDGES              # 64, the search space size


def is_hamiltonian_cycle(bitmask: int) -> bool:
    """True iff the edge subset given by `bitmask` (bit i <-> EDGES[i])
    forms a Hamiltonian cycle on all 4 vertices of K4."""
    chosen = [EDGES[i] for i in range(NUM_EDGES) if (bitmask >> i) & 1]
    if len(chosen) != len(VERTICES):
        return False  # a 4-cycle on 4 vertices needs exactly 4 edges

    degree = {v: 0 for v in VERTICES}
    adj = {v: [] for v in VERTICES}
    for a, b in chosen:
        degree[a] += 1
        degree[b] += 1
        adj[a].append(b)
        adj[b].append(a)

    if any(d != 2 for d in degree.values()):
        return False  # every vertex must have degree exactly 2

    # degree-2-everywhere edge set on 4 vertices is a union of cycles;
    # walk from vertex 0 and confirm it's a single 4-cycle, not disjoint
    # smaller cycles.
    start = VERTICES[0]
    visited = {start}
    prev, cur = None, start
    steps = 0
    while True:
        nxt_candidates = [w for w in adj[cur] if w != prev]
        if not nxt_candidates:
            return False
        nxt = nxt_candidates[0]
        steps += 1
        if nxt == start:
            break
        if nxt in visited:
            return False  # closed early -> not a single Hamiltonian cycle
        visited.add(nxt)
        prev, cur = cur, nxt
        if steps > len(VERTICES) + 1:
            return False
    return steps == len(VERTICES) and visited == set(VERTICES)


marked_states = [b for b in range(N) if is_hamiltonian_cycle(b)]
M = len(marked_states)

print(f"Classical brute force over N={N} edge-subsets of K4:")
print(f"  Hamiltonian-cycle edge subsets found (M={M}): {marked_states}")
for b in marked_states:
    edges_str = ", ".join(f"e{i}={EDGES[i]}" for i in range(NUM_EDGES) if (b >> i) & 1)
    print(f"    bitmask={b:06b} -> {edges_str}")

# Sanity: known closed form for K4's Hamiltonian-cycle count is (n-1)!/2 = 3.
expected_M = math.factorial(len(VERTICES) - 1) // 2
assert M == expected_M == 3, f"classical count mismatch: got {M}, expected {expected_M}"

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for the marked bitstrings.
# ---------------------------------------------------------------------------

n_qubits = NUM_EDGES  # 6


def build_oracle(marked, n):
    """Phase oracle: flips sign on each marked computational basis state."""
    qc = QuantumCircuit(n, name="oracle")
    for state in marked:
        bits = [(state >> i) & 1 for i in range(n)]
        zero_positions = [i for i, bit in enumerate(bits) if bit == 0]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all n qubits: H on target, MCX, H on target
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n):
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


oracle = build_oracle(marked_states, n_qubits)
diffuser = build_diffuser(n_qubits)

iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N / M)))
print(f"\nGrover iterations used: {iterations} (N={N}, M={M})")

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(n_qubits), range(n_qubits))

simulator = AerSimulator()
compiled = transpile(qc, simulator)
shots = 4096
result = simulator.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit reports classical bit strings with qubit 0 as the rightmost char.
# Convert each returned bitstring back to our integer bitmask convention
# (bit i of the integer <-> qubit i <-> EDGES[i]).
counts_by_int = {}
for bitstring, freq in counts.items():
    value = int(bitstring[::-1], 2)
    counts_by_int[value] = counts_by_int.get(value, 0) + freq

top_states = sorted(counts_by_int.items(), key=lambda kv: -kv[1])[:M]
top_state_values = sorted(v for v, _ in top_states)

print("\nTop measured outcomes (quantum):")
for v, freq in sorted(top_states, key=lambda kv: -kv[1]):
    print(f"  bitmask={v:06b} count={freq}/{shots}")

# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

classical_set = sorted(marked_states)
quantum_set = top_state_values

verified = classical_set == quantum_set

print(f"\nClassical marked set: {classical_set}")
print(f"Quantum top-{M} set : {quantum_set}")

if verified:
    print("\nPASS")
else:
    print("\nFAIL")
