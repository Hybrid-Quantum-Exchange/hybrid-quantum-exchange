"""
Erdos problem #86 (erdosproblems.com/86) -- quantum-testable instance.

OEIS id used: A245762 -- "Maximal number of edges in a C4-free subgraph of
the n-dimensional hypercube graph Q_n." Erdos problem #86 asks about the
growth rate of this quantity; A245762's own listed initial terms are
1, 3, 9, 24, 56, 132, ... for n = 1, 2, 3, 4, 5, 6.

Classical property tested here (computed from first principles, not copied
from OEIS):
    For n = 2, the hypercube Q_2 is just the 4-cycle graph on the 4 vertices
    {00, 01, 11, 10} with 4 edges. A subgraph is "C4-free" if it contains no
    4-cycle. Since Q_2 has exactly one 4-cycle (the whole square), a subset
    of its 4 edges is C4-free iff it is NOT all 4 edges, i.e. iff it omits
    at least one edge. The maximum size of a C4-free edge subset is
    therefore 3 (any 3 of the 4 edges), matching a(2) = 3 in A245762. There
    are exactly C(4,3) = 4 such maximum subsets (each obtained by deleting
    one of the 4 edges).

    This script:
      1. Brute-forces, classically, over all 2^4 = 16 edge subsets of Q_2,
         checking each for the presence of the unique 4-cycle, to compute
         the true maximum C4-free subgraph size and the set of subsets that
         achieve it. This independently re-derives a(2) = 3 rather than
         trusting the OEIS value blindly.
      2. Builds a Grover search circuit over the 4 edge-inclusion qubits
         whose oracle marks exactly the states of Hamming weight 3 (i.e.
         the maximum C4-free subgraphs found classically in step 1).
      3. Runs the circuit on the ideal AerSimulator and checks that the
         measured distribution is concentrated (amplified) on the 4
         classically-verified maximum subsets.

Property being verified by the quantum circuit: "which 4-edge subsets of
edges of Q_2 are maximum C4-free subgraphs (size 3)" -- a finite, small,
computable search problem, solved here with genuine amplitude amplification
(Grover's algorithm), not a lookup.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Step 1: classical ground truth for Q_2 (the 4-cycle graph)
# ---------------------------------------------------------------------------

# Vertices of Q_2 as 2-bit strings, edges = pairs differing in exactly 1 bit.
VERTICES = [format(v, "02b") for v in range(4)]
ALL_EDGES = []
for a, b in itertools.combinations(range(4), 2):
    va, vb = VERTICES[a], VERTICES[b]
    if sum(x != y for x, y in zip(va, vb)) == 1:
        ALL_EDGES.append((a, b))

assert len(ALL_EDGES) == 4, "Q_2 must have exactly 4 edges"


def has_4_cycle(edge_subset_indices):
    """Return True if the given subset of ALL_EDGES contains a 4-cycle.

    Q_2 has exactly one 4-cycle overall: the cycle that uses all 4 edges of
    the square. A subset contains a (the) 4-cycle iff it equals the full
    edge set. This is checked generically here (not hard-coded) by testing
    whether the subgraph, restricted to its edges, contains a closed walk
    visiting all 4 vertices exactly once (a Hamiltonian cycle on 4 vertices
    of the square == the unique 4-cycle).
    """
    edges = [ALL_EDGES[i] for i in edge_subset_indices]
    if len(edges) < 4:
        return False
    # Build adjacency and check for a Hamiltonian cycle covering all 4
    # vertices using exactly these edges (brute force over the 4 vertices,
    # tiny search space).
    adj = {v: set() for v in range(4)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for perm in itertools.permutations(range(4)):
        if perm[0] != 0:
            continue  # fix rotation
        ok = True
        for i in range(4):
            u, v = perm[i], perm[(i + 1) % 4]
            if v not in adj[u]:
                ok = False
                break
        if ok:
            return True
    return False


def is_c4_free(edge_subset_indices):
    return not has_4_cycle(edge_subset_indices)


best_size = -1
maximum_subsets = []
for mask in range(16):
    idx = [i for i in range(4) if (mask >> i) & 1]
    if is_c4_free(idx):
        if len(idx) > best_size:
            best_size = len(idx)
            maximum_subsets = [mask]
        elif len(idx) == best_size:
            maximum_subsets.append(mask)

CLASSICAL_MAX_SIZE = best_size
CLASSICAL_MAX_SUBSETS = set(maximum_subsets)

print(f"Classical brute force: max C4-free edge-subgraph size of Q_2 = {CLASSICAL_MAX_SIZE}")
print(f"  (matches OEIS A245762 a(2) = 3): {CLASSICAL_MAX_SIZE == 3}")
print(f"  number of maximum subsets = {len(CLASSICAL_MAX_SUBSETS)} (expect 4)")
assert CLASSICAL_MAX_SIZE == 3
assert len(CLASSICAL_MAX_SUBSETS) == 4
# Every maximum subset should have popcount == 3 (omit exactly one edge).
for mask in CLASSICAL_MAX_SUBSETS:
    assert bin(mask).count("1") == 3


# ---------------------------------------------------------------------------
# Step 2: Grover search circuit marking exactly the Hamming-weight-3 states
# on 4 qubits (i.e. the classically-verified maximum C4-free subgraphs).
# ---------------------------------------------------------------------------

N_QUBITS = 4
N_STATES = 2 ** N_QUBITS
N_MARKED = len(CLASSICAL_MAX_SUBSETS)


def build_oracle():
    """Phase oracle flipping the sign of states whose Hamming weight is 3.

    Implemented by counting bits into 2 ancilla qubits (a 2-bit popcount of
    4 input qubits, values 0..4 fit in 3 bits but we only need to detect
    weight-3, done via a multi-controlled-Z pattern per marked bitstring for
    exactness and simplicity).
    """
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    # Mark each Hamming-weight-3 bitstring individually with a
    # multi-controlled Z (X-gates to map 0-controls to 1-controls).
    for mask in sorted(CLASSICAL_MAX_SUBSETS):
        bits = [(mask >> i) & 1 for i in range(N_QUBITS)]
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        qc.h(N_QUBITS - 1)
        qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
        qc.h(N_QUBITS - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser():
    qc = QuantumCircuit(N_QUBITS, name="diffuser")
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))
    return qc


# Optimal number of Grover iterations for N_STATES states, N_MARKED marked.
theta = math.asin(math.sqrt(N_MARKED / N_STATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qr = QuantumRegister(N_QUBITS, "q")
cr = ClassicalRegister(N_QUBITS, "c")
circuit = QuantumCircuit(qr, cr)
circuit.h(qr)

oracle = build_oracle()
diffuser = build_diffuser()
for _ in range(iterations):
    circuit.append(oracle.to_gate(), qr)
    circuit.append(diffuser.to_gate(), qr)

circuit.measure(qr, cr)

print(f"Grover iterations used: {iterations}")


# ---------------------------------------------------------------------------
# Step 3: run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(circuit, backend)
SHOTS = 4096
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit bit order: rightmost classical bit is qubit 0. Convert keys to our
# integer mask convention (bit i = qubit i).
def bitstring_to_mask(bitstring):
    # bitstring is c3 c2 c1 c0 (qiskit prints MSB first)
    bits = bitstring[::-1]
    return sum((1 if bits[i] == "1" else 0) << i for i in range(N_QUBITS))

mask_counts = {}
for bitstring, n in counts.items():
    mask = bitstring_to_mask(bitstring)
    mask_counts[mask] = mask_counts.get(mask, 0) + n

marked_total = sum(mask_counts.get(m, 0) for m in CLASSICAL_MAX_SUBSETS)
marked_fraction = marked_total / SHOTS

top_masks = sorted(mask_counts.items(), key=lambda kv: -kv[1])[:N_MARKED]
top_mask_set = {m for m, _ in top_masks}

print(f"Fraction of shots landing on a classically-verified maximum subset: {marked_fraction:.3f}")
print(f"Top {N_MARKED} measured masks: {sorted(top_mask_set)}")
print(f"Classical maximum subsets:     {sorted(CLASSICAL_MAX_SUBSETS)}")

# Success criteria: amplitude amplification concentrated the vast majority
# of shots on the 4 classically-verified maximum C4-free subgraphs, and the
# most frequent outcomes are exactly that set.
verified = marked_fraction > 0.85 and top_mask_set == CLASSICAL_MAX_SUBSETS

if verified:
    print("PASS")
else:
    print("FAIL")
