"""
Erdos problem #620 -- quantum-testable lane.

Source metadata (data/problems.yaml, erdosproblems clone):
    number: "620"
    tags: ["graph theory"]
    comments: "Erdos-Rogers problem"
    oeis: ["possible"]   <-- NOT a real OEIS id. The upstream metadata for
                              problem 620 carries the literal placeholder
                              string "possible" in the oeis field, not an
                              actual sequence identifier (no A-number is
                              given anywhere in the record for #620).

LIMITATION (stated honestly, per instructions): because problem #620 has no
real OEIS sequence id attached in the source data, this script cannot test
"membership in OEIS sequence A......." the way most lanes in this library
do. Rather than fabricate an OEIS id or copy a value with no derivation,
this script instead builds a genuine small, finite, computable instance of
the actual mathematical content behind the Erdos-Rogers problem's tag
("graph theory") and its classical Turan/Ramsey-adjacent flavor: extremal
triangle-free graphs.

Chosen finite, computable property
-----------------------------------
Consider all labeled graphs on n = 4 vertices. There are C(4,2) = 6 possible
edges, so there are 2^6 = 64 candidate graphs, each encoded as a 6-bit
string (one bit per potential edge). Define the classical predicate

    P(g) := g is triangle-free  AND  g has exactly 4 edges

By Turan's theorem / Mantel's theorem, the maximum number of edges in a
triangle-free graph on 4 vertices is floor(4^2/4) = 4, achieved (uniquely,
up to labeling) by the complete bipartite graph K_{2,2}. So P(g) is
satisfied by exactly the 4-edge triangle-free labeled graphs on 4 vertices
-- a small, well-defined, and independently checkable classical fact that
this script verifies itself by brute force before ever touching a qubit.

We then use Grover's algorithm on 6 qubits (one qubit per edge) to search
the 64-element space for graphs satisfying P(g), and check that the
quantum search recovers exactly the classically-computed marked set (as
the dominant measurement outcomes).

This is a real Grover circuit (oracle + diffusion, built from the
classically-enumerated marked bitstrings, run on the ideal AerSimulator),
not a lookup table dressed up as a quantum computation.
"""

from itertools import combinations, product

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, derived from first principles.
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = list(combinations(range(N_VERTICES), 2))  # 6 possible edges
N_EDGES = len(EDGES)
assert N_EDGES == 6

TRIANGLES = list(combinations(range(N_VERTICES), 3))  # 4 possible triangles


def edge_index(u, v):
    return EDGES.index((min(u, v), max(u, v)))


def is_triangle_free(bits):
    """bits: tuple of 0/1, length N_EDGES, bits[i] = 1 iff EDGES[i] present."""
    for (a, b, c) in TRIANGLES:
        if bits[edge_index(a, b)] and bits[edge_index(b, c)] and bits[edge_index(a, c)]:
            return False
    return True


def satisfies_P(bits):
    return sum(bits) == 4 and is_triangle_free(bits)


all_graphs = list(product([0, 1], repeat=N_EDGES))
marked_bits = [g for g in all_graphs if satisfies_P(g)]

# Sanity: Turan/Mantel says the max triangle-free edge count on 4 vertices
# is floor(4^2/4) = 4, uniquely realized (up to labeling) by K_{2,2}.
max_triangle_free_edges = max(sum(g) for g in all_graphs if is_triangle_free(g))
assert max_triangle_free_edges == 4, "Mantel's theorem check failed"

# Encode each marked bitstring as an integer (bit i -> qubit i, little-endian
# to match Qiskit's convention where qubit 0 is the least-significant bit).
def bits_to_int(bits):
    val = 0
    for i, b in enumerate(bits):
        if b:
            val |= (1 << i)
    return val


marked_ints = sorted(bits_to_int(b) for b in marked_bits)

print("Classical brute force over all 64 labeled graphs on 4 vertices:")
print(f"  Triangle-free graphs with exactly 4 edges (marked set): {len(marked_ints)}")
print(f"  Marked integers (6-bit edge-presence encoding): {marked_ints}")

CLASSICAL_ANSWER = set(marked_ints)


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffusion, built from the marked set above.
# ---------------------------------------------------------------------------

def build_oracle(n_qubits, marked_values):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for val in marked_values:
        bits = [(val >> i) & 1 for i in range(n_qubits)]
        # Flip qubits that should be 0 so the marked pattern becomes all-1s.
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(i)
        # Multi-controlled Z on all n_qubits (phase flip when all are |1>).
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        # Undo the X flips.
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(i)
    return qc


def build_diffusion(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffusion")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


N = N_EDGES  # 6 qubits
M = len(marked_ints)  # number of marked items out of 2^N = 64
N_STATES = 2 ** N

# Optimal number of Grover iterations for M marked items out of N_STATES.
theta = np.arcsin(np.sqrt(M / N_STATES))
iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

oracle = build_oracle(N, marked_ints)
diffusion = build_diffusion(N)

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(iterations):
    qc.compose(oracle, range(N), inplace=True)
    qc.compose(diffusion, range(N), inplace=True)
qc.measure(range(N), range(N))

print(f"\nGrover circuit: {N} qubits, {M} marked items / {N_STATES} states, "
      f"{iterations} iteration(s).")

sim = AerSimulator()
shots = 4096
job = sim.run(qc, shots=shots)
result = job.result()
counts = result.get_counts()

# Qiskit's classical register bit-ordering has bit 0 (qubit 0) as the
# rightmost character in the returned bitstring.
def counts_key_to_int(key):
    return int(key[::-1], 2)

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_results = sorted_counts[: max(M, 1) * 2]  # look at a generous top slice

print("\nTop measurement outcomes (bitstring: count):")
for key, cnt in top_results[:10]:
    val = counts_key_to_int(key)
    print(f"  {key} (edge-set {val:2d}, marked={val in CLASSICAL_ANSWER}): {cnt}")

# ---------------------------------------------------------------------------
# 3. Verify: Grover should overwhelmingly concentrate probability on the
#    classically-computed marked set.
# ---------------------------------------------------------------------------

marked_shots = sum(cnt for key, cnt in counts.items() if counts_key_to_int(key) in CLASSICAL_ANSWER)
marked_fraction = marked_shots / shots

# Also check which *distinct* outcomes appear among the top-M measured
# bitstrings, and whether they match the classical marked set.
top_M_ints = {counts_key_to_int(key) for key, _ in sorted_counts[:M]}
distinct_match = top_M_ints == CLASSICAL_ANSWER

print(f"\nFraction of shots landing on a classically-marked graph: {marked_fraction:.3f}")
print(f"Top-{M} measured outcomes match classical marked set exactly: {distinct_match}")

# Grover amplification should push the marked fraction well above the
# uniform baseline (M / N_STATES), and ideally the top-M outcomes should
# be exactly the marked set.
baseline = M / N_STATES
verified = distinct_match and marked_fraction > baseline * 3

if verified:
    print("\nPASS: Grover search on the ideal AerSimulator recovered exactly the "
          "classically-verified set of maximal triangle-free graphs on 4 vertices "
          "(the K_{2,2}-type extremal graphs behind the Erdos-Rogers / Turan "
          "context of problem #620).")
else:
    print("\nFAIL: quantum search result did not match the classical answer.")

print(f"\nran_ok=True verified_against_classical={verified}")
