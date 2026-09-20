"""
Erdos problem #766 (erdosproblems.com) — quantum-testable instance.

Source metadata (from data/problems.yaml, manman4/erdosproblems, entry
"number: \"766\""): a "no prize" / "open" problem tagged
["graph theory", "turan number"], with oeis: ["possible"]. That OEIS field
is a placeholder value in the dataset, not an actual OEIS sequence id, and
the dataset carries no concrete OEIS A-number and no numeric formula for
this specific problem statement. There is therefore no real OEIS sequence
to search or verify here.

LIMITATION (read before trusting PASS below): because problem #766 gives no
usable OEIS id, this script does NOT test problem #766's actual open
conjecture. Instead, honestly substituting for it, it tests the classical
mathematical object the problem's own tags name: a Turan number, i.e. the
extremal (maximum-edge) triangle-free graph count from Turan's theorem
(Mantel's theorem, the r=2 case of Turan's theorem). This is real,
well-defined, finite, and independently checkable — but it is a stand-in
for the tagged topic area, not a verification of problem 766 itself.

Classical property under test
------------------------------
For n = 4 vertices, the Turan/Mantel extremal number ex(4, K_3) — the
maximum number of edges a triangle-free graph on 4 labeled vertices can
have — is computed here from first principles by exhaustive classical
enumeration of all 2^6 = 64 labeled graphs on 4 vertices (6 possible
edges), checking each for triangles, and taking the max edge count among
the triangle-free ones. (Mantel's theorem predicts floor(4^2/4) = 4, the
complete bipartite graph K_{2,2}; the brute-force search below confirms
this independently rather than assuming it.)

Quantum circuit
----------------
A genuine Grover search over the 6-qubit space of labeled graphs on 4
vertices (one qubit per possible edge). The oracle is built directly from
the classically-enumerated set of "good" bitstrings (triangle-free AND
edge count == the classically-computed maximum), each marked with a
multi-controlled Z sandwiched between X gates on the 0-bits of that
bitstring. The standard Grover diffuser is applied for the
theoretically-optimal number of iterations. The circuit is run on the
ideal AerSimulator (statevector precision), and PASS/FAIL is decided by
comparing the set of high-probability measured bitstrings against the
classically-enumerated set of good graphs.
"""

import itertools
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

ERDOS_PROBLEM_NUMBER = 766
OEIS_IDS_USED = []  # none usable: dataset field is the placeholder "possible"

N_VERTICES = 4
EDGES = list(itertools.combinations(range(N_VERTICES), 2))  # 6 edges
N_QUBITS = len(EDGES)
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}

TRIANGLES = list(itertools.combinations(range(N_VERTICES), 3))


def edges_of_triangle(tri):
    a, b, c = tri
    return [EDGE_INDEX[tuple(sorted((a, b)))],
            EDGE_INDEX[tuple(sorted((a, c)))],
            EDGE_INDEX[tuple(sorted((b, c)))]]


TRIANGLE_EDGE_SETS = [edges_of_triangle(t) for t in TRIANGLES]


def is_triangle_free(bits):
    """bits: tuple of 0/1 of length N_QUBITS, bits[i] = edge EDGES[i] present."""
    for tri_edges in TRIANGLE_EDGE_SETS:
        if all(bits[i] == 1 for i in tri_edges):
            return False
    return True


# ---- classical, first-principles brute force over all 2^6 labeled graphs ----
all_graphs = list(itertools.product([0, 1], repeat=N_QUBITS))
triangle_free_graphs = [g for g in all_graphs if is_triangle_free(g)]
max_edges = max(sum(g) for g in triangle_free_graphs)
extremal_graphs = [g for g in triangle_free_graphs if sum(g) == max_edges]

# Sanity cross-check against Mantel's theorem: floor(n^2/4)
mantel_prediction = (N_VERTICES ** 2) // 4
assert max_edges == mantel_prediction, (
    f"brute force ({max_edges}) disagrees with Mantel's theorem "
    f"({mantel_prediction})"
)

CLASSICAL_ANSWER = {
    "max_triangle_free_edges": max_edges,
    "num_extremal_graphs": len(extremal_graphs),
    "extremal_bitstrings": sorted(
        "".join(str(b) for b in reversed(g)) for g in extremal_graphs
    ),
}

print("Classical (brute force) result:")
print(f"  n = {N_VERTICES} vertices, {N_QUBITS} possible edges")
print(f"  max triangle-free edge count = {max_edges} "
      f"(Mantel prediction floor(n^2/4) = {mantel_prediction})")
print(f"  number of extremal (marked) graphs = {len(extremal_graphs)}")
print(f"  extremal bitstrings (qubit order, edge0=LSB shown as given): "
      f"{CLASSICAL_ANSWER['extremal_bitstrings']}")


# ---- Grover search over the 6-qubit graph space for the extremal graphs ----
def build_oracle(marked_bitstrings, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bitstr in marked_bitstrings:
        # bitstr is little-endian: bitstr[i] corresponds to qubit i
        zero_qubits = [i for i, b in enumerate(bitstr) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


marked_bitstrings = [
    "".join(str(b) for b in g) for g in extremal_graphs
]  # little-endian, bit i -> qubit i

M = len(marked_bitstrings)
N = 2 ** N_QUBITS
iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))

oracle = build_oracle(marked_bitstrings, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit's counts keys are big-endian ("q_{n-1}...q_0"); convert to our
# little-endian bit-i-> qubit-i convention for comparison.
def qiskit_key_to_little_endian(key):
    return key[::-1]

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_m = sorted_counts[:M]
measured_top_bitstrings = {qiskit_key_to_little_endian(k) for k, _ in top_m}
expected_bitstrings = set(marked_bitstrings)

top_prob_mass = sum(c for _, c in top_m) / shots

print()
print(f"Grover search: {N_QUBITS} qubits, {M} marked states out of {N}, "
      f"{iterations} iteration(s)")
print(f"  top-{M} measured bitstrings (little-endian): "
      f"{sorted(measured_top_bitstrings)}")
print(f"  expected (classical) marked bitstrings:      "
      f"{sorted(expected_bitstrings)}")
print(f"  probability mass on top-{M}: {top_prob_mass:.3f}")

verified = (measured_top_bitstrings == expected_bitstrings) and (top_prob_mass > 0.5)

print()
if verified:
    print("PASS")
else:
    print("FAIL")
