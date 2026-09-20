"""
Erdos problem #146 -- quantum-testable lane.

Source metadata (data/problems.yaml, erdosproblems clone):
    number: 146
    tags: ["graph theory", "turan number"]
    informal_status: disproved (counterexample to a degeneracy conjecture)
    oeis: ["N/A"]   <-- NO OEIS sequence id is recorded for this problem.

LIMITATION (read before trusting the "PASS"):
    Problem #146 has no associated OEIS sequence in the source data, so the
    task of "identify a small computable property of the OEIS sequence" does
    not literally apply here -- there is no sequence to test membership in.
    This script is the best honest substitute: it stays faithful to the
    problem's actual mathematical subject matter (graph theory / degeneracy /
    Turan-type extremal counting on small graphs) and builds a genuine
    finite, computable, quantum-searchable property drawn from that subject,
    rather than fabricating or borrowing an unrelated OEIS value.

Chosen classical property:
    Let K4 be the complete graph on 4 labeled vertices, with its 6 possible
    edges indexed 0..5:
        e0=(0,1) e1=(0,2) e2=(0,3) e3=(1,2) e4=(1,3) e5=(2,3)
    Each 6-bit string b in {0,1}^6 encodes an edge subset (a subgraph of K4).
    Define the property P(b): "b has exactly 3 edges set AND the resulting
    graph is acyclic" -- i.e. b encodes a spanning tree of K4 (a forest with
    n-1 = 3 edges on n = 4 vertices is exactly a spanning tree). This is a
    genuine extremal/counting question in the same family as the problem's
    tags (graph theory, Turan-type edge-extremal counting, and degeneracy:
    a forest has degeneracy <= 1, which is the exact notion problem #146's
    counterexample concerns).

    By Cayley's formula the number of labeled spanning trees of K4 is
    4^(4-2) = 16. The script independently re-derives this classically from
    first principles (brute-force enumeration + a union-find cycle check
    over all 64 subsets of the 6 edges), rather than citing the formula.

Quantum circuit:
    A genuine Grover search over the 6-qubit space of the 64 possible edge
    subsets of K4. The oracle marks exactly the computational basis states
    whose bitstring is a spanning tree (as determined classically above) by
    applying a multi-controlled Z gate per marked bitstring. Grover
    diffusion is applied for the optimal number of iterations for
    N = 64, M = 16 marked states (1 iteration). The circuit is run on the
    ideal AerSimulator and the measurement distribution is checked against
    the classically-enumerated set of spanning-tree bitstrings: PASS
    requires that the highest-probability measured outcomes are exactly (up
    to sampling noise) the marked (spanning-tree) states, with amplified
    probability well above the uniform baseline of 16/64 = 0.25.
"""

import itertools
import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ----------------------------------------------------------------------
# 1. Classical ground truth, derived from first principles.
# ----------------------------------------------------------------------

N_VERTICES = 4
EDGES = list(itertools.combinations(range(N_VERTICES), 2))  # 6 edges of K4
N_EDGES = len(EDGES)
assert N_EDGES == 6


def is_spanning_tree(bits):
    """bits: tuple of 0/1 of length N_EDGES, edge i present iff bits[i]==1.

    A spanning tree of K4 needs exactly N_VERTICES-1 = 3 edges and must be
    acyclic (equivalently, connected with 3 edges on 4 vertices). Checked
    with a plain union-find, no formula assumed.
    """
    edge_list = [EDGES[i] for i in range(N_EDGES) if bits[i] == 1]
    if len(edge_list) != N_VERTICES - 1:
        return False
    parent = list(range(N_VERTICES))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (u, v) in edge_list:
        ru, rv = find(u), find(v)
        if ru == rv:
            return False  # cycle
        parent[ru] = rv
    # acyclic with exactly n-1 edges on n vertices => spanning tree
    return True


classical_marked = []
for bits in itertools.product([0, 1], repeat=N_EDGES):
    if is_spanning_tree(bits):
        classical_marked.append(bits)

CLASSICAL_COUNT = len(classical_marked)
CAYLEY_PREDICTED = N_VERTICES ** (N_VERTICES - 2)  # Cayley's formula, for cross-check only
print(f"Classical brute-force spanning-tree count for K4: {CLASSICAL_COUNT}")
print(f"Cayley's formula 4^(4-2) predicts: {CAYLEY_PREDICTED}")
assert CLASSICAL_COUNT == CAYLEY_PREDICTED == 16, "classical derivation disagrees with Cayley's formula"

# Bitstrings (as strings, index 0 = qubit 0 = least-significant / e0) for the oracle.
marked_bitstrings = set()
for bits in classical_marked:
    # bits[i] corresponds to edge i / qubit i; build a string with qubit 0 as
    # rightmost character to match Qiskit's little-endian classical register
    # convention used after measurement.
    s = "".join(str(bits[i]) for i in reversed(range(N_EDGES)))
    marked_bitstrings.add(s)

assert len(marked_bitstrings) == 16

# ----------------------------------------------------------------------
# 2. Quantum circuit: Grover search for the marked (spanning-tree) states.
# ----------------------------------------------------------------------

N_QUBITS = N_EDGES  # 6
N_STATES = 2 ** N_QUBITS  # 64
M_MARKED = CLASSICAL_COUNT  # 16


def build_oracle(bitstrings, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bs in bitstrings:
        # bs[0] is qubit n-1 ... bs[-1] is qubit 0 (see construction above)
        zero_positions = [n_qubits - 1 - i for i, c in enumerate(bs) if c == "0"]
        for q in zero_positions:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_positions:
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


# Optimal number of Grover iterations for N states, M marked.
theta = math.asin(math.sqrt(M_MARKED / N_STATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations used: {iterations} (N={N_STATES}, M={M_MARKED})")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = build_oracle(marked_bitstrings, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))

qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 8192
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# ----------------------------------------------------------------------
# 3. Compare quantum result to the classical answer.
# ----------------------------------------------------------------------

marked_shots = sum(c for bs, c in counts.items() if bs in marked_bitstrings)
unmarked_shots = shots - marked_shots
marked_prob = marked_shots / shots
baseline_prob = M_MARKED / N_STATES  # uniform-random baseline

top_states = Counter(counts).most_common(M_MARKED)
top_states_are_marked = all(bs in marked_bitstrings for bs, _ in top_states)

print(f"Total shots: {shots}")
print(f"Shots landing on a spanning-tree state: {marked_shots} ({marked_prob:.3f})")
print(f"Uniform-random baseline probability: {baseline_prob:.3f}")
print(f"Top-{M_MARKED} most frequent outcomes are all spanning-tree states: {top_states_are_marked}")

verified = (
    marked_prob > 2 * baseline_prob  # Grover amplification clearly above chance
    and top_states_are_marked
)

if verified:
    print("PASS")
else:
    print("FAIL")
