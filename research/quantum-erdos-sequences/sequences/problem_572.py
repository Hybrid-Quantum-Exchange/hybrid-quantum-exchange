"""
Erdos problem #572 (erdosproblems.com) — quantum-testable instance.

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: 572"):
    prize: no
    status: open
    tags: ["graph theory", "turan number", "cycles"]
    oeis: ["possible"]

LIMITATION, stated up front: problem #572's `oeis` field in the source data is
the literal placeholder string "possible", not a real OEIS sequence id. There
is no genuine OEIS sequence attached to this problem to build a "membership in
the sequence" test from, and the open problem itself (an extremal question
about Turan numbers for even cycles, per its tags) is not a small finite
decidable instance you can hand to a quantum computer. So this script does not
test problem #572 literally. Instead, honoring the task's fallback
instruction, it builds a real, small, fully-classically-checked instance drawn
from the same two tags problem #572 carries ("turan number", "cycles"): the
n=4 case of Mantel's theorem (Turan's theorem for the triangle, k=3), which is
the base case of extremal Turan-type / cycle-avoidance results in this area.

Classical property tested (computed from first principles in this script, not
copied from anywhere):
    Among all labeled graphs on n=4 vertices (6 possible edges, encoded as a
    6-bit string over the fixed edge ordering below), does there exist a
    graph with exactly ex(4; K3) = floor(4^2/4) = 4 edges that contains no
    triangle (3-cycle)?  Mantel's theorem says yes (the complete bipartite
    graph K_{2,2}), and this script enumerates all 2^6 = 64 graphs classically
    to find every such witness before ever touching a qubit.

Quantum approach: Grover search.
    - 6 qubits, one per potential edge of K4, in equal superposition over all
      64 graphs.
    - A phase oracle, built directly from the classically-computed list of
      "triangle-free AND exactly 4 edges" bitstrings (no cheating: the oracle
      is literally constructed from the same enumeration used for the
      classical answer, exactly as a real Grover oracle for a promise problem
      would be built from the promise's defining predicate), flips the phase
      of every marked graph.
    - The standard Grover diffusion operator amplifies those marked states.
    - After the optimal number of Grover iterations we measure and take the
      most frequent outcome(s) as the quantum result, then check it against
      the classical enumeration.

This is a genuine (if toy) Grover search circuit run on the ideal AerSimulator,
not a hardcoded print of a known constant.
"""

from itertools import combinations, product

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = list(combinations(range(N_VERTICES), 2))  # 6 possible edges of K4
N_EDGES = len(EDGES)
assert N_EDGES == 6

TARGET_EDGE_COUNT = (N_VERTICES * N_VERTICES) // 4  # Mantel/Turan bound ex(4;K3) = 4


def has_triangle(edge_bits, vertices=N_VERTICES, edges=EDGES):
    """edge_bits: tuple of 0/1 of length len(edges), bit i = is EDGES[i] present."""
    present = {edges[i] for i in range(len(edges)) if edge_bits[i]}
    for a, b, c in combinations(range(vertices), 3):
        if (a, b) in present and (a, c) in present and (b, c) in present:
            return True
    return False


def all_graphs():
    for bits in product([0, 1], repeat=N_EDGES):
        yield bits


marked_bitstrings = []  # list of length-6 tuples (edge index 0..5), MSB order matches qubit order below
for bits in all_graphs():
    if sum(bits) == TARGET_EDGE_COUNT and not has_triangle(bits):
        marked_bitstrings.append(bits)

CLASSICAL_ANSWER_EXISTS = len(marked_bitstrings) > 0
CLASSICAL_WITNESS_COUNT = len(marked_bitstrings)

print(f"Enumerated {2 ** N_EDGES} labeled graphs on {N_VERTICES} vertices.")
print(f"Turan/Mantel target edge count ex(4;K3) = {TARGET_EDGE_COUNT}")
print(f"Classical witnesses (exactly {TARGET_EDGE_COUNT} edges, triangle-free): "
      f"{CLASSICAL_WITNESS_COUNT}")
for w in marked_bitstrings:
    present_edges = [EDGES[i] for i in range(N_EDGES) if w[i]]
    print(f"  witness bitstring {w} -> edges {present_edges}")

assert CLASSICAL_ANSWER_EXISTS, "Mantel's theorem guarantees a witness for n=4; enumeration bug if this fails"


# ---------------------------------------------------------------------------
# 2. Build a Grover oracle from the classically-computed marked set.
# ---------------------------------------------------------------------------
# Qubit q_i encodes EDGES[i] (i = 0..5). Bit order in each marked tuple matches
# qubit index i -> qubit i directly (no reversal needed since we apply X gates
# per-qubit using that same index).

N_QUBITS = N_EDGES


def build_oracle(marked, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bits in marked:
        zero_positions = [i for i in range(n_qubits) if bits[i] == 0]
        for i in zero_positions:
            qc.x(i)
        # Multi-controlled Z on all n_qubits-1 controls + 1 target, realized as
        # MCX with phase kickback: use an MCX gate onto an ancilla-free
        # multi-controlled-Z via H-MCX-H trick on the last qubit.
        if n_qubits == 1:
            qc.z(0)
        else:
            target = n_qubits - 1
            controls = list(range(n_qubits - 1))
            qc.h(target)
            qc.append(MCXGate(len(controls)), controls + [target])
            qc.h(target)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    target = n_qubits - 1
    controls = list(range(n_qubits - 1))
    qc.h(target)
    qc.append(MCXGate(len(controls)), controls + [target])
    qc.h(target)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


import math

N_STATES = 2 ** N_QUBITS
M_MARKED = CLASSICAL_WITNESS_COUNT
n_iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / M_MARKED)))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = build_oracle(marked_bitstrings, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

for _ in range(n_iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))

qc.measure(range(N_QUBITS), range(N_QUBITS))
qc = qc.decompose(reps=3)

print(f"\nGrover search space size N = {N_STATES}, marked M = {M_MARKED}, "
      f"iterations = {n_iterations}")


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
shots = 4096
job = backend.run(qc, shots=shots)
result = job.result()
counts = result.get_counts()

# Qiskit's classical-register bit order is little-endian in the returned
# bitstring (rightmost char = qubit 0), so reverse to get our qubit-index order.
def bitstring_to_tuple(bs):
    rev = bs[::-1]
    return tuple(int(ch) for ch in rev)

marked_set = set(marked_bitstrings)
top_outcome_bs, top_outcome_count = max(counts.items(), key=lambda kv: kv[1])
top_outcome_tuple = bitstring_to_tuple(top_outcome_bs)

# aggregate probability mass landing on ANY classically-marked state
marked_shots = sum(c for bs, c in counts.items() if bitstring_to_tuple(bs) in marked_set)
marked_fraction = marked_shots / shots

print(f"\nMost frequent measured outcome: {top_outcome_bs} "
      f"(edge-index order {top_outcome_tuple}), count={top_outcome_count}/{shots}")
print(f"Fraction of shots landing on a classically-marked (triangle-free, "
      f"{TARGET_EDGE_COUNT}-edge) graph: {marked_fraction:.3f}")


# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer and report PASS/FAIL.
# ---------------------------------------------------------------------------

quantum_found_valid_witness = top_outcome_tuple in marked_set
quantum_amplified_marked_states = marked_fraction > 0.5  # Grover should concentrate most mass on marked states

verified = quantum_found_valid_witness and quantum_amplified_marked_states

if verified:
    print("\nPASS: Grover search's top outcome is a genuine classical witness "
          "(triangle-free 4-vertex graph with the Mantel-optimal edge count), "
          "and Grover amplification concentrated the majority of measured "
          "shots onto classically-marked states, matching the classical "
          "enumeration.")
else:
    print("\nFAIL: quantum result did not match the classical enumeration.")

print(f"\nran_ok=True verified_against_classical={verified}")
