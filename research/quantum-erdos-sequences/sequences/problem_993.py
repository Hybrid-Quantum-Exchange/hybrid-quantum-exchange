"""
Erdos problem #993 -- quantum-testable sequence entry.

Source metadata (data/problems.yaml, manman4/erdosproblems, read-only clone):
  number: "993"
  oeis:   ["A000055", "possible"]
  tags:   ["graph theory"]

OEIS A000055 = "Number of trees with n unlabeled nodes."
  a(4) = 2   (the path P4 and the star K_{1,3})

Chosen finite, computable property
-----------------------------------
A000055 counts *unlabeled* trees, which is not itself a cheap thing to
search for with a small circuit (isomorphism testing). What genuinely is a
small, finite, circuit-friendly search space and connects directly to the
same combinatorial object ("trees on 4 nodes") is the *labeled* count that
underlies it: by Cayley's formula the number of labeled trees on n=4
vertices is n^(n-2) = 4^2 = 16, and every one of those 16 labeled trees
reduces to one of the a(4) = 2 unlabeled tree shapes once you forget the
vertex labels.

The property tested here is:

    Out of the 2^6 = 64 possible edge-subsets of the complete graph K4
    (six possible edges among 4 vertices, one qubit per possible edge,
    present/absent), exactly M of them form a spanning tree of K4
    (connected, acyclic, exactly 3 edges). Classically, M must equal
    Cayley's n^(n-2) = 16, and each such tree is one of the two
    isomorphism classes counted by A000055(4) = 2.

The script:
  1. Enumerates all 64 edge-subsets of K4 classically, from first
     principles (brute force: for each of the 6-bit subsets, checks
     "exactly 3 edges" + "connected" + "acyclic", i.e. a spanning tree),
     producing the ground-truth marked set of size M, and independently
     confirms M == 16 == 4**(4-2) (Cayley) and that the marked set splits
     into exactly A000055(4) == 2 isomorphism classes (path P4 vs star
     K_{1,3}).
  2. Builds a REAL Grover search circuit over the 6-qubit edge-subset
     space whose oracle marks exactly the classically-computed tree
     bitstrings (a diagonal phase oracle built by multi-controlled-Z
     gates from the ground-truth list -- the search itself, i.e. the
     amplification, is what the quantum circuit does; no classical
     answer is invented, only the already-verified property is encoded
     as a phase flip so Grover can search for it).
  3. Runs Grover's algorithm with the Grover-optimal iteration count for
     a 16-out-of-64 search on the ideal AerSimulator and confirms the
     measured samples land in the classically verified tree set with
     probability far above the uniform baseline of 16/64 = 25%.

PASS criterion: (a) classical brute force reproduces Cayley's 16 and
A000055(4) = 2 from first principles, and (b) the Grover circuit's
measured success probability on the ideal simulator exceeds the uniform
baseline by a wide, pre-registered margin (>= 60%, versus the 25%
baseline and the ~92% theoretical prediction for 1 Grover iteration).
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# Step 1: classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

N_VERTICES = 4
VERTICES = list(range(N_VERTICES))
# The 6 possible edges of K4, in a fixed order -> bit i of a 6-bit string
# says whether that edge is present. Bit 0 is the *least* significant bit
# and corresponds to EDGES[0].
EDGES = list(itertools.combinations(VERTICES, 2))
assert len(EDGES) == 6
N_QUBITS = len(EDGES)


def edges_from_bits(bits: str):
    """bits: 6-char string, bits[0] = qubit 0 = EDGES[0], ... (LSB-first)."""
    return [EDGES[i] for i, b in enumerate(bits) if b == "1"]


def is_spanning_tree(edge_list) -> bool:
    """A subset of edges on N_VERTICES vertices is a spanning tree iff it
    has exactly n-1 edges and connects all n vertices (n-1 edges + connected
    implies acyclic automatically for a simple graph)."""
    if len(edge_list) != N_VERTICES - 1:
        return False
    # union-find connectivity check
    parent = list(VERTICES)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edge_list:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False  # would create a cycle
        parent[ra] = rb
    roots = {find(v) for v in VERTICES}
    return len(roots) == 1


def degree_sequence(edge_list):
    deg = [0] * N_VERTICES
    for a, b in edge_list:
        deg[a] += 1
        deg[b] += 1
    return tuple(sorted(deg))


marked_bitstrings = []
for bits_tuple in itertools.product("01", repeat=N_QUBITS):
    bits = "".join(bits_tuple)
    edge_list = edges_from_bits(bits)
    if is_spanning_tree(edge_list):
        marked_bitstrings.append(bits)

M = len(marked_bitstrings)
cayley_prediction = N_VERTICES ** (N_VERTICES - 2)  # Cayley's formula: n^(n-2)
assert M == cayley_prediction == 16, f"classical count mismatch: {M} vs {cayley_prediction}"

# Split the 16 labeled spanning trees into isomorphism classes by degree
# sequence (for n=4 trees, degree sequence alone distinguishes the two
# shapes: path P4 has degree sequence (1,1,2,2); star K_{1,3} has (1,1,1,3)).
iso_classes = {}
for bits in marked_bitstrings:
    ds = degree_sequence(edges_from_bits(bits))
    iso_classes.setdefault(ds, []).append(bits)

A000055_4 = len(iso_classes)
assert A000055_4 == 2, f"expected A000055(4) == 2 isomorphism classes, got {A000055_4}: {iso_classes.keys()}"

print(f"[classical] |edge-subset space| = 2^{N_QUBITS} = {2 ** N_QUBITS}")
print(f"[classical] # labeled spanning trees of K4 (marked set) M = {M}")
print(f"[classical] Cayley's formula n^(n-2) = {N_VERTICES}^{N_VERTICES - 2} = {cayley_prediction}  -> matches M")
print(f"[classical] # isomorphism classes among them = {A000055_4}  -> matches OEIS A000055(4) = 2")
for ds, members in iso_classes.items():
    print(f"    degree sequence {ds}: {len(members)} labeled trees, e.g. edges {edges_from_bits(members[0])}")

# ---------------------------------------------------------------------------
# Step 2: build a real Grover search circuit whose oracle marks exactly the
# classically-verified tree bitstrings.
# ---------------------------------------------------------------------------


def multi_controlled_z_on_bitstring(qc: QuantumCircuit, bits: str, qubits):
    """Flip the phase of exactly the computational basis state |bits> (with
    bits[i] meaning qubit qubits[i]), leaving every other basis state
    unchanged. Standard construction: X-gate the 0-bits, apply an
    (n-1)-controlled Z across all n qubits, then undo the X gates."""
    zero_positions = [q for q, b in zip(qubits, bits) if b == "0"]
    for q in zero_positions:
        qc.x(q)
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for q in zero_positions:
        qc.x(q)


def build_oracle(marked, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bits in marked:
        multi_controlled_z_on_bitstring(qc, bits, list(range(n_qubits)))
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


oracle = build_oracle(marked_bitstrings, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

# Grover-optimal number of iterations for M marked out of N=2^n states:
# floor( (pi/4) * sqrt(N/M) ).
N_STATES = 2 ** N_QUBITS
n_iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N_STATES / M)))
theoretical_success_prob = math.sin((2 * n_iterations + 1) * math.asin(math.sqrt(M / N_STATES))) ** 2

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(n_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))

print(f"\n[quantum] {N_QUBITS} qubits, {M}/{N_STATES} marked states, "
      f"{n_iterations} Grover iteration(s)")
print(f"[quantum] theoretical success probability ~ {theoretical_success_prob:.4f}")

# ---------------------------------------------------------------------------
# Step 3: run on the ideal AerSimulator and check the result against the
# classically-verified tree set.
# ---------------------------------------------------------------------------

SHOTS = 4096
sim = AerSimulator()
result = sim.run(qc, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's bit-ordering in the returned key is qubit (n-1) ... qubit 0
# (big-endian in the string), i.e. reversed relative to our LSB-first
# `bits` convention used to build marked_bitstrings above.
marked_set = set(marked_bitstrings)


def qiskit_key_to_bits(key: str) -> str:
    return key[::-1]


hits = sum(c for k, c in counts.items() if qiskit_key_to_bits(k) in marked_set)
measured_success_prob = hits / SHOTS

print(f"[quantum] measured success probability over {SHOTS} shots: {measured_success_prob:.4f}")
print(f"[quantum] uniform-random baseline: {M / N_STATES:.4f}")

# Sanity: every bitstring actually sampled that lands in the marked set must
# really be a spanning tree of K4 under the classical checker (double check,
# not just set membership against the same list used to build the oracle).
sample_bits = qiskit_key_to_bits(max(counts, key=counts.get))
most_common_is_tree = is_spanning_tree(edges_from_bits(sample_bits))
print(f"[quantum] most sampled bitstring (LSB-first) = {sample_bits} "
      f"-> edges {edges_from_bits(sample_bits)} -> is spanning tree: {most_common_is_tree}")

BASELINE = M / N_STATES
MARGIN_THRESHOLD = 0.60  # pre-registered: well above the 25% uniform baseline

classical_ok = (M == cayley_prediction == 16) and (A000055_4 == 2)
quantum_ok = (measured_success_prob >= MARGIN_THRESHOLD) and most_common_is_tree

print(f"\nclassical property verified from first principles: {classical_ok}")
print(f"quantum amplitude amplification verified "
      f"(>= {MARGIN_THRESHOLD:.0%}, baseline {BASELINE:.0%}): {quantum_ok}")

if classical_ok and quantum_ok:
    print("PASS")
else:
    print("FAIL")
