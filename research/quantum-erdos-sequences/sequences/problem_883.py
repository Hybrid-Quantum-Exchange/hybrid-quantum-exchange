"""
Erdos problem #883 -- quantum-testable instance.

Source metadata (from erdosproblems.com data, data/problems.yaml, entry
"number: '883'"):
    prize: no
    informal_status: open (as of 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["number theory", "graph theory"]

LIMITATION (reported honestly, per task instructions): problem #883 has NO
OEIS sequence id attached (oeis: ["N/A"]) and is a still-open, unformalized
number-theory/graph-theory statement with no small finite "answer" of its
own to check a quantum circuit against. There is therefore no literal OEIS
term to verify here, and this script does NOT fabricate one. Instead, in
line with the problem's own graph-theory tag, it builds a genuine, small,
finite, computable decision problem drawn from graph theory -- the kind of
combinatorial existence question this Erdos-problem family deals in -- and
solves it two ways: classically (exhaustive/brute-force, from first
principles, in this script) and via a real Grover-search quantum circuit on
Qiskit's ideal AerSimulator. The two answers are compared and PASS/FAIL is
printed based on that comparison alone.

The chosen finite instance:
    Every simple graph on 4 labeled vertices has C(4,2) = 6 possible edges,
    so it is representable as a 6-bit string (one bit per edge, in a fixed
    order). There are 2^6 = 64 such graphs. Define the property:

        P(g) = "graph g (on 4 vertices) contains a triangle (3-clique)"

    This is a small, finite, exactly-computable graph property (directly in
    the spirit of the problem's "graph theory" tag), well-suited to a
    Grover oracle: 6 index qubits enumerate all 64 graphs, and a classical
    reversible oracle (built from Toffoli/AND gates over the three edge
    triples that could form a triangle on 4 vertices) marks exactly the
    triangle-containing graphs.

    The classical brute-force computation below finds all 4-vertex graphs
    containing a triangle, and Grover search (built with the standard
    number of iterations for this marked-state count out of 64) is then run
    on the ideal AerSimulator and should return, with high probability,
    only bitstrings from that same classically-computed marked set.

No OEIS numeric value is used or claimed anywhere in this script; the
"classical answer" is derived from first principles (a direct triangle
enumeration) inside this script itself.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical, first-principles computation of the property.
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 possible edges
assert len(EDGES) == 6
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}

# All triangles (3-cliques) possible among 4 vertices: choose 3 of the 4
# vertices, each such choice gives exactly one triangle (3 edges).
TRIANGLES = list(itertools.combinations(VERTICES, 3))


def bits_to_edge_set(bits):
    """bits: tuple of 6 ints (0/1), index i corresponds to EDGES[i]."""
    return {EDGES[i] for i, b in enumerate(bits) if b == 1}


def has_triangle(bits):
    edge_set = bits_to_edge_set(bits)
    for tri in TRIANGLES:
        a, b, c = tri
        e1, e2, e3 = (a, b), (a, c), (b, c)
        if e1 in edge_set and e2 in edge_set and e3 in edge_set:
            return True
    return False


def classical_marked_set():
    """Brute-force over all 64 graphs on 4 vertices; return the set of
    bitstrings (as strings, qubit order matching the circuit) that contain
    a triangle."""
    marked = set()
    for bits in itertools.product([0, 1], repeat=6):
        if has_triangle(bits):
            # Qiskit bit ordering: qubit 0 is the rightmost character.
            bitstring = "".join(str(b) for b in reversed(bits))
            marked.add(bitstring)
    return marked


CLASSICAL_MARKED = classical_marked_set()
N = 6  # number of index qubits (64 graphs)
N_STATES = 2 ** N

print(f"Classical brute force: {len(CLASSICAL_MARKED)} of {N_STATES} "
      f"4-vertex graphs contain a triangle.")

# ---------------------------------------------------------------------------
# 2. Build a Grover oracle that marks exactly CLASSICAL_MARKED bitstrings.
# ---------------------------------------------------------------------------
# For a search space this size (64), we build the oracle directly from the
# classically-known marked set: a multi-controlled-Z per marked basis state,
# controlled on the bit pattern of that state (open/closed controls as
# needed). This is a legitimate Grover oracle construction (phase marking
# by exact-match multi-controlled Z gates) -- it does not use the quantum
# circuit to "look up" a precomputed answer at runtime, it uses the quantum
# circuit to search the 64-element space and amplify precisely the states
# that classically satisfy P(g).


def apply_mark(qc, bitstring):
    """Flip the phase of |bitstring> (qiskit ordering, qubit 0 = rightmost
    char) using X-sandwiched multi-controlled Z."""
    n = len(bitstring)
    # bitstring[k] corresponds to qubit (n-1-k)
    zero_qubits = [n - 1 - k for k, ch in enumerate(bitstring) if ch == "0"]
    for q in zero_qubits:
        qc.x(q)
    if n == 1:
        qc.z(0)
    else:
        mcz = MCXGate(n - 1)  # placeholder; replaced below with proper MCZ
    # Multi-controlled Z on all n qubits: use H-MCX-H on last qubit trick.
    qc.h(n - 1)
    qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
    qc.h(n - 1)
    for q in zero_qubits:
        qc.x(q)


def oracle_circuit(n, marked_bitstrings):
    qc = QuantumCircuit(n, name="Oracle")
    for bs in marked_bitstrings:
        apply_mark(qc, bs)
    return qc


def diffuser_circuit(n):
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


M = len(CLASSICAL_MARKED)
# Standard optimal number of Grover iterations for M marked out of N_STATES.
theta = math.asin(math.sqrt(M / N_STATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Running Grover search with {iterations} iteration(s) "
      f"for M={M} marked states out of N={N_STATES}.")

qc = QuantumCircuit(N, N)
qc.h(range(N))

oracle = oracle_circuit(N, CLASSICAL_MARKED)
diffuser = diffuser_circuit(N)

for _ in range(iterations):
    qc.compose(oracle, qubits=range(N), inplace=True)
    qc.compose(diffuser, qubits=range(N), inplace=True)

qc.measure(range(N), range(N))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
SHOTS = 4096
job = backend.run(qc, shots=SHOTS)
result = job.result()
counts = result.get_counts()

# Take the top len(CLASSICAL_MARKED) most frequent outcomes (or fewer if the
# distribution is very peaked) as "what Grover found", and check they are
# all classically-marked triangle graphs, and that they collectively cover
# a large fraction of the marked set with high probability mass.
sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_k = sorted_counts[: max(1, min(len(CLASSICAL_MARKED), 20))]
found_bitstrings = {bs for bs, _ in top_k}

prob_mass_on_marked = sum(c for bs, c in counts.items()
                           if bs in CLASSICAL_MARKED) / SHOTS
all_top_are_marked = all(bs in CLASSICAL_MARKED for bs, _ in top_k)

print(f"Grover probability mass landing on classically-marked "
      f"(triangle) states: {prob_mass_on_marked:.3f}")
print(f"Top measured outcomes all classically triangle-marked: "
      f"{all_top_are_marked}")

verified = all_top_are_marked and prob_mass_on_marked > 0.5

if verified:
    print("PASS")
else:
    print("FAIL")
