"""
Erdos problem #837 (erdosproblems.com) -- quantum-testable companion script.

Source metadata (data/problems.yaml, manman4/erdosproblems, read-only clone):
    number: "837"
    tags: ["graph theory", "hypergraphs"]
    oeis: ["N/A"]

LIMITATION, stated honestly up front: problem #837 has NO associated OEIS
sequence id in the source data (oeis: ["N/A"]). The task specification for
this lane asks to derive a property from "its OEIS sequence id(s) and tags"
-- there is no sequence id here, so there is nothing to test membership/terms
of. What *can* be honestly built from the real content that IS present (the
tags "graph theory" / "hypergraphs", and Erdos's well known interest in
hypergraph 2-colorability / "Property B") is a genuine, self-contained finite
decision problem in that same area, verified by both classical brute force
and a real Grover search circuit. This is NOT a claim about problem #837's
actual open conjecture (which remains open and is not resolved by this
script) -- it is a small, honestly-labeled, adjacent finite instance built
because no OEIS sequence exists to query.

The property tested: 2-colorability of a small 3-uniform hypergraph
("Property B" in Erdos's terminology -- can the vertices be 2-colored so
that no hyperedge is monochromatic?).

Concrete finite instance:
    6 vertices: 0,1,2,3,4,5  (6 qubits, search space size N = 2**6 = 64)
    6 hyperedges (3-element vertex subsets):
        E1 = {0, 1, 2}
        E2 = {2, 3, 4}
        E3 = {4, 5, 0}
        E4 = {1, 3, 5}
        E5 = {0, 3, 5}
        E6 = {1, 2, 4}

    A candidate coloring is a bitstring b0 b1 b2 b3 b4 b5 in {0,1}^6.
    A hyperedge is "monochromatic" under b if all three of its vertices get
    the same bit. The instance is Property-B-satisfiable (2-colorable) iff
    at least one bitstring makes ALL SIX hyperedges non-monochromatic.

Classical ground truth is computed in this script by brute force over all
64 assignments (first principles, no OEIS lookup, no hard-coded answer).

Quantum method: Grover's algorithm. A reversible oracle is built directly
from the hyperedge structure (not from a precomputed lookup table): for each
hyperedge it computes, into an ancilla, whether the edge is monochromatic
(via a pair of multi-controlled-X gates catching the all-1 and the all-0
cases), then flags "good" states as those where every edge-ancilla is 0,
and applies a phase flip to exactly those states. This is wrapped in the
standard number of Grover iterations (pi/4 * sqrt(N/M) with M estimated from
the classical count) and run on the ideal AerSimulator. The most frequent
sampled bitstring is compared against the classical set of valid colorings.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# Instance definition
# ---------------------------------------------------------------------------
N_VERTICES = 6
EDGES = [(0, 1, 2), (2, 3, 4), (4, 5, 0), (1, 3, 5), (0, 3, 5), (1, 2, 4)]
N = 2 ** N_VERTICES


def is_monochromatic(bits, edge):
    a, b, c = (bits[v] for v in edge)
    return a == b == c


def is_valid_coloring(bits):
    """bits: tuple of 0/1 of length N_VERTICES, qubit i is bits[i]."""
    return not any(is_monochromatic(bits, e) for e in EDGES)


# ---------------------------------------------------------------------------
# Step 1: classical ground truth, brute force, first principles
# ---------------------------------------------------------------------------
def classical_brute_force():
    valid = []
    for x in range(N):
        bits = tuple((x >> i) & 1 for i in range(N_VERTICES))
        if is_valid_coloring(bits):
            valid.append(x)
    return valid


VALID_COLORINGS = classical_brute_force()
M = len(VALID_COLORINGS)  # number of "good" (marked) states
print(f"Classical brute force over {N} colorings of {N_VERTICES} vertices, "
      f"{len(EDGES)} hyperedges {EDGES}:")
print(f"  Property-B-valid (non-monochromatic on both edges) count M = {M}")
print(f"  e.g. first few valid bitstrings (v0 v1 v2 v3 v4): "
      f"{[format(x, '0' + str(N_VERTICES) + 'b')[::-1] for x in VALID_COLORINGS[:5]]}")

assert 0 < M < N, "instance must be neither trivially unsatisfiable nor all-satisfying"

# ---------------------------------------------------------------------------
# Step 2: Grover oracle built directly from the hyperedge structure
# ---------------------------------------------------------------------------
N_EDGES = len(EDGES)
DATA = list(range(N_VERTICES))                      # qubits 0..4
ANC = list(range(N_VERTICES, N_VERTICES + N_EDGES))  # one ancilla per edge
OUT = N_VERTICES + N_EDGES                           # phase-kickback output qubit
NQ = N_VERTICES + N_EDGES + 1


def mark_edge_ancilla(qc, edge, anc_qubit):
    """Set anc_qubit to 1 iff the three `edge` qubits are all equal (000 or 111)."""
    qc.mcx(list(edge), anc_qubit)              # catches |111>
    for v in edge:
        qc.x(v)
    qc.mcx(list(edge), anc_qubit)              # catches |000> (after inversion)
    for v in edge:
        qc.x(v)


def build_oracle():
    qc = QuantumCircuit(NQ, name="oracle")
    for e, a in zip(EDGES, ANC):
        mark_edge_ancilla(qc, e, a)
    # OUT qubit is prepared in |-> so an X controlled on "all edge-ancillas == 0"
    # phase-kicks exactly the good states. Flip ancillas so "0 (non-mono)" -> control 1.
    for a in ANC:
        qc.x(a)
    qc.mcx(ANC, OUT)
    for a in ANC:
        qc.x(a)
    # uncompute ancillas
    for e, a in zip(EDGES, ANC):
        mark_edge_ancilla(qc, e, a)
    return qc


def build_diffuser():
    qc = QuantumCircuit(N_VERTICES, name="diffuser")
    qc.h(DATA)
    qc.x(DATA)
    qc.h(DATA[-1])
    qc.mcx(DATA[:-1], DATA[-1])
    qc.h(DATA[-1])
    qc.x(DATA)
    qc.h(DATA)
    return qc


def build_grover_circuit(n_iterations):
    qc = QuantumCircuit(NQ, N_VERTICES)
    qc.h(DATA)
    qc.x(OUT)
    qc.h(OUT)

    oracle = build_oracle()
    diffuser = build_diffuser()

    for _ in range(n_iterations):
        qc.append(oracle.to_instruction(), range(NQ))
        qc.append(diffuser.to_instruction(), DATA)

    qc.h(OUT)
    qc.x(OUT)
    qc.measure(DATA, list(range(N_VERTICES)))
    return qc.decompose()


n_iter = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover iterations used: {n_iter} (N={N}, M={M})")

circuit = build_grover_circuit(n_iter)

# ---------------------------------------------------------------------------
# Step 3: run on the ideal AerSimulator
# ---------------------------------------------------------------------------
sim = AerSimulator()
SHOTS = 4096
result = sim.run(circuit, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit bit order: classical register bit c_i <- DATA[i]; returned key is
# c4 c3 c2 c1 c0 (rightmost = qubit 0). Convert to our integer encoding
# x = sum bits[i] * 2**i, matching classical_brute_force().
def key_to_x(key):
    bits = key[::-1]  # bits[i] corresponds to qubit i
    return sum(int(bits[i]) << i for i in range(N_VERTICES))

sampled_x = {key_to_x(k): c for k, c in counts.items()}
most_likely_x = max(sampled_x, key=sampled_x.get)
most_likely_prob = sampled_x[most_likely_x] / SHOTS

marked_prob_mass = sum(c for x, c in sampled_x.items() if x in VALID_COLORINGS) / SHOTS

print(f"Most sampled bitstring (as integer x): {most_likely_x} "
      f"= {format(most_likely_x, '0'+str(N_VERTICES)+'b')[::-1]} (v0..v{N_VERTICES-1}), "
      f"empirical prob {most_likely_prob:.3f}")
print(f"Total probability mass on classically-valid colorings: {marked_prob_mass:.3f}")

# ---------------------------------------------------------------------------
# Step 4: verify quantum result against classical ground truth
# ---------------------------------------------------------------------------
top_is_valid = most_likely_x in VALID_COLORINGS
amplified = marked_prob_mass > (M / N) + 0.15  # Grover should amplify well above baseline

verified = top_is_valid and amplified

print()
if verified:
    print("PASS")
else:
    print("FAIL")

RAN_OK = True
VERIFIED_AGAINST_CLASSICAL = verified
