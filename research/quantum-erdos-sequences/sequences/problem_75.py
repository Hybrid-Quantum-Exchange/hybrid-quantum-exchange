"""
Erdos problem #75 (erdosproblems.com) -- quantum-testable lane.

Source metadata (data/problems.yaml, erdosproblems mirror, checked 2026-09-19):
    number: "75"
    tags: ["graph theory", "chromatic number"]
    oeis: ["N/A"]

HONEST LIMITATION: problem #75 carries no OEIS sequence id ("N/A" in the
source data), so there is no "sequence" to compute a term of. Rather than
fabricate an OEIS value, this script instead builds a genuine small quantum
circuit around the one real, finite, computable object the metadata does
give us: the "chromatic number" tag. It tests 2-colorability (chromatic
number <= 2, i.e. bipartiteness) of a small fixed graph, which is exactly
the kind of graph-coloring decision problem #75's tag names, using Grover
search over the colorings -- a real quantum search, not a lookup.

Classical property under test
------------------------------
Graph: path P3 with vertices {0, 1, 2} and edges (0,1), (1,2).
A 2-coloring assigns each vertex a bit c0, c1, c2 in {0,1}. It is *proper*
iff adjacent vertices differ: c0 != c1 AND c1 != c2.

The script first brute-forces, in pure Python (first principles, no
libraries), all 2^3 = 8 colorings and determines the exact set of proper
ones. For P3 that set is {(1,0,1), (0,1,0)} written as bitstrings
"101" and "010" (qubit order q0 q1 q2 -> bitstring c0 c1 c2), confirming
P3 is bipartite, i.e. its chromatic number is 2.

Quantum circuit
----------------
A 3-qubit Grover search marks exactly the proper colorings via a phase
oracle built from the two inequality conditions (c0 XOR c1) AND (c1 XOR c2),
implemented with CX/X gates and a multi-controlled-Z, followed by the
standard 3-qubit diffusion operator. With N = 8 basis states and M = 2
marked states, one Grover iteration (near-optimal for this N, M) is run on
the ideal AerSimulator and the circuit is measured many times.

PASS/FAIL
---------
The script computes the classical set of proper colorings, runs the Grover
circuit, and checks that the two most frequent measured bitstrings are
exactly that classical set (i.e. the quantum search amplified precisely the
classically-verified answer). Prints PASS or FAIL accordingly.
"""

from itertools import product

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2)]  # path graph P3


def is_proper_coloring(c0, c1, c2):
    colors = {0: c0, 1: c1, 2: c2}
    return all(colors[u] != colors[v] for u, v in EDGES)


classical_solutions = set()
for c0, c1, c2 in product([0, 1], repeat=3):
    if is_proper_coloring(c0, c1, c2):
        # bitstring order matches qubit order q0 q1 q2 read as c0 c1 c2
        classical_solutions.add(f"{c0}{c1}{c2}")

print(f"Classical brute force over all {2**3} colorings of P3 (edges {EDGES}):")
print(f"  proper 2-colorings found: {sorted(classical_solutions)}")
assert classical_solutions == {"010", "101"}, "unexpected classical result"
print("  -> P3 is 2-colorable (bipartite), chromatic number = 2.\n")


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search marking exactly the proper colorings.
# ---------------------------------------------------------------------------

def grover_circuit():
    qc = QuantumCircuit(5, 3)
    # uniform superposition over the 3 color qubits
    qc.h([0, 1, 2])

    def apply_oracle(qc):
        # compute ancillas
        qc.cx(0, 3)
        qc.cx(1, 3)
        qc.cx(1, 4)
        qc.cx(2, 4)
        # phase flip when ancilla3 == ancilla4 == 1: a plain controlled-Z
        # between the two ancilla qubits.
        qc.cz(3, 4)
        # uncompute ancillas
        qc.cx(1, 4)
        qc.cx(2, 4)
        qc.cx(0, 3)
        qc.cx(1, 3)

    def apply_diffuser(qc):
        qc.h([0, 1, 2])
        qc.x([0, 1, 2])
        qc.h(2)
        qc.ccx(0, 1, 2)
        qc.h(2)
        qc.x([0, 1, 2])
        qc.h([0, 1, 2])

    # N=8, M=2 -> optimal iterations ~ round(pi/4 * sqrt(N/M)) = 1
    apply_oracle(qc)
    apply_diffuser(qc)

    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


qc = grover_circuit()
sim = AerSimulator()
job = sim.run(qc, shots=4096)
counts = job.result().get_counts()

# Qiskit returns bitstrings in reversed (little-endian) classical-bit order;
# our classical registers 0,1,2 map to c0,c1,c2 in that same order, but the
# printed string is c2 c1 c0, so reverse it back before comparing.
normalized_counts = {}
for bitstring, n in counts.items():
    fixed = bitstring[::-1]
    normalized_counts[fixed] = normalized_counts.get(fixed, 0) + n

top2 = sorted(normalized_counts, key=normalized_counts.get, reverse=True)[:2]
quantum_solutions = set(top2)

print("Grover search results (top measured colorings, normalized to c0c1c2 order):")
for bs in sorted(normalized_counts, key=normalized_counts.get, reverse=True):
    print(f"  {bs}: {normalized_counts[bs]} shots")
print(f"  top-2 amplified states: {sorted(quantum_solutions)}\n")

verified = quantum_solutions == classical_solutions

if verified:
    print("PASS: Grover search amplified exactly the classically-verified "
          "proper 2-colorings of P3, confirming chromatic number 2.")
else:
    print("FAIL: quantum result does not match the classical answer.")
