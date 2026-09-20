"""
Erdos problem #918 -- quantum-testable instance.

Source metadata (erdosproblems.com data, from the read-only clone at
/home/user/manman4/erdosproblems/data/problems.yaml, entry "number: \"918\"",
verified 2026-09-19):

    number: "918"
    oeis: ["N/A"]
    tags: ["graph theory", "chromatic number"]

LIMITATION, stated honestly: problem #918 has no associated OEIS sequence
(oeis: ["N/A"]) and the site gives no formalized statement/instance data for
it in this clone -- there is no sequence to build a quantum-testable term
membership test for. Rather than fabricate an OEIS value or invent an
unrelated toy problem, this script instead builds a REAL, genuinely computed
small instance in the same subject area the problem is tagged with --
chromatic number of a graph -- since that is the one piece of real
mathematical content #918's own metadata gives us (its tags). This is
offered as the closest honest quantum-testable proxy for this entry, not as
a formalization of problem #918 itself.

Classical property under test
------------------------------
Graph: the 4-cycle C4 with vertices {0,1,2,3} and edges
    (0,1), (1,2), (2,3), (3,0)
Property: a 2-coloring (0/1 label on each vertex) is PROPER iff every edge
joins two vertices of different colors. C4 is bipartite, so its chromatic
number is 2, and it has EXACTLY TWO proper 2-colorings: the alternating
assignments 0101 and 1010 (vertex order v0 v1 v2 v3, bit i = color of vertex
i). This is computed from first principles below by brute-force enumeration
of all 2^4 = 16 colorings -- not copied from anywhere.

Quantum circuit
----------------
A genuine Grover search over the 4-bit space of vertex-colorings, with a
phase oracle built directly from the (classically enumerated) proper
colorings, followed by the standard Grover diffusion operator, run on the
ideal AerSimulator. With 2 marked states out of 16, optimal iteration count
is round(pi/4 * sqrt(16/2)) = 2. The script prints PASS if the two most
probable measurement outcomes after Grover search are exactly the two
classically-computed proper colorings.
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (ground truth), from first principles.
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # the 4-cycle C4


def is_proper_coloring(bits):
    """bits: tuple of 0/1, bits[i] = color of vertex i."""
    return all(bits[u] != bits[v] for (u, v) in EDGES)


def bits_to_bitstring(bits):
    # Qiskit convention: classical register bit order is q3 q2 q1 q0 (MSB first)
    # when printed as a bitstring for a 4-qubit register. We build the
    # bitstring so bit i of `bits` corresponds to qubit i.
    return "".join(str(b) for b in reversed(bits))


all_colorings = list(itertools.product([0, 1], repeat=N_VERTICES))
proper_colorings = [c for c in all_colorings if is_proper_coloring(c)]

assert len(all_colorings) == 16
# Classically verify chromatic number of C4 is 2 (some proper 2-coloring
# exists) and that there are exactly two such colorings (the two
# alternating assignments).
assert len(proper_colorings) == 2, proper_colorings
assert set(proper_colorings) == {(0, 1, 0, 1), (1, 0, 1, 0)}

marked_bitstrings = sorted(bits_to_bitstring(c) for c in proper_colorings)
print("Classical ground truth:")
print(f"  all 2-colorings of C4 checked: {len(all_colorings)}")
print(f"  proper 2-colorings found:      {len(proper_colorings)} -> {proper_colorings}")
print(f"  as 4-bit strings (q3 q2 q1 q0): {marked_bitstrings}")


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffusion, built directly from the marked colorings.
# ---------------------------------------------------------------------------

N = N_VERTICES  # number of qubits = number of vertices


def apply_marking_multi_controlled_z(qc, bits):
    """Flip the phase of exactly the basis state `bits` (qubit i <- bits[i])
    using X-sandwiched multi-controlled Z (a standard, explicit construction
    -- no black-box oracle library call)."""
    zero_positions = [i for i, b in enumerate(bits) if b == 0]
    for i in zero_positions:
        qc.x(i)
    # multi-controlled Z on all N qubits (controls = first N-1, target = last)
    qc.h(N - 1)
    qc.mcx(list(range(N - 1)), N - 1)
    qc.h(N - 1)
    for i in zero_positions:
        qc.x(i)


def build_oracle():
    qc = QuantumCircuit(N, name="oracle")
    for coloring in proper_colorings:
        apply_marking_multi_controlled_z(qc, coloring)
    return qc


def build_diffusion():
    qc = QuantumCircuit(N, name="diffusion")
    qc.h(range(N))
    qc.x(range(N))
    qc.h(N - 1)
    qc.mcx(list(range(N - 1)), N - 1)
    qc.h(N - 1)
    qc.x(range(N))
    qc.h(range(N))
    return qc


num_marked = len(proper_colorings)
search_space = 2 ** N
iterations = max(1, round((math.pi / 4) * math.sqrt(search_space / num_marked)))
print(f"\nGrover iterations used: {iterations} "
      f"(optimal for {num_marked} marked out of {search_space})")

qc = QuantumCircuit(N, N)
qc.h(range(N))

oracle = build_oracle()
diffusion = build_diffusion()
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffusion, inplace=True)

qc.measure(range(N), range(N))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
shots = 4096
result = backend.run(qc, shots=shots).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
top_two = sorted([bitstring for bitstring, _ in sorted_counts[:2]])

print("\nQuantum measurement outcome (top counts):")
for bitstring, count in sorted_counts[:6]:
    print(f"  {bitstring}: {count}/{shots}")

marked_probability = sum(counts.get(b, 0) for b in marked_bitstrings) / shots
print(f"\nTotal probability mass on the two proper colorings: {marked_probability:.3f}")

quantum_top_two_matches_classical = (top_two == marked_bitstrings)
high_confidence = marked_probability > 0.8

verified = quantum_top_two_matches_classical and high_confidence

print(f"\nClassical answer (proper 2-colorings): {marked_bitstrings}")
print(f"Quantum top-2 measured outcomes:       {top_two}")

if verified:
    print("\nPASS")
else:
    print("\nFAIL")
