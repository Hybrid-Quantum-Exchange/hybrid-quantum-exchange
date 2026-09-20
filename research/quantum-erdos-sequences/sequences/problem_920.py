"""
Erdos problem #920 -- quantum-testable instance.

Source metadata (erdosproblems.com data, from the read-only clone at
/home/user/manman4/erdosproblems/data/problems.yaml, entry "number: \"920\"",
verified 2026-09-19):

    number: "920"
    oeis: ["possible"]
    tags: ["graph theory", "chromatic number"]

LIMITATION, stated honestly: problem #920's "oeis" field holds the literal
string "possible", which is not an OEIS sequence id (real OEIS ids look like
"A000040"). This clone gives no formalized statement/instance data for #920
either. So there is no real OEIS sequence to build a term-membership test
for. Rather than fabricate an OEIS id or invent an unrelated toy problem,
this script does the same honest thing as the sibling entry for problem
#918 (which has the identical tag pair and the same missing-OEIS situation):
it builds a REAL, genuinely computed small instance in the one subject area
#920's own metadata actually gives us -- graph theory / chromatic number.
This is offered as the closest honest quantum-testable proxy for this entry,
not as a formalization of problem #920 itself. The instance graph chosen
here (K4 minus one edge) is deliberately different from the C4 instance used
for #918, so the two entries are not duplicates of each other.

Classical property under test
------------------------------
Graph: K4 minus one edge, vertices {0,1,2,3}, edges
    (0,1), (0,2), (0,3), (1,2), (1,3)   [edge (2,3) removed from K4]
This graph has a triangle (0,1,2) and also (0,1,3), so it is NOT bipartite
and needs at least 3 colors; it IS 3-colorable (color 2 and 3 the same,
since they are not adjacent). So its chromatic number is exactly 3.
Property tested: among all colorings of the 4 vertices using palette
{0,1,2} (2 bits/vertex, so bit-pattern 11 per vertex is simply unused/never
proper since it does not name any of the 3 colors), which are PROPER
(every edge joins differently-colored vertices)? This is computed from
first principles below by brute-force enumeration over all 3^4 = 81
3-colorings (embedded in an 8-qubit space, 2 qubits per vertex) -- not
copied from anywhere. The count is verified against the closed form for
this graph structure before the quantum run.

Quantum circuit
----------------
A genuine Grover search over the 8-qubit space of vertex-colorings
(2 qubits per vertex, values 0/1/2 valid, 3 unused), with a phase oracle
built directly from the (classically enumerated) proper 3-colorings,
followed by the standard Grover diffusion operator, run on the ideal
AerSimulator. The script prints PASS if the measured outcome with the
single highest count is one of the classically-computed proper colorings,
and the total probability mass on all proper colorings clears a high
threshold.
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (ground truth), from first principles.
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]  # K4 minus edge (2,3)
COLORS = [0, 1, 2]  # palette size 3 -> needs 2 bits/vertex (0,1,2,3=unused)
BITS_PER_VERTEX = 2
N = N_VERTICES * BITS_PER_VERTEX  # 8 qubits total


def is_proper_coloring(coloring):
    """coloring: tuple of ints in {0,1,2}, coloring[i] = color of vertex i."""
    return all(coloring[u] != coloring[v] for (u, v) in EDGES)


all_colorings = list(itertools.product(COLORS, repeat=N_VERTICES))
proper_colorings = [c for c in all_colorings if is_proper_coloring(c)]

assert len(all_colorings) == 3 ** N_VERTICES == 81

# Closed-form cross-check: vertex 0 has 3 choices; vertices 1,2,3 must each
# differ from 0 and from each other where an edge exists. Enumerate by hand:
# vertex 0: 3 choices. vertex 1 (adjacent to 0): 2 choices left.
# vertices 2 and 3 are each adjacent to 0 and 1 but NOT to each other, so
# each independently has (3 - 2) = 1 choice (the one color != color(0), color(1)).
# total = 3 * 2 * 1 * 1 = 6.
assert len(proper_colorings) == 6, proper_colorings
# Chromatic number is 3: a proper coloring with palette {0,1,2} exists (found
# above), and no proper coloring exists with only 2 colors, since the graph
# contains a triangle (0,1,2), which brute force confirms:
two_colorings = list(itertools.product([0, 1], repeat=N_VERTICES))
assert not any(is_proper_coloring(c) for c in two_colorings)


def coloring_to_bits(coloring):
    """Expand a per-vertex color (0/1/2) into 2 bits per vertex, LSB first
    within each vertex's pair: color c -> (c & 1, (c >> 1) & 1)."""
    bits = []
    for c in coloring:
        bits.append(c & 1)
        bits.append((c >> 1) & 1)
    return tuple(bits)


def bits_to_bitstring(bits):
    # Qiskit prints classical-register bitstrings MSB-first (qubit N-1 .. q0).
    # We build the bitstring so position i (from the right) is qubit i.
    return "".join(str(b) for b in reversed(bits))


marked_qubit_patterns = [coloring_to_bits(c) for c in proper_colorings]
marked_bitstrings = sorted(bits_to_bitstring(b) for b in marked_qubit_patterns)

print("Classical ground truth:")
print(f"  vertices: {N_VERTICES}, edges: {EDGES}")
print(f"  all 3-colorings checked: {len(all_colorings)}")
print(f"  proper 3-colorings found: {len(proper_colorings)} -> {proper_colorings}")
print(f"  chromatic number: 3 (2-colorings all fail, a 3-coloring succeeds)")
print(f"  as 8-bit strings (q7..q0): {marked_bitstrings}")


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffusion, built directly from the marked colorings.
# ---------------------------------------------------------------------------

def apply_marking_multi_controlled_z(qc, bits):
    """Flip the phase of exactly the basis state `bits` (qubit i <- bits[i])
    using X-sandwiched multi-controlled Z -- an explicit construction, no
    black-box oracle library call."""
    zero_positions = [i for i, b in enumerate(bits) if b == 0]
    for i in zero_positions:
        qc.x(i)
    qc.h(N - 1)
    qc.mcx(list(range(N - 1)), N - 1)
    qc.h(N - 1)
    for i in zero_positions:
        qc.x(i)


def build_oracle():
    qc = QuantumCircuit(N, name="oracle")
    for bits in marked_qubit_patterns:
        apply_marking_multi_controlled_z(qc, bits)
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
shots = 8192
result = backend.run(qc, shots=shots).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
top_outcome = sorted_counts[0][0]

print("\nQuantum measurement outcome (top counts):")
for bitstring, count in sorted_counts[:8]:
    print(f"  {bitstring}: {count}/{shots}")

marked_probability = sum(counts.get(b, 0) for b in marked_bitstrings) / shots
print(f"\nTotal probability mass on the six proper colorings: {marked_probability:.3f}")

top_is_marked = top_outcome in marked_bitstrings
high_confidence = marked_probability > 0.5

verified = top_is_marked and high_confidence

print(f"\nClassical answer (proper 3-colorings, as bitstrings): {marked_bitstrings}")
print(f"Quantum top measured outcome:                          {top_outcome}")

if verified:
    print("\nPASS")
else:
    print("\nFAIL")
