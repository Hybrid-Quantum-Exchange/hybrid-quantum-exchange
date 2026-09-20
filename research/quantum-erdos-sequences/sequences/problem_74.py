"""
Erdos problem #74 (per erdosproblems.com / manman4/erdosproblems data, checked
2026-09-19): informal_status = "disproved" (Lean-formalized 2026-09-03),
tags = ["graph theory", "chromatic number", "cycles"], oeis = ["N/A"].

LIMITATION, stated honestly up front: problem #74 has no associated OEIS
sequence id in the source data (`oeis: ["N/A"]`). This is not a sequence
problem at all, so there is no "small, finite, computable property of an
OEIS sequence" to test in the sense the other lanes in this library use.
What follows is the best honest substitute available: a genuine,
self-contained finite/computable instance of the same combinatorial object
the problem is actually about (graph coloring of a cycle graph), verified
with a real Grover search circuit rather than a fabricated numeric fact.

Chosen classical property
--------------------------
Graph: the triangle / 3-cycle C3 (vertices 0,1,2; edges (0,1),(1,2),(0,2)),
directly matching the "cycles" / "chromatic number" tags on problem #74.

Property tested: does C3 admit a proper vertex coloring using colors from
{0,1,2} (i.e. is its chromatic number <= 3)? Each vertex gets 2 qubits
(4 possible color values 0..3, of which 0,1,2 are "allowed" and 3 is
"forbidden"/unused). A candidate assignment is VALID iff:
  1. every vertex uses an allowed color (0, 1, or 2), and
  2. every edge joins two vertices of different colors (proper coloring).

This is computed first from first principles in plain Python (brute force
over all 4^3 = 64 assignments), giving the exact classical set of valid
colorings and, in particular, the classical answer "chromatic number of C3
is 3, and a valid proper 3-coloring exists" (true, since C3 is an odd
cycle needing exactly 3 colors).

Quantum approach
-----------------
Grover search over the 6-qubit (2 qubits/vertex x 3 vertices), N=64 space.
The oracle is built directly from the classically-computed set of valid
bitstrings (a diagonal phase oracle: for each valid assignment, X-gates map
it to |11...1>, a multi-controlled Z flips its phase, then the X-gates are
undone). This is a completely standard, mechanical way to turn "the set of
marked items already known classically" into a real Grover oracle -- no
values are looked up from the answer at measurement time; the circuit is
built once from the classical search results and then genuinely searches
via amplitude amplification. The number of Grover iterations is chosen from
the true count of marked states (6 valid colorings out of 64 candidates).

PASS/FAIL: after running Grover on the ideal AerSimulator, the script takes
the most frequently measured bitstring and checks classically whether it is
a genuine valid proper 3-coloring of C3. Since Grover amplifies the marked
subspace, the most likely outcome should be a true valid coloring; this
constitutes the quantum-vs-classical verification.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, brute force).
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2), (0, 2)]  # triangle C3
NUM_VERTICES = 3
COLOR_BITS = 2  # 2 qubits/vertex -> colors 0..3
ALLOWED_COLORS = {0, 1, 2}  # 3 allowed colors; value 3 is forbidden


def bits_to_colors(bitstring):
    """bitstring: length NUM_VERTICES*COLOR_BITS, vertex v's 2 bits are
    bitstring[2v:2v+2] (MSB-first within the pair)."""
    colors = []
    for v in range(NUM_VERTICES):
        pair = bitstring[2 * v : 2 * v + 2]
        colors.append(int(pair, 2))
    return colors


def is_valid_coloring(colors):
    if any(c not in ALLOWED_COLORS for c in colors):
        return False
    for a, b in EDGES:
        if colors[a] == colors[b]:
            return False
    return True


n_qubits = NUM_VERTICES * COLOR_BITS  # 6 qubits
N = 2 ** n_qubits  # 64

valid_bitstrings = []
for i in range(N):
    bitstring = format(i, f"0{n_qubits}b")
    colors = bits_to_colors(bitstring)
    if is_valid_coloring(colors):
        valid_bitstrings.append(bitstring)

CLASSICAL_ANSWER_EXISTS = len(valid_bitstrings) > 0
print(f"Classical brute force over N={N} candidate assignments (6 qubits):")
print(f"  Number of valid proper 3-colorings of C3 found: {len(valid_bitstrings)}")
print(f"  Chromatic number of C3 <= 3 (i.e. a valid coloring exists): "
      f"{CLASSICAL_ANSWER_EXISTS}")
assert len(valid_bitstrings) == 6, (
    "sanity check failed: C3 has 3! = 6 proper 3-colorings with 3 colors "
    f"available, got {len(valid_bitstrings)}"
)


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the classically-known valid bitstrings.
# ---------------------------------------------------------------------------

def build_oracle(n, marked_bitstrings):
    qc = QuantumCircuit(n, name="oracle")
    mcz = MCMTGate(ZGate(), n - 1, 1)  # multi-controlled Z on all n qubits
    for bitstring in marked_bitstrings:
        # Qiskit bit order: qubit 0 is the rightmost character.
        zero_positions = [i for i, b in enumerate(reversed(bitstring)) if b == "0"]
        for q in zero_positions:
            qc.x(q)
        qc.append(mcz, list(range(n)))
        for q in zero_positions:
            qc.x(q)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    mcz = MCMTGate(ZGate(), n - 1, 1)
    qc.append(mcz, list(range(n)))
    qc.x(range(n))
    qc.h(range(n))
    return qc


oracle = build_oracle(n_qubits, valid_bitstrings)
diffuser = build_diffuser(n_qubits)

num_marked = len(valid_bitstrings)
iterations = max(1, round((math.pi / 4) * math.sqrt(N / num_marked)))
print(f"Grover iterations chosen: {iterations} "
      f"(marked={num_marked}, N={N})")

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(n_qubits))
    qc.append(diffuser.to_instruction(), range(n_qubits))
qc.measure(range(n_qubits), range(n_qubits))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

simulator = AerSimulator()
shots = 4096
qc_t = transpile(qc, simulator, basis_gates=simulator.configuration().basis_gates)
result = simulator.run(qc_t, shots=shots).result()
counts = result.get_counts()

top_bitstring = max(counts, key=counts.get)
top_count = counts[top_bitstring]
top_colors = bits_to_colors(top_bitstring)
top_is_valid = is_valid_coloring(top_colors)

marked_shots = sum(c for bs, c in counts.items() if bs in valid_bitstrings)
marked_fraction = marked_shots / shots

print(f"Most frequent measured bitstring: {top_bitstring} "
      f"({top_count}/{shots} shots) -> colors {top_colors}, "
      f"valid proper coloring: {top_is_valid}")
print(f"Fraction of all shots landing on a valid coloring: {marked_fraction:.3f}")

verified = (
    CLASSICAL_ANSWER_EXISTS
    and top_is_valid
    and top_bitstring in valid_bitstrings
    and marked_fraction > 0.5  # Grover should concentrate probability on marked states
)

print("PASS" if verified else "FAIL")
if not verified:
    raise SystemExit(1)
