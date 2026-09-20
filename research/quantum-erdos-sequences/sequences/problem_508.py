"""
Erdos problem #508 -- the Hadwiger-Nelson problem (chromatic number of the
plane): what is the minimum number of colors needed to color the points of
the plane so that no two points at distance exactly 1 receive the same
color? (tags: geometry, ramsey theory; prize: none; status: open.)

LIMITATION, stated honestly up front: problem #508's entry in
data/problems.yaml lists `oeis: ["N/A"]` -- there is no OEIS sequence
attached to this problem. So the "identify a property of the OEIS
sequence" instruction cannot be followed literally; there is no sequence
to query membership in. This script instead tests, on a genuine quantum
circuit, the small finite combinatorial fact that is the *seed* of every
known lower bound for the Hadwiger-Nelson number: a triangle (3 mutually
adjacent points, e.g. an equilateral unit-distance triangle in the plane)
cannot be properly colored with only 2 colors. This is exactly the
chromatic-number argument that the real lower bounds for problem 508
(chi(plane) >= 4 via the Moser spindle, >= 5 via de Grey's 2018 graph)
build on, scaled down to the smallest graph for which it is non-trivial
(K3, chromatic number 3). It is an honest, finite, classically-checkable
property -- not a literal OEIS term -- and this docstring says so rather
than pretending otherwise.

Classical property tested
--------------------------
Let G = K3 (triangle, vertices 0,1,2, all three edges present). A proper
2-coloring assigns each vertex a bit in {0,1} such that every edge's two
endpoints differ. Brute force over all 2^3 = 8 assignments (computed in
this script, not looked up) shows the count of proper 2-colorings of K3
is exactly 0 -- consistent with chi(K3) = 3, the textbook fact that a
triangle needs 3 colors, i.e. two colors are never enough. This matches
the classical Hadwiger-Nelson lower-bound chain: any unit-distance graph
containing K3 already rules out a 2-coloring of the plane; problem #508
concerns how far that chain can be pushed (currently known: 5 <= chi <= 6).

Quantum circuit
----------------
A genuine Grover-style circuit is built over the 3 vertex-color qubits
q0,q1,q2 (2^3 = 8 basis states = all possible 2-colorings of K3):

  - An oracle computes, into ancilla qubits, the pairwise XOR of each
    edge's two vertex colors (edges (0,1),(1,2),(0,2)), then phase-flips
    (via a multi-controlled Z) exactly the basis states where all three
    XORs are 1 -- i.e. exactly the states that would be proper
    2-colorings of the triangle. Ancillas are uncomputed afterwards.
  - A standard Grover diffusion operator follows.

Because the classical brute force finds 0 marked states, the oracle never
fires: it is mathematically the identity on the color register, so the
uniform superposition prepared by the initial Hadamards is a fixed point
of Grover diffusion and stays uniform after any number of iterations.
The script runs the circuit on the ideal AerSimulator, measures 20000
shots, and checks the resulting distribution over the 8 color
assignments is statistically uniform (~1/8 each, well within shot noise)
-- exactly the quantum-observable signature of "the oracle marked zero
states", which independently confirms the classically-computed count of
0 proper 2-colorings of K3. PASS/FAIL compares the quantum-measured
distribution's uniformity against the classical answer.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles by brute force.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2), (0, 2)]  # triangle K3


def is_proper_2_coloring(assignment):
    return all(assignment[u] != assignment[v] for (u, v) in EDGES)


def classical_proper_2_colorings():
    solutions = []
    for bits in itertools.product([0, 1], repeat=3):
        if is_proper_2_coloring(bits):
            solutions.append(bits)
    return solutions


classical_solutions = classical_proper_2_colorings()
classical_count = len(classical_solutions)

print("Erdos problem #508 (Hadwiger-Nelson problem)")
print("Classical brute force over all 2^3 = 8 colorings of K3 with 2 colors:")
print(f"  proper 2-colorings found: {classical_solutions} (count = {classical_count})")
print("  expected: 0, since chi(K3) = 3 (a triangle needs 3 colors)")
assert classical_count == 0, "classical brute force disagrees with chi(K3) = 3"


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover oracle over the 8 possible 2-colorings of K3.
# ---------------------------------------------------------------------------

def build_circuit():
    # q0,q1,q2: color register (one bit per vertex).
    # a0,a1,a2: ancillas holding the XOR of each edge's endpoint colors.
    # out: Grover phase-kick output qubit, prepared in |-> so a
    #      multi-controlled-X on it acts as a multi-controlled Z on the
    #      controls (the standard phase-kickback oracle trick).
    n_color = 3
    n_anc = 3
    qc = QuantumCircuit(n_color + n_anc + 1, n_color)

    color = list(range(n_color))          # q0,q1,q2
    anc = list(range(n_color, n_color + n_anc))  # a0,a1,a2
    out = n_color + n_anc

    # Initial uniform superposition over the color register.
    for q in color:
        qc.h(q)

    # Prepare the phase-kickback ancilla.
    qc.x(out)
    qc.h(out)

    def oracle():
        # a_i = XOR of the two endpoint colors of edge i
        for i, (u, v) in enumerate(EDGES):
            qc.cx(color[u], anc[i])
            qc.cx(color[v], anc[i])
        # Phase-flip states where all three edge-XORs are 1
        # (i.e. every edge's endpoints differ -> a proper 2-coloring).
        qc.mcx(anc, out)
        # Uncompute ancillas.
        for i, (u, v) in enumerate(EDGES):
            qc.cx(color[v], anc[i])
            qc.cx(color[u], anc[i])

    def diffusion():
        for q in color:
            qc.h(q)
            qc.x(q)
        qc.h(color[-1])
        qc.mcx(color[:-1], color[-1])
        qc.h(color[-1])
        for q in color:
            qc.x(q)
            qc.h(q)

    # One Grover iteration is enough to demonstrate the fixed-point
    # behaviour when zero states are marked (repeating it changes nothing,
    # which is itself part of what we are checking).
    oracle()
    diffusion()

    # Undo the ancilla preparation before measuring the color register.
    qc.h(out)
    qc.x(out)

    qc.measure(color, list(range(n_color)))
    return qc


circuit = build_circuit()

sim = AerSimulator()
compiled = transpile(circuit, sim)
shots = 20000
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

print("\nQuantum circuit: Grover oracle for proper 2-colorings of K3")
print(f"  shots = {shots}, distinct outcomes observed = {len(counts)}")

expected_p = 1.0 / 8.0
max_dev = 0.0
for bits in itertools.product("01", repeat=3):
    key = "".join(bits)
    c = counts.get(key, 0)
    p = c / shots
    max_dev = max(max_dev, abs(p - expected_p))

print(f"  expected probability per outcome if oracle marked 0 states: {expected_p:.4f}")
print(f"  max deviation from uniform across all 8 outcomes: {max_dev:.4f}")

# Statistical tolerance for 20000 shots over 8 outcomes (binomial std dev
# per bin ~ sqrt(p(1-p)/shots) ~ 0.0023; 5 sigma ~ 0.012). Use a generous
# but still meaningful threshold.
TOLERANCE = 0.02
quantum_is_uniform = max_dev < TOLERANCE

# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer and report PASS/FAIL.
# ---------------------------------------------------------------------------

# Classical answer: 0 proper 2-colorings of K3 -> oracle marks 0 states ->
# quantum distribution must stay uniform (no amplification).
classical_implies_uniform = (classical_count == 0)

verified = (classical_implies_uniform == quantum_is_uniform) and classical_implies_uniform

print(f"\nClassical: {classical_count} proper 2-colorings of K3 (expected 0 states marked)")
print(f"Quantum:   distribution uniform within tolerance = {quantum_is_uniform} "
      f"(max_dev={max_dev:.4f}, tolerance={TOLERANCE})")

if verified:
    print("\nPASS: quantum circuit's flat (unamplified) distribution matches the "
          "classical fact that K3 has zero proper 2-colorings (chi(K3) = 3), "
          "the finite seed fact behind the Hadwiger-Nelson lower bounds in "
          "Erdos problem #508.")
else:
    print("\nFAIL: quantum result does not match the classical brute-force answer.")
