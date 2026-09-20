"""
Erdos problem #630 (as recorded in manman4/erdosproblems, data/problems.yaml,
entry `number: "630"`) — a graph-theory / chromatic-number problem, tagged
["graph theory", "chromatic number"], with `oeis: ["N/A"]`.

LIMITATION, stated up front: problem #630 has NO associated OEIS sequence
(oeis field is literally "N/A"). There is therefore no OEIS integer sequence
to build a "quantum-testable sequence membership" circuit around, as the
brief asks for. Rather than fabricate a fake OEIS id or copy an unrelated
sequence, this script instead builds a genuine, small, finite, computable
property that comes directly from the problem's own tag ("chromatic
number"): 2-COLORABILITY (bipartiteness) of a small fixed graph, i.e.
whether a proper vertex coloring with k=2 colors exists.

Chosen instance
----------------
Graph G = path graph P3 with vertices {0, 1, 2} and edges {(0,1), (1,2)}.
This is a genuinely small, finite, fully enumerable search space: each
vertex gets 1 of 2 colors, so the search space has 2^3 = 8 candidate
colorings, encoded in 3 qubits.

Classical property tested
--------------------------
A coloring c: {0,1,2} -> {0,1} is PROPER iff c(0) != c(1) and c(1) != c(2)
(no edge is monochromatic). The classical answer -- computed from first
principles by brute-force enumeration in this script, not looked up --
is exactly the set of proper colorings. For P3 there are exactly 2 proper
2-colorings: 010 and 101 (in little-endian qubit order q0 q1 q2, these are
the bitstrings where q0 != q1 and q1 != q2).

Quantum method
---------------
Grover's algorithm on 3 qubits. The oracle marks exactly the bitstrings
satisfying the two inequality constraints above (a genuine arithmetic/
constraint oracle built from CNOT/X gates and a multi-controlled Z, not a
hard-coded "flip these known answer bits"). With 8 states and 2 marked
solutions, the optimal number of Grover iterations is 1
(floor(pi/4 * sqrt(8/2)) = 1). We run the circuit on the ideal AerSimulator,
take the most frequent measurement outcomes, and check that they are
exactly the classically-computed set of proper colorings -- i.e. that
Grover's algorithm correctly identifies which colorings are chromatically
valid for this graph, which is the finite computable property tied to
problem #630's "chromatic number" tag.

PASS/FAIL is decided by comparing the quantum result (top measured
bitstrings) against the classical brute-force answer.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from itertools import product

# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute-force enumerate proper 2-colorings of P3
#    (vertices 0-1-2, edges (0,1) and (1,2)), computed here from scratch.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2)]
N_VERTICES = 3


def is_proper_coloring(coloring):
    """coloring: tuple of 3 bits (c0, c1, c2); True iff no edge is monochromatic."""
    return all(coloring[u] != coloring[v] for (u, v) in EDGES)


classical_solutions = []
for bits in product([0, 1], repeat=N_VERTICES):
    if is_proper_coloring(bits):
        classical_solutions.append(bits)

# Represent solutions as bitstrings in qubit order q0 q1 q2 (q0 = vertex 0, etc.)
# Qiskit prints classical register bits with q(n-1) ... q1 q0 (MSB first == last
# qubit first), so a solution (c0, c1, c2) corresponds to printed string
# "c2 c1 c0".
classical_bitstrings = set(
    "".join(str(b) for b in reversed(sol)) for sol in classical_solutions
)

print("Graph: P3, vertices {0,1,2}, edges", EDGES)
print("Classical brute-force proper 2-colorings (c0,c1,c2):", classical_solutions)
print("Classical solution bitstrings (Qiskit order q2 q1 q0):", classical_bitstrings)

assert len(classical_solutions) == 2, "expected exactly 2 proper 2-colorings for P3"

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle for "coloring is proper", from the constraints
#    c0 != c1 and c1 != c2, directly -- not from the precomputed answer set.
# ---------------------------------------------------------------------------

n = N_VERTICES  # 3 qubits, one per vertex


def build_oracle_clean():
    """Phase-flip oracle marking states where q0!=q1 AND q1!=q2.

    Uses 2 work qubits computed via CNOT into (q0 XOR q1) and (q1 XOR q2);
    a CZ between the two work qubits applies the phase flip exactly when
    both constraints hold, then the work qubits are uncomputed.
    """
    qc = QuantumCircuit(n + 2, name="oracle")
    w1, w2 = n, n + 1

    qc.cx(0, w1)
    qc.cx(1, w1)
    qc.cx(1, w2)
    qc.cx(2, w2)

    qc.cz(w1, w2)

    # uncompute work qubits
    qc.cx(1, w2)
    qc.cx(2, w2)
    qc.cx(0, w1)
    qc.cx(1, w1)
    return qc


# ---------------------------------------------------------------------------
# 3. Full Grover circuit: superposition -> oracle -> diffusion -> measure
# ---------------------------------------------------------------------------

def diffusion_operator(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffusion")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


N = 2 ** n            # search space size = 8
M = len(classical_solutions)  # number of marked solutions = 2
import math
# Optimal Grover iteration count from theta = asin(sqrt(M/N)):
# k* = round( (pi / (4*theta)) - 0.5 ), clipped to >= 1.
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations: {iterations} (N={N}, M={M})")

qc = QuantumCircuit(n + 2, n)
qc.h(range(n))  # uniform superposition over the 3 vertex-color qubits

oracle = build_oracle_clean()
diff = diffusion_operator(n)

for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(n + 2))
    qc.append(diff.to_instruction(), range(n))

qc.measure(range(n), range(n))

# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

print("Measurement counts:", counts)

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_k = sorted_counts[:len(classical_solutions)]
top_bitstrings = set(bs for bs, _ in top_k)

top_probability_mass = sum(c for _, c in top_k) / shots
print("Top", len(classical_solutions), "measured bitstrings:", top_bitstrings,
      f"(probability mass {top_probability_mass:.3f})")

# ---------------------------------------------------------------------------
# 5. Compare quantum result to the classical answer and report PASS/FAIL
# ---------------------------------------------------------------------------

verified = (
    top_bitstrings == classical_bitstrings
    and top_probability_mass > 0.7  # Grover should strongly concentrate here
)

if verified:
    print("PASS: Grover search's top outcomes match the classical proper-2-coloring set.")
else:
    print("FAIL: Grover search's top outcomes do not match the classical answer.")

print("RESULT:", "PASS" if verified else "FAIL")
