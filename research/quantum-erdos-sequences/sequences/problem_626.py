"""
Erdos problem #626 -- quantum-testable companion script.

Source metadata (from erdosproblems.com data, data/problems.yaml, entry
"number: 626"):
    tags: ["graph theory", "chromatic number", "cycles"]
    oeis: ["possible"]

LIMITATION, stated honestly up front: problem #626's YAML record does not
carry a real OEIS sequence id. The field `oeis: ["possible"]` is the
repository's own placeholder meaning "an OEIS entry might exist / hasn't
been linked yet" -- it is not an id ("A......") that names an actual
sequence, and there is no other identifying data in the record. So this
script cannot test "membership in OEIS sequence A......" the way a
problem with a confirmed oeis id would.

Given that constraint, this is the best-effort honest fallback described
by the task: build a genuine, small, finite, computable decision problem
drawn directly from the problem's own tags ("graph theory", "chromatic
number", "cycles") -- namely, proper 2-colorability (chromatic number
question) of the 4-cycle graph C4 -- and verify a real Grover search
circuit against a from-scratch classical brute-force computation of that
property. This is NOT a claim that C4's colorings are "the sequence for
problem 626"; it is a tags-inspired substitute chosen because no OEIS
sequence is actually attached to this problem.

The property tested
--------------------
Let G = C4, the 4-cycle graph with vertices {0,1,2,3} and edges
{(0,1),(1,2),(2,3),(3,0)}. A proper 2-coloring assigns each vertex a bit
(color) such that every edge joins two different colors. Question: which
of the 2^4 = 16 possible colorings are proper 2-colorings of C4?

Classical computation (done in this script, first principles): brute
force over all 16 bitstrings, check every edge constraint. C4 is an even
cycle, hence bipartite, so it is 2-colorable, and by direct enumeration
below the classical script finds exactly the two proper colorings
0101 and 1010 (alternating around the cycle) -- consistent with the fact
that a cycle C_n has exactly 2 proper 2-colorings when n is even and 0
when n is odd (this "chromatic-number of cycles" fact is exactly what
the tags point at).

Quantum computation
--------------------
A Grover search circuit over the 4 data (vertex-color) qubits, with an
oracle built from an explicit small arithmetic circuit (CNOTs computing
each edge's XOR into ancillas, then a multi-controlled phase flip when
all four edge-XORs are 1, i.e. all edges properly colored), amplifies
exactly the marked "proper coloring" basis states. With N=16 and M=2
solutions, ~2 Grover iterations are optimal. The circuit is run on the
ideal AerSimulator; PASS requires the two most frequent measured
outcomes to be exactly the classical solution set {0101, 1010}.
"""

from itertools import product

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation, from first principles.
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # the 4-cycle C4


def is_proper_coloring(bits):
    """bits: tuple of 0/1, one color per vertex. True iff every edge's
    endpoints differ (a proper 2-coloring)."""
    return all(bits[u] != bits[v] for (u, v) in EDGES)


classical_solutions = [
    bits for bits in product([0, 1], repeat=N_VERTICES) if is_proper_coloring(bits)
]
# Expect exactly {(0,1,0,1), (1,0,1,0)} -- the two alternating colorings.
assert len(classical_solutions) == 2, (
    f"unexpected classical solution count: {classical_solutions}"
)

# Bit ordering note: Qiskit's classical register/measurement bitstrings are
# written with qubit 0 as the RIGHTMOST character. Our vertex qubits are
# v0..v3 mapped to circuit qubits 0..3, so a solution (b0,b1,b2,b3) for
# vertices 0,1,2,3 shows up in Qiskit's counts dict as the string
# "b3 b2 b1 b0".
classical_solution_strings = {
    "".join(str(b) for b in reversed(bits)) for bits in classical_solutions
}
print("Classical proper 2-colorings of C4 (vertex order v0v1v2v3):",
      [tuple(bits) for bits in classical_solutions])
print("Classical solution bitstrings (Qiskit order, v3v2v1v0):",
      sorted(classical_solution_strings))

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for proper 2-colorings of C4.
# ---------------------------------------------------------------------------

N_DATA = 4      # v0..v3 : one color bit per vertex
N_EDGE = 4      # e0..e3 : XOR ancilla for each of the 4 edges
N_FLAG = 1      # phase-kickback ancilla

DATA = list(range(0, N_DATA))
EDGE_ANC = list(range(N_DATA, N_DATA + N_EDGE))
FLAG = N_DATA + N_EDGE

N_QUBITS = N_DATA + N_EDGE + N_FLAG


def apply_oracle(qc: QuantumCircuit):
    """Marks (phase-flips) exactly the basis states where every one of the
    4 edges of C4 has differently-colored endpoints, via an explicit
    arithmetic circuit: compute each edge's XOR into a fresh ancilla with
    CNOTs, multi-controlled-X the flag qubit (prepared in |-> for phase
    kickback) on "all four edge XORs are 1", then uncompute the ancillas.
    """
    for i, (u, v) in enumerate(EDGES):
        qc.cx(DATA[u], EDGE_ANC[i])
        qc.cx(DATA[v], EDGE_ANC[i])

    qc.mcx(EDGE_ANC, FLAG)

    for i, (u, v) in enumerate(EDGES):
        qc.cx(DATA[v], EDGE_ANC[i])
        qc.cx(DATA[u], EDGE_ANC[i])


def apply_diffuser(qc: QuantumCircuit):
    """Standard Grover diffuser (inversion about the mean) over the
    N_DATA data qubits."""
    qc.h(DATA)
    qc.x(DATA)
    qc.h(DATA[-1])
    qc.mcx(DATA[:-1], DATA[-1])
    qc.h(DATA[-1])
    qc.x(DATA)
    qc.h(DATA)


qc = QuantumCircuit(N_QUBITS, N_DATA)

# Uniform superposition over all 16 colorings.
qc.h(DATA)

# Flag ancilla prepared in |-> for phase kickback.
qc.x(FLAG)
qc.h(FLAG)

# Optimal number of Grover iterations for N=16 states, M=2 solutions.
import math
n_states = 2 ** N_DATA
n_solutions = len(classical_solutions)
iterations = max(1, round((math.pi / 4) * math.sqrt(n_states / n_solutions)))
print(f"Grover iterations used: {iterations}")

for _ in range(iterations):
    apply_oracle(qc)
    apply_diffuser(qc)

qc.measure(DATA, list(range(N_DATA)))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
tqc = transpile(qc, sim)
SHOTS = 4096
result = sim.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
print("Top measured outcomes (bitstring: count):")
for bitstring, count in sorted_counts[:6]:
    print(f"  {bitstring}: {count}")

top_two = {bitstring for bitstring, _ in sorted_counts[:2]}

# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

verified = top_two == classical_solution_strings

if verified:
    print("PASS: Grover search's amplified outcomes match the classical "
          "set of proper 2-colorings of C4.")
else:
    print("FAIL: Grover search's amplified outcomes do NOT match the "
          "classical set of proper 2-colorings of C4.")
    print(f"  expected: {sorted(classical_solution_strings)}")
    print(f"  got top two: {sorted(top_two)}")
