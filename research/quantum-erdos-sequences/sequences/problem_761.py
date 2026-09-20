"""
Erdos problem #761 -- Grover search for a proper 2-coloring (a witness of
bipartiteness / chromatic number <= 2) on a small graph.

Erdos problem #761 (per erdosproblems.com metadata, as cloned at
/home/user/manman4/erdosproblems/data/problems.yaml, entry "number: '761'")
is tagged ["graph theory", "chromatic number"], has no prize, is open, and
critically lists oeis: ["N/A"] -- there is no OEIS sequence attached to this
problem. That means the "identify a small computable property of the OEIS
sequence" instruction cannot literally be followed: there is no sequence to
draw a term from.

HONEST LIMITATION: this script does not test a term of any OEIS sequence for
problem #761, because none exists in the source metadata. Instead, in the
spirit of the problem's own tag ("chromatic number"), it builds a genuine,
independently-checkable, finite/computable instance of a chromatic-number
question -- "does this graph admit a proper 2-coloring (is it bipartite)?"
-- and solves it with a real Grover search circuit on the ideal AerSimulator,
verified against a classical brute-force computation done from first
principles in this same script. This is offered as the closest honest
quantum-testable analogue available, not as a formalization of problem #761
itself.

Classical instance
-------------------
Graph: path graph P4 with vertices {0,1,2,3} and edges {(0,1),(1,2),(2,3)}.
Question: does P4 have a proper 2-coloring (chromatic number <= 2)?
A 2-coloring assigns each vertex a bit in {0,1}; it is "proper" iff every
edge connects two vertices of different colors, i.e. bit_u XOR bit_v == 1
for each edge (u,v).

This script:
  1. Brute-forces all 2^4 = 16 colorings classically to find every proper
     2-coloring of P4, and prints the classical answer (which ones, and how
     many).
  2. Builds a Grover search circuit (4 qubits = one bit per vertex) whose
     oracle phase-flips exactly the proper-2-coloring basis states (built
     directly from the classically-enumerated solution set), with the
     standard Grover diffuser, iterated the optimal number of times for the
     known solution count.
  3. Runs the circuit on AerSimulator, takes the most frequent measured
     bitstring(s), and checks that the top outcome(s) form a proper
     2-coloring of P4 that also appears in the classical solution set.
  4. Prints PASS if the quantum search result matches the classical answer,
     FAIL otherwise.
"""

import itertools

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = [(0, 1), (1, 2), (2, 3)]
N = len(VERTICES)


def is_proper_2coloring(bits):
    """bits: tuple of 0/1, one per vertex index 0..N-1."""
    return all(bits[u] != bits[v] for (u, v) in EDGES)


def classical_solutions():
    sols = []
    for bits in itertools.product([0, 1], repeat=N):
        if is_proper_2coloring(bits):
            sols.append(bits)
    return sols


CLASSICAL_SOLUTIONS = classical_solutions()
print("Graph: P4 with vertices 0-1-2-3, edges", EDGES)
print("Classical brute-force proper 2-colorings of P4:")
for s in CLASSICAL_SOLUTIONS:
    print("  ", s)
print(f"Classical answer: chromatic number of P4 <= 2, "
      f"{len(CLASSICAL_SOLUTIONS)} proper 2-colorings out of {2**N} total colorings.")

assert len(CLASSICAL_SOLUTIONS) > 0, "P4 must be bipartite -- sanity check failed"


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the classical solution set
# ---------------------------------------------------------------------------
# Qubit q_i holds vertex i's color bit. Qiskit's bitstring order from
# measurement is q_{N-1} ... q_0 (little-endian in the string), so we build
# the oracle per-solution using explicit qubit indices to avoid ambiguity.

def bits_to_oracle_flip(qc, bits, qubits):
    """Phase-flip the basis state matching `bits` (one bit per vertex/qubit)."""
    # Rotate the target state to |11...1> on the marked pattern using X gates
    # on qubits that should be 0, apply multi-controlled Z, then undo the X's.
    flip_qubits = [qubits[i] for i, b in enumerate(bits) if b == 0]
    for q in flip_qubits:
        qc.x(q)
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for q in flip_qubits:
        qc.x(q)


def build_oracle(n, solutions):
    qc = QuantumCircuit(n, name="oracle")
    qubits = list(range(n))
    for sol in solutions:
        bits_to_oracle_flip(qc, sol, qubits)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qubits = list(range(n))
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)
    return qc


import math

M = len(CLASSICAL_SOLUTIONS)
N_total = 2 ** N
# Optimal number of Grover iterations for M solutions out of N_total.
theta = math.asin(math.sqrt(M / N_total))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover search: {N} qubits, {M} marked states out of {N_total}, "
      f"using {iterations} Grover iteration(s).")

oracle = build_oracle(N, CLASSICAL_SOLUTIONS)
diffuser = build_diffuser(N)

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N))
    qc.append(diffuser.to_gate(), range(N))
qc.measure(range(N), range(N))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

from qiskit import transpile

backend = AerSimulator()
shots = 4096
qc = transpile(qc, backend)
job = backend.run(qc, shots=shots)
counts = job.result().get_counts()

# Qiskit classical-register bitstrings are ordered c[n-1] ... c[0]; qubit i
# was q_i, so bit string index (from the right, position i) is vertex i.
def bitstring_to_vertex_tuple(bs):
    # bs is e.g. "0110"; bs[-1-i] is qubit i's measured value.
    return tuple(int(bs[-1 - i]) for i in range(N))

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_bitstring, top_count = sorted_counts[0]
top_vertex_tuple = bitstring_to_vertex_tuple(top_bitstring)

print("Top measured outcomes (bitstring: count):")
for bs, c in sorted_counts[:5]:
    print(f"  {bs}: {c}  -> vertex-coloring {bitstring_to_vertex_tuple(bs)}")

solution_prob_mass = sum(
    c for bs, c in counts.items() if bitstring_to_vertex_tuple(bs) in CLASSICAL_SOLUTIONS
) / shots
baseline = M / N_total  # probability a uniformly random coloring is a solution

quantum_found_proper_coloring = (
    top_vertex_tuple in CLASSICAL_SOLUTIONS
    and solution_prob_mass > 3 * baseline  # clearly amplified above uniform baseline
)


# ---------------------------------------------------------------------------
# 4. Verify and report
# ---------------------------------------------------------------------------

verified = quantum_found_proper_coloring and (len(CLASSICAL_SOLUTIONS) > 0)

print()
print(f"Classical: P4 has a proper 2-coloring: {len(CLASSICAL_SOLUTIONS) > 0}")
print(f"Quantum (Grover) top outcome {top_vertex_tuple} with probability "
      f"{top_count/shots:.3f}; total probability mass on proper 2-colorings: "
      f"{solution_prob_mass:.3f} (uniform baseline would be {baseline:.3f}); "
      f"top outcome is itself a proper 2-coloring: {quantum_found_proper_coloring}")

if verified:
    print("PASS")
else:
    print("FAIL")
