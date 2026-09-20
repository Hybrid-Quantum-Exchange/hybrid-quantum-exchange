"""
Erdos problem #751 (from erdosproblems.com / manman4/erdosproblems data).

Metadata found for problem 751 in data/problems.yaml:
    tags: ["graph theory", "chromatic number"]
    oeis: ["N/A"]
    status: "disproved (Lean)"

LIMITATION: Problem 751 has no associated OEIS sequence id ("N/A"), so there
is no OEIS term-membership or term-defining property available to test here.
Its tags do point at genuine, finite, computable mathematical content though:
graph chromatic number / proper vertex colorings. In place of an OEIS-derived
property, this script tests the closest honest finite/computable property
suggested by the tags themselves:

    CLASSICAL PROPERTY TESTED:
    For the 4-cycle graph C4 (vertices 0,1,2,3; edges (0,1),(1,2),(2,3),(3,0)),
    is C4 properly 2-colorable, and if so what are its valid 2-colorings
    (assignments of {0,1} to each vertex such that no edge has both endpoints
    the same color)?

    This is exactly a chromatic-number question (chi(C4) <= 2, i.e. C4 is
    bipartite) reduced to a small finite search: 2^4 = 16 candidate colorings
    of 4 vertices, searched with 4 qubits.

The classical answer is computed here from first principles by brute-force
enumeration of all 16 colorings, before the quantum circuit is built. C4 is
bipartite (chi(C4) = 2), so there are exactly 2 valid proper 2-colorings:
(0,1,0,1) and (1,0,1,0) (as 4-bit strings v0 v1 v2 v3).

QUANTUM APPROACH: Grover search over the 4-qubit space of colorings, with an
oracle built from an XOR/edge-inequality check (a direct arithmetic/oracle
circuit rather than a black box), amplifying exactly the valid colorings.
One Grover iteration (optimal for 2 marked out of 16 states) is run on the
ideal AerSimulator, and the script checks that the two most probable
measured bitstrings are exactly the two valid colorings computed classically.

This is a genuine, if modest, quantum circuit computing a real graph-theory
property connected to problem 751's own tags (chromatic number / graph
coloring) -- it is not a literal OEIS value, because problem 751 has no
OEIS id to draw one from.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (brute force), computed from first principles.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # C4
N = 4  # number of vertices / qubits


def is_valid_coloring(bits):
    """bits: tuple of 0/1 of length N, bits[i] = color of vertex i."""
    return all(bits[u] != bits[v] for (u, v) in EDGES)


def classical_valid_colorings():
    valid = []
    for bits in itertools.product([0, 1], repeat=N):
        if is_valid_coloring(bits):
            valid.append(bits)
    return valid


CLASSICAL_VALID = classical_valid_colorings()
# Bitstrings as Qiskit prints them: qubit 0 is the rightmost character.
CLASSICAL_VALID_STRINGS = {
    "".join(str(b) for b in reversed(bits)) for bits in CLASSICAL_VALID
}

print("Classical brute-force result:")
print(f"  C4 is bipartite (chi(C4) = 2): {len(CLASSICAL_VALID) > 0}")
print(f"  Valid 2-colorings (v0 v1 v2 v3): {CLASSICAL_VALID}")
print(f"  As bitstrings (qiskit order, q3 q2 q1 q0): {sorted(CLASSICAL_VALID_STRINGS)}")

assert len(CLASSICAL_VALID) == 2, "Expected exactly 2 valid 2-colorings of C4"


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search for valid colorings.
# ---------------------------------------------------------------------------
#
# Oracle: for each edge (u, v), compute XOR(qu, qv) into an ancilla via CNOTs
# (this is 1 iff the edge is properly colored, i.e. endpoints differ).
# A coloring is valid iff ALL edge-ancillas are 1, i.e. AND of 4 edge bits.
# We phase-flip the state when all edge conditions hold, using a
# multi-controlled Z on the 4 edge-ancilla qubits, and uncompute the
# ancillas afterward (standard oracle-with-ancilla-then-uncompute pattern).

def build_oracle():
    qc = QuantumCircuit(N + len(EDGES), name="oracle")
    color = list(range(N))              # qubits 0..3: vertex colors
    anc = list(range(N, N + len(EDGES)))  # qubits 4..7: per-edge "differ" bits

    # compute anc[i] = color[u] XOR color[v] for each edge
    for i, (u, v) in enumerate(EDGES):
        qc.cx(color[u], anc[i])
        qc.cx(color[v], anc[i])

    # phase flip iff all 4 ancillas are 1 (multi-controlled Z via H-MCX-H)
    qc.h(anc[-1])
    qc.mcx(anc[:-1], anc[-1])
    qc.h(anc[-1])

    # uncompute ancillas
    for i, (u, v) in enumerate(EDGES):
        qc.cx(color[v], anc[i])
        qc.cx(color[u], anc[i])

    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def build_grover_circuit():
    n_total = N + len(EDGES)
    qc = QuantumCircuit(n_total, N)

    # uniform superposition over the 4 color qubits
    qc.h(range(N))

    oracle = build_oracle()
    diffuser = build_diffuser(N)

    # Optimal number of Grover iterations for M=2 marked out of N_states=16:
    # theta = asin(sqrt(M/N_states)); iterations ~ round(pi/(4*theta) - 0.5)
    num_states = 2 ** N
    marked = len(CLASSICAL_VALID)
    theta = math.asin(math.sqrt(marked / num_states))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_total))
        qc.append(diffuser.to_instruction(), range(N))

    qc.measure(range(N), range(N))
    return qc, iterations


qc, iterations = build_grover_circuit()
print(f"\nGrover circuit built with {iterations} iteration(s), "
      f"{qc.num_qubits} qubits total.")

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

print(f"\nMeasurement counts (top results): "
      f"{sorted(counts.items(), key=lambda kv: -kv[1])[:6]}")

# ---------------------------------------------------------------------------
# 3. Verify: the two most probable outcomes should be exactly the two
#    classically valid colorings.
# ---------------------------------------------------------------------------

top2 = sorted(counts.items(), key=lambda kv: -kv[1])[:2]
top2_strings = {bitstring for bitstring, _ in top2}

top2_probability_mass = sum(c for _, c in top2) / shots

print(f"\nTop-2 measured bitstrings: {sorted(top2_strings)}")
print(f"Expected (classical) valid bitstrings: {sorted(CLASSICAL_VALID_STRINGS)}")
print(f"Probability mass on top-2 outcomes: {top2_probability_mass:.3f}")

verified = (
    top2_strings == CLASSICAL_VALID_STRINGS
    and top2_probability_mass > 0.7
)

if verified:
    print("\nPASS")
else:
    print("\nFAIL")
