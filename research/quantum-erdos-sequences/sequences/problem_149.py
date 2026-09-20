"""
Erdos problem #149 -- quantum-testable companion script.

Source record: erdosproblems.com problem #149 (see the read-only clone at
/home/user/manman4/erdosproblems/data/problems.yaml, entry `number: "149"`):

    prize: no
    status: open (informal_status: open, formal_status: unformalized)
    oeis: ["N/A"]
    tags: ["graph theory"]

LIMITATION (must be stated honestly): problem #149's YAML record carries no
OEIS sequence id at all -- `oeis: ["N/A"]`. There is therefore no concrete
integer sequence to build a "sequence membership" style quantum test around,
and the problem statement itself is not represented in this data file beyond
its tag ("graph theory"). Faking an OEIS-derived property here would violate
the task's own instructions, so instead this script falls back to the
documented alternative: a genuine, small, finite, computable property drawn
from the problem's *tag* (graph theory) rather than from a specific OEIS
term. This is an honest substitute, not a claim that it verifies problem
#149's actual open conjecture.

Chosen property (finite, computable, and checked classically from first
principles in this script before any quantum step):

    Let G be the 4-cycle graph C4 on vertices {0,1,2,3} with edges
    (0,1), (1,2), (2,3), (3,0). A "coloring" assigns each vertex a bit in
    {0,1}. A coloring is PROPER (a valid 2-coloring / witness that G is
    bipartite) iff every edge joins two differently-colored vertices.

    Classical brute force over all 2^4 = 16 colorings finds exactly the
    proper 2-colorings of C4 (there are exactly 2: 0101 and 1010, the two
    alternating colorings -- C4 is bipartite, as every even cycle is).

Quantum step: Grover's algorithm searches the 4-bit coloring space for a
proper 2-coloring of C4, using an oracle built from reversible XOR gates
(one ancilla per edge, flagging "endpoints differ") combined with a
multi-controlled phase flip when all four edge flags are 1, uncomputed
after use. The diffuser is the standard Grover diffusion operator on 4
qubits. With N=16 states and M=2 marked states, the optimal number of
Grover iterations is floor(pi/4 * sqrt(N/M)) = 2.

PASS/FAIL: the script runs the Grover circuit on the ideal AerSimulator,
takes the most-probable measured bitstring, and checks (a) that it is one
of the two proper 2-colorings found classically, and (b) that Grover
concentrated probability mass on the marked subspace far above the
1/8 = 2/16 baseline of uniform random guessing.
"""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # the 4-cycle C4


def is_proper_coloring(bits):
    """bits: tuple of 4 ints in {0,1}, bits[v] = color of vertex v."""
    return all(bits[u] != bits[v] for (u, v) in EDGES)


classical_valid = [bits for bits in product([0, 1], repeat=4) if is_proper_coloring(bits)]
classical_valid_strs = sorted("".join(str(b) for b in reversed(bits)) for bits in classical_valid)
# reversed() because Qiskit's bit order in a measured bitstring is q3 q2 q1 q0

N = 16
M = len(classical_valid)
assert M == 2, f"expected exactly 2 proper 2-colorings of C4, found {M}"

print(f"Classical brute force over all {N} colorings of C4:")
print(f"  proper 2-colorings found: {classical_valid_strs}  (count={M})")

# ---------------------------------------------------------------------
# 2. Grover oracle: mark colorings where every edge's endpoints differ.
# ---------------------------------------------------------------------
# Register layout:
#   q[0..3]  : data qubits, one per vertex (0,1,2,3)
#   anc[0..3]: one ancilla per edge, computes XOR of its two endpoints
#   phase    : single ancilla used for phase-kickback marking

n_data = 4
n_anc = len(EDGES)
data = list(range(n_data))
anc = list(range(n_data, n_data + n_anc))
phase = n_data + n_anc
n_qubits = n_data + n_anc + 1


def build_oracle():
    qc = QuantumCircuit(n_qubits, name="oracle")
    # compute anc[i] = data[u] XOR data[v] for each edge
    for i, (u, v) in enumerate(EDGES):
        qc.cx(data[u], anc[i])
        qc.cx(data[v], anc[i])
    # phase kickback: flip phase of |...> when all anc bits == 1
    qc.mcx(anc, phase)
    # uncompute ancillas
    for i, (u, v) in enumerate(EDGES):
        qc.cx(data[v], anc[i])
        qc.cx(data[u], anc[i])
    return qc


def build_diffuser():
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(data)
    qc.x(data)
    qc.h(data[-1])
    qc.mcx(data[:-1], data[-1])
    qc.h(data[-1])
    qc.x(data)
    qc.h(data)
    return qc


def build_grover_circuit(iterations):
    qc = QuantumCircuit(n_qubits, n_data)
    # phase ancilla prepared in |-> for phase kickback
    qc.x(phase)
    qc.h(phase)
    # uniform superposition over data qubits
    qc.h(data)

    oracle = build_oracle()
    diffuser = build_diffuser()
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(data, list(range(n_data)))
    return qc


iterations = int(np.floor((np.pi / 4) * np.sqrt(N / M)))
print(f"Grover iterations used: {iterations}")

qc = build_grover_circuit(iterations)

# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# Aggregate probability mass landing on the classically-verified marked set.
marked_mass = sum(c for bitstring, c in counts.items() if bitstring in classical_valid_strs)
marked_fraction = marked_mass / shots

top_bitstring = max(counts, key=counts.get)
top_fraction = counts[top_bitstring] / shots

print(f"Measurement counts (top 5): {sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")
print(f"Most frequent measured bitstring: {top_bitstring} (fraction {top_fraction:.3f})")
print(f"Total probability mass on classically-verified marked states: {marked_fraction:.3f}")

# ---------------------------------------------------------------------
# 4. Verify quantum result against the classical answer.
# ---------------------------------------------------------------------

baseline = M / N  # uniform-random baseline probability of hitting a marked state
top_is_marked = top_bitstring in classical_valid_strs
amplified = marked_fraction > 3 * baseline  # Grover should amplify well above baseline

verified = top_is_marked and amplified

print(f"Uniform-random baseline probability of a marked state: {baseline:.3f}")
print(f"Top measured state is a classically-verified proper 2-coloring: {top_is_marked}")
print(f"Marked-state probability mass amplified over baseline (>3x): {amplified}")

if verified:
    print("PASS")
else:
    print("FAIL")
