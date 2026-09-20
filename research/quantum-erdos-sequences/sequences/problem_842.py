"""
Quantum-testable instance for Erdos problem #842.

Erdos problem #842 (data/problems.yaml in the erdosproblems repository,
manman4/erdosproblems, block `- number: "842"`) is tagged
["graph theory", "chromatic number"] and its `oeis` field is `["N/A"]` --
the problem carries no OEIS sequence id at all. This is reported honestly
below (ran_ok/verified_against_classical do not depend on an OEIS lookup
that cannot happen): there is no sequence to test membership/terms of, so
this script instead builds a genuine, small, finite, computable instance of
the problem's own subject matter -- proper graph colorings and the
chromatic number -- and tests it with a real Grover search circuit.

Classical property being tested
--------------------------------
Graph: the path graph P4 on vertices {0,1,2,3} with edges
    E = {(0,1), (1,2), (2,3)}
Search space: all 2^4 = 16 assignments of one of 2 colors (bit 0/1) to
each of the 4 vertices, encoded as a 4-qubit computational basis state
|v0 v1 v2 v3>.

Property: an assignment is a PROPER 2-COLORING of P4 iff every edge
connects two differently-colored vertices, i.e. v_i != v_j for every
(i,j) in E.

The classical answer (computed in this script by brute force over all 16
assignments, from first principles, before any quantum code runs) is that
there are exactly 2 proper 2-colorings of P4: the two alternating
colorings 0101 and 1010 (bit order v0 v1 v2 v3). P4 is bipartite/a tree,
so its chromatic number is 2, consistent with having proper 2-colorings
at all (a graph with an odd cycle, e.g. C5, would have zero -- this
script's oracle construction generalizes to that case too, just with
different EDGES/N).

Quantum circuit
----------------
A standard Grover search: a phase oracle marks (with a -1 phase) exactly
the computational basis states that are proper 2-colorings of P4, built
by computing each edge's XOR into an ancilla qubit, multi-controlling a
phase flip on the AND of all edge ancillas being 1, then uncomputing the
ancillas. This is run on the ideal AerSimulator for the optimal number of
Grover iterations for N=16, M=2 solutions, and the two most frequently
measured 4-bit strings are compared against the classical solution set.

PASS iff the two states measured with the highest counts are exactly the
brute-force computed solution set {0101, 1010} (regardless of which of
the two ties for first/second place, since both are true solutions and
should each get roughly half the amplified probability).
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------

N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3)]  # path graph P4


def is_proper_2_coloring(bits):
    """bits: tuple of 0/1, length N_VERTICES, indexed by vertex."""
    return all(bits[i] != bits[j] for (i, j) in EDGES)


def bits_to_bitstring(bits):
    # Qiskit measurement bitstrings list the highest-index qubit first,
    # so build the string to match qiskit's convention (qN-1 ... q0).
    return "".join(str(b) for b in reversed(bits))


classical_solutions = []
for assignment in itertools.product([0, 1], repeat=N_VERTICES):
    if is_proper_2_coloring(assignment):
        classical_solutions.append(bits_to_bitstring(assignment))

classical_solutions = sorted(classical_solutions)
N = 2 ** N_VERTICES
M = len(classical_solutions)

print(f"Classical brute force over all {N} colorings of P4:")
print(f"  proper 2-colorings found: {classical_solutions}  (M={M})")
assert classical_solutions == ["0101", "1010"], (
    "unexpected classical result -- P4 chromatic number logic is broken"
)

# ---------------------------------------------------------------------
# 2. Grover oracle: phase-flip states that are proper 2-colorings.
# ---------------------------------------------------------------------

n_v = N_VERTICES          # vertex/color qubits: q0..q3
n_e = len(EDGES)          # ancilla qubits, one per edge: q4..q6
flag = n_v + n_e          # flag qubit: q7
n_qubits = n_v + n_e + 1


def build_oracle():
    qc = QuantumCircuit(n_qubits, name="oracle")
    edge_qubits = list(range(n_v, n_v + n_e))

    # Compute: ancilla_k = v_i XOR v_j for edge k=(i,j)
    for k, (i, j) in enumerate(EDGES):
        a = edge_qubits[k]
        qc.cx(i, a)
        qc.cx(j, a)

    # Flip the phase of the flag (prepared in |-> outside) controlled on
    # ALL edge ancillas being 1 (i.e. every edge is properly colored).
    qc.mcx(edge_qubits, flag)

    # Uncompute the ancillas so the oracle is its own inverse otherwise.
    for k, (i, j) in reversed(list(enumerate(EDGES))):
        a = edge_qubits[k]
        qc.cx(j, a)
        qc.cx(i, a)

    return qc


def build_diffuser():
    qc = QuantumCircuit(n_v, name="diffuser")
    qc.h(range(n_v))
    qc.x(range(n_v))
    qc.h(n_v - 1)
    qc.mcx(list(range(n_v - 1)), n_v - 1)
    qc.h(n_v - 1)
    qc.x(range(n_v))
    qc.h(range(n_v))
    return qc


iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
print(f"Grover iterations used: {iterations} (optimal for N={N}, M={M})")

qc = QuantumCircuit(n_qubits, n_v)

# Uniform superposition over the 4 color qubits.
qc.h(range(n_v))

# Flag qubit prepared in |-> for phase-kickback oracle.
qc.x(flag)
qc.h(flag)

oracle = build_oracle()
diffuser = build_diffuser()

for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(n_qubits))
    qc.append(diffuser.to_instruction(), range(n_v))

qc.measure(range(n_v), range(n_v))
qc = qc.decompose()

# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
print("Measurement counts (top 6):")
for bitstring, c in sorted_counts[:6]:
    print(f"  {bitstring}: {c}")

top_two = sorted({bs for bs, _ in sorted_counts[:2]})

# ---------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------

quantum_solutions = sorted(top_two)
verified = quantum_solutions == classical_solutions

print()
print(f"Classical proper 2-colorings of P4: {classical_solutions}")
print(f"Quantum (Grover) top-{len(top_two)} measured states: {quantum_solutions}")

if verified:
    print("PASS")
else:
    print("FAIL")
