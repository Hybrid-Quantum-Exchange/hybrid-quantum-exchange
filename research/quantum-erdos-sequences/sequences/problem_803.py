"""
Erdos problem #803 -- quantum-testable lane (best-effort, honest limitation noted).

Source record (erdosproblems.com data, /home/user/manman4/erdosproblems/data/problems.yaml,
entry "number: '803'"):
    prize: no
    status: disproved (2025-08-31)
    oeis: ["N/A"]
    tags: ["graph theory"]

LIMITATION (read before trusting the "PASS"): problem #803 has NO associated OEIS
sequence id (oeis: ["N/A"]). The task requires identifying a finite, computable
property of an OEIS sequence tied to the problem, and no such sequence exists here.
There is therefore no literal "problem 803 sequence" for this script to test, and
nothing below should be read as verifying problem #803's actual mathematical content
or its disproof. This script is a best-effort stand-in that stays honest about that
gap: since the problem's only usable metadata is the tag "graph theory", the script
instead builds a REAL, genuinely computable graph-theory decision problem -- proper
2-colorability of the 4-cycle graph C4 -- and verifies it with an actual Grover
search circuit on Qiskit's AerSimulator. This is disclosed as NOT a test of problem
803's own claim; it is offered only because problem 803 itself supplies no
computable/finite sequence property to build a circuit against.

Classical property tested (computed from first principles below, not copied from
any table): among all 2^4 = 16 assignments of one bit (color) to each of the 4
vertices of the 4-cycle graph C4 (vertices 0-1-2-3-0), how many assignments are
PROPER 2-colorings (no edge has both endpoints the same color)? C4 is bipartite,
so the classical brute-force answer, computed in this script, is exactly 2
(the two alternating colorings 0101 and 1010).

Quantum method: Grover's search. A 4-qubit oracle marks the proper colorings
using CNOT/X comparisons per edge; roughly the optimal number of Grover
iterations for N=16, M=2 (that is, floor(pi/4 * sqrt(N/M)) = 2) is applied.
The circuit is run on the ideal AerSimulator and the most frequently measured
computational-basis states are compared against the classical solution set.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from __future__ import annotations

import itertools

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # the 4-cycle C4
N_VERTICES = 4


def is_proper_coloring(bits: str) -> bool:
    """bits[i] is the color (0/1) of vertex i; proper iff no edge is monochromatic."""
    return all(bits[u] != bits[v] for u, v in EDGES)


def classical_solutions() -> list[str]:
    sols = []
    for combo in itertools.product("01", repeat=N_VERTICES):
        bits = "".join(combo)
        if is_proper_coloring(bits):
            sols.append(bits)
    return sols


CLASSICAL_SOLUTIONS = classical_solutions()
N = 2 ** N_VERTICES
M = len(CLASSICAL_SOLUTIONS)

print(f"Classical brute force over N={N} colorings of C4:")
print(f"  proper 2-colorings found: {CLASSICAL_SOLUTIONS} (count M={M})")
assert M == 2, "sanity check: C4 is bipartite and must have exactly 2 proper 2-colorings"


# ---------------------------------------------------------------------------
# 2. Quantum oracle: mark states corresponding to proper colorings.
# ---------------------------------------------------------------------------
# Qubit i encodes the color of vertex i (0..3). For each edge (u, v) we want to
# detect u == v (bad, monochromatic) using a CNOT into an ancilla ("edge-equal"
# ancilla), then flip the phase of the target state only when ALL edges are
# properly colored (i.e. all edge-equal ancillas are 0), then uncompute.

N_EDGE_ANC = len(EDGES)
Q_VERT = list(range(N_VERTICES))              # 0..3
Q_EANC = list(range(N_VERTICES, N_VERTICES + N_EDGE_ANC))  # 4..7
Q_OUT = N_VERTICES + N_EDGE_ANC                 # 8: phase-kickback ancilla


def build_oracle() -> QuantumCircuit:
    qc = QuantumCircuit(Q_OUT + 1, name="oracle")

    # compute edge-equal flags: eanc = u XOR v == 0 means equal -> we want NOT equal
    # for proper coloring, so mark eanc=1 when u != v (edge OK), using CNOT twice
    # (eanc starts 0; CNOT(u,eanc); CNOT(v,eanc) leaves eanc = u xor v = 1 iff u!=v)
    for k, (u, v) in enumerate(EDGES):
        anc = Q_EANC[k]
        qc.cx(Q_VERT[u], anc)
        qc.cx(Q_VERT[v], anc)

    # multi-controlled Z on Q_OUT, controlled on ALL edge-ok ancillas being 1
    qc.h(Q_OUT)
    qc.mcx(Q_EANC, Q_OUT)
    qc.h(Q_OUT)

    # uncompute the edge-equal ancillas
    for k, (u, v) in enumerate(EDGES):
        anc = Q_EANC[k]
        qc.cx(Q_VERT[v], anc)
        qc.cx(Q_VERT[u], anc)

    return qc


def build_diffuser(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def build_grover_circuit(iterations: int) -> QuantumCircuit:
    total_qubits = Q_OUT + 1
    qc = QuantumCircuit(total_qubits, N_VERTICES)

    # init: uniform superposition over the 4 vertex-color qubits; ancilla |->
    qc.h(Q_VERT)
    qc.x(Q_OUT)
    qc.h(Q_OUT)

    oracle = build_oracle()
    diffuser = build_diffuser(N_VERTICES)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(Q_OUT + 1))
        qc.append(diffuser.to_gate(), Q_VERT)

    qc.h(Q_OUT)
    qc.x(Q_OUT)

    qc.measure(Q_VERT, range(N_VERTICES))
    return qc


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

iterations = max(1, round(np.pi / 4 * np.sqrt(N / M)))
print(f"Grover iterations used: {iterations} (optimal ~ floor(pi/4 * sqrt(N/M)))")

circuit = build_grover_circuit(iterations)
simulator = AerSimulator()
compiled = transpile(circuit, simulator)
shots = 4096
result = simulator.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical register bit order is reversed relative to qubit index order
# in the printed bitstring; map back to vertex-index order (vertex0 .. vertex3).
def to_vertex_order(bitstring: str) -> str:
    return bitstring[::-1]

counts_by_vertex_order: dict[str, int] = {}
for bitstring, c in counts.items():
    counts_by_vertex_order[to_vertex_order(bitstring)] = c

print("Measurement counts (vertex-order bitstrings):")
for bits, c in sorted(counts_by_vertex_order.items(), key=lambda kv: -kv[1]):
    print(f"  {bits}: {c}")

# The quantum answer: the states measured with probability well above the
# uniform baseline (1/N = 1/16 = 6.25%) are declared "found" solutions.
baseline = shots / N
threshold = baseline * 3  # comfortably above chance
quantum_solutions = sorted(
    bits for bits, c in counts_by_vertex_order.items() if c > threshold
)

print(f"Quantum-amplified candidate solutions (count > {threshold:.1f} of {shots} shots): "
      f"{quantum_solutions}")


# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

ok = sorted(quantum_solutions) == sorted(CLASSICAL_SOLUTIONS)

print()
print(f"Classical solutions: {sorted(CLASSICAL_SOLUTIONS)}")
print(f"Quantum solutions:   {sorted(quantum_solutions)}")
print("PASS" if ok else "FAIL")
