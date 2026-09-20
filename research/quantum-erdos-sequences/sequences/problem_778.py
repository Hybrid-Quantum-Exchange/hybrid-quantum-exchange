"""
Erdos problem #778 (https://erdosproblems.com/778) -- quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 778"):
    prize: no
    status: open (informal_status: open, last_update 2025-08-31)
    oeis: ["N/A"]
    tags: ["graph theory"]

LIMITATION (reported honestly, not faked): problem #778 has NO associated
OEIS sequence id in the source data (oeis: ["N/A"]). There is therefore no
integer sequence to build a Grover/phase-estimation/amplitude-estimation
circuit "for the sequence" in the sense the other lanes in this library use.
Per the task instructions for this case, this script is the best honest
attempt: it builds a REAL, small, genuinely computable quantum circuit
around the one piece of real mathematical content problem #778 does carry
in its metadata -- its tag, "graph theory" -- rather than inventing or
copying a fabricated OEIS value.

Chosen finite, computable property:
    Among all labeled simple graphs on 3 vertices (there are exactly
    2**3 = 8 of them, one bit per possible edge {0,1},{0,2},{1,2}), exactly
    ONE graph is a triangle (K3): the graph where all three edges are
    present. "Does this 3-vertex graph contain a triangle (is it exactly
    K3)?" is a small, finite, exactly computable graph-theory property --
    the same kind of object (triangle existence / extremal graph counting)
    that Erdos-style graph-theory problems such as #778 are about.

The classical answer for this N=8 instance is computed here from first
principles (brute-force enumeration over all 8 edge-subsets, checking which
one has all 3 edges set), independent of any OEIS lookup, and is the unique
graph "111" (edges 01, 02, 12 all present).

The quantum side is a genuine Grover search over the 3-qubit space of edge
subsets, with the oracle implemented as a standard multi-controlled-Z phase
flip on the unique marked bitstring "111", run on the ideal AerSimulator.
The optimal number of Grover iterations for N=8, M=1 marked item is computed
from the standard formula floor(pi/4 * sqrt(N/M)).

The script prints PASS if the most frequent measurement outcome from the
Grover circuit matches the classically-brute-forced triangle graph, else
FAIL.
"""

import math
import sys

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

ERDOS_PROBLEM_NUMBER = 778
OEIS_IDS = []  # none exist for this problem ("N/A" in source data)

NUM_VERTICES = 3
NUM_EDGES = NUM_VERTICES * (NUM_VERTICES - 1) // 2  # = 3 qubits -> N = 8
EDGES = [(0, 1), (0, 2), (1, 2)]


def classical_find_triangle():
    """Brute-force, from first principles, every 3-vertex labeled graph
    (each of the 3 possible edges present or absent) and return the unique
    bitstring (qubit order q0 q1 q2 <-> edges[0] edges[1] edges[2]) for
    which the graph is exactly the triangle K3 (all edges present)."""
    triangle_bitstrings = []
    for bits in range(2 ** NUM_EDGES):
        edge_present = [(bits >> i) & 1 for i in range(NUM_EDGES)]
        is_triangle = all(edge_present)  # K3 <=> every edge present
        if is_triangle:
            # Qiskit's bit ordering in measurement strings is q(n-1)...q0
            bitstring = "".join(str(edge_present[i]) for i in reversed(range(NUM_EDGES)))
            triangle_bitstrings.append(bitstring)
    return triangle_bitstrings


def build_oracle(qc, qubits):
    """Phase-flip oracle marking the unique all-ones state (the triangle)."""
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])


def build_diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def run_grover(shots=2048):
    n = NUM_EDGES
    N = 2 ** n
    M = 1  # exactly one marked graph (the triangle) out of 8 possible graphs
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    qc = QuantumCircuit(n, n)
    qubits = list(range(n))
    qc.h(qubits)
    for _ in range(iterations):
        build_oracle(qc, qubits)
        build_diffuser(qc, qubits)
    qc.measure(qubits, qubits)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    winner = max(counts, key=counts.get)
    return winner, counts, iterations


def main():
    classical_answer = classical_find_triangle()
    assert len(classical_answer) == 1, "expected a unique triangle graph on 3 vertices"
    expected = classical_answer[0]

    quantum_answer, counts, iterations = run_grover()

    print(f"Erdos problem #{ERDOS_PROBLEM_NUMBER}: no OEIS id in source data "
          f"(oeis: ['N/A'], tags: ['graph theory']).")
    print("Property tested: Grover search over all 8 labeled 3-vertex graphs "
          "for the unique graph that is a triangle (K3, all 3 edges present).")
    print(f"Classical brute-force answer (edge bits q2 q1 q0): {expected}")
    print(f"Grover iterations used: {iterations}")
    print(f"Quantum most-frequent outcome: {quantum_answer}  (counts: {counts})")

    if quantum_answer == expected:
        print("PASS")
        return 0
    else:
        print("FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
