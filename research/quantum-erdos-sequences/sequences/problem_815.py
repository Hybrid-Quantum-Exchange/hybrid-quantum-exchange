"""
Erdos problem #815 -- quantum-testable lane.

Source metadata (data/problems.yaml, entry "number: '815'", read-only clone
of github.com/manman4/erdosproblems):

    number: "815"
    prize: "no"
    informal_status: disproved (last_update 2025-08-31)
    formal_status:   unformalized
    oeis: ["N/A"]
    tags: ["graph theory"]

LIMITATION (reported honestly, not papered over): problem #815 carries no
OEIS sequence id at all (oeis == ["N/A"]), and the metadata clone available
here gives no further statement text for the problem, only the fields above.
There is therefore no actual integer sequence from this problem to build a
membership/counting/search oracle out of -- any claim to test "the sequence
for problem 815" would be fabricated. Per the task's own fallback
instructions, this script is the best-honest-effort substitute: a genuine,
correctly verified quantum circuit exercising the one concrete keyword the
metadata *does* give ("graph theory"), on a small, fully specified, finite
combinatorial object, with its correct answer computed classically from
first principles in this same script. It is NOT a property of problem 815's
sequence (there isn't one available), and should not be reported as such.

Chosen finite computable property
----------------------------------
Let G range over all labeled graphs on the 4 vertices {0,1,2,3} (i.e. over
all 2**6 = 64 subsets of the 6 possible edges of K4). Fix the triangle
T = {edge(0,1), edge(0,2), edge(1,2)}.

Property tested: "G contains triangle T", i.e. all three of those edges are
present in G (the other 3 edges of K4 are unconstrained).

This is classically trivial to enumerate (64 graphs) and gives a clean,
known marked-set size (2**3 = 8 graphs satisfy it, since the other 3 edges
are free), which is exactly the setting Grover's algorithm is built for:
search an unstructured space of size N=64 for a set of M=8 marked items
using an oracle built directly from the property's own logical definition
(three controlled-phase conditions), not from a pre-computed answer key.

Circuit
-------
6 qubits encode the 6 possible edges of K4 (q0=edge01, q1=edge02, q2=edge12,
q3=edge03, q4=edge13, q5=edge23). One ancilla qubit, prepared in the |->
state, is used for phase kickback: a Toffoli(q0,q1,q2 -> ancilla) flips the
sign of exactly the computational-basis states where q0=q1=q2=1, i.e.
exactly the graphs containing triangle T -- this is the oracle, built
straight from the property's definition. The standard Grover diffuser
follows. With N=64, M=8, the optimal iteration count is
round(pi/4 * sqrt(N/M)) = 2, which the script computes rather than hard-codes.

Verification: the script (a) classically enumerates all 64 graphs and
computes the exact marked set and its size, (b) runs the Grover circuit on
the ideal AerSimulator, (c) checks that the measured-outcome probability
mass landing in the classically-computed marked set is boosted far above
the uniform baseline (8/64 = 12.5%) to a value consistent with Grover's
known amplification, and (d) checks the single most frequent measured
outcome is itself a classically verified triangle-containing graph.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_EDGE_QUBITS = 6
EDGES = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]  # qubit index -> edge
TRIANGLE_QUBITS = (0, 1, 2)  # edges (0,1),(0,2),(1,2) == triangle {0,1,2}


def classical_marked_set():
    """Enumerate all 2**6 edge-subsets of K4 and return those containing
    the triangle on vertices {0,1,2} (i.e. edges 0,1,2 all present)."""
    marked = []
    for bits in product([0, 1], repeat=N_EDGE_QUBITS):
        if all(bits[q] == 1 for q in TRIANGLE_QUBITS):
            # bit order here matches qiskit little-endian bitstrings later
            marked.append(bits)
    return marked


def bits_to_qiskit_bitstring(bits):
    # qiskit reports bitstrings as q(n-1)...q1 q0 (little-endian in string)
    return "".join(str(b) for b in reversed(bits))


def build_grover_circuit(iterations):
    n = N_EDGE_QUBITS
    qc = QuantumCircuit(n + 1, n)  # +1 ancilla for phase kickback

    # uniform superposition over the 6 edge qubits
    qc.h(range(n))

    # ancilla in |-> for phase-kickback oracle
    qc.x(n)
    qc.h(n)

    def oracle(qc):
        # flips phase of exactly the states with q0=q1=q2=1 (triangle present)
        qc.mcx(list(TRIANGLE_QUBITS), n)

    def diffuser(qc):
        qc.h(range(n))
        qc.x(range(n))
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        qc.x(range(n))
        qc.h(range(n))

    for _ in range(iterations):
        oracle(qc)
        diffuser(qc)

    qc.measure(range(n), range(n))
    return qc


def main():
    marked = classical_marked_set()
    N = 2 ** N_EDGE_QUBITS
    M = len(marked)
    assert M == 8, f"expected 8 triangle-containing graphs, got {M}"
    marked_bitstrings = {bits_to_qiskit_bitstring(b) for b in marked}

    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
    print(f"N={N} states, M={M} marked, using {iterations} Grover iteration(s)")

    qc = build_grover_circuit(iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    marked_shots = sum(c for bstr, c in counts.items() if bstr in marked_bitstrings)
    marked_prob = marked_shots / shots
    baseline_prob = M / N

    top_outcome = max(counts.items(), key=lambda kv: kv[1])[0]
    top_outcome_is_marked = top_outcome in marked_bitstrings

    # theoretical Grover success probability for this N, M, iteration count
    theta = math.asin(math.sqrt(M / N))
    theoretical_prob = math.sin((2 * iterations + 1) * theta) ** 2

    print(f"baseline (uniform) marked probability : {baseline_prob:.4f}")
    print(f"theoretical Grover marked probability  : {theoretical_prob:.4f}")
    print(f"measured marked probability ({shots} shots): {marked_prob:.4f}")
    print(f"most frequent outcome: {top_outcome} (marked={top_outcome_is_marked})")

    # PASS criteria: measured probability is close to theory and clearly
    # boosted over the uniform baseline, and the top outcome is genuinely
    # a triangle-containing graph per the classical check.
    boosted = marked_prob > baseline_prob * 2.5
    close_to_theory = abs(marked_prob - theoretical_prob) < 0.08
    verified = boosted and close_to_theory and top_outcome_is_marked

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
