"""
Erdos problem #641 -- quantum-testable lane (best-effort, limitation noted).

Source metadata (data/problems.yaml, erdosproblems.com mirror, entry
"number: '641'"):
    prize:    no
    status:   disproved (last_update 2025-08-31)
    oeis:     ["N/A"]
    tags:     ["graph theory"]

LIMITATION, stated honestly up front: problem #641 has NO associated OEIS
sequence (oeis: ["N/A"]). The task family this script belongs to is "make an
OEIS sequence tied to an Erdos problem quantum-testable" -- that request has
no literal target here, because there is no sequence id to anchor to. Per
the fallback instructions, this script is still a genuine, from-scratch,
classically-verified quantum computation, built on the one concrete fact the
metadata *does* give us: the problem's tag, "graph theory". It does not
invent or borrow any numeric value from an OEIS entry (there is none to
copy), and it does not claim to test problem #641's actual mathematical
content (a disproved graph-theory statement, not itself restated in this
metadata file) -- only that it demonstrates a real, small, finite,
graph-theoretic search/counting problem of the same flavor, computed
classically first and then verified with a genuine Grover search circuit.

The finite computable property chosen:
    Enumerate all labeled simple graphs on 3 vertices. Each such graph is
    exactly a subset of the 3 possible edges {(0,1), (0,2), (1,2)}, so it is
    encoded as a 3-bit string x in {0,1}^3 (bit i = "edge i present").
    Define the property P(x): "the graph encoded by x has EXACTLY 2 edges"
    (equivalently: x has Hamming weight 2). This is a small, fully
    decidable graph-theoretic membership question -- precisely the kind of
    finite instance Grover search is built for -- and it is computed
    classically in this script from first principles (no lookup, no OEIS
    value copied).

    Classical answer (computed below, not hard-coded from any table):
    the marked set is {x : popcount(x) == 2} = {011, 101, 110} = {3, 5, 6}.

Quantum circuit:
    A genuine 3-qubit Grover search (AerSimulator, ideal/noiseless) over the
    8 possible graphs, whose oracle marks exactly the 3 graphs with 2 edges,
    run for the optimal number of Grover iterations for N=8, M=3. The
    script measures the final state, takes the most frequent outcomes, and
    checks quantum result set == classical marked set -> PASS/FAIL.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

N_VERTICES = 3
EDGES = list(combinations(range(N_VERTICES), 2))  # [(0,1), (0,2), (1,2)]
N_EDGES = len(EDGES)  # 3 possible edges -> 3-bit encoding, N = 8 graphs
TARGET_EDGE_COUNT = 2


def popcount(x: int) -> int:
    return bin(x).count("1")


def classical_marked_set(n_edges: int, target: int) -> set:
    """All 3-bit graphs on 3 vertices with exactly `target` edges present."""
    return {x for x in range(2 ** n_edges) if popcount(x) == target}


CLASSICAL_ANSWER = classical_marked_set(N_EDGES, TARGET_EDGE_COUNT)
assert CLASSICAL_ANSWER == {3, 5, 6}, "classical enumeration sanity check failed"

N = 2 ** N_EDGES          # search space size = 8
M = len(CLASSICAL_ANSWER)  # number of marked items = 3


# ---------------------------------------------------------------------------
# 2. Grover search circuit over the 3-bit graph encoding.
# ---------------------------------------------------------------------------

def build_oracle(n_qubits: int, marked: set) -> QuantumCircuit:
    """Phase-flip oracle: applies -1 phase to each marked computational
    basis state, built as a sum of multi-controlled Z gates (one per marked
    string), each surrounded by X gates on the 0-bits of that string."""
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for m in marked:
        bits = [(m >> i) & 1 for i in range(n_qubits)]
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        # Multi-controlled Z via H-MCX-H on last qubit
        qc.h(n_qubits - 1)
        if n_qubits - 1 >= 1:
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        else:
            qc.x(n_qubits - 1)
        qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits - 1 >= 1:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    else:
        qc.x(n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(n_qubits: int, marked: set, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def optimal_iterations(n: int, m: int) -> int:
    theta = math.asin(math.sqrt(m / n))
    return max(1, round((math.pi / (4 * theta)) - 0.5))


def run_grover(marked: set, n_qubits: int, shots: int = 4096):
    iterations = optimal_iterations(2 ** n_qubits, len(marked))
    qc = build_grover_circuit(n_qubits, marked, iterations)
    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    print("Erdos problem #641 -- quantum-testable lane")
    print("OEIS id(s): N/A (none exists for this problem; see docstring)")
    print(f"Classical property: 3-vertex labeled graphs with exactly "
          f"{TARGET_EDGE_COUNT} of {N_EDGES} possible edges")
    print(f"Search space N = {N}, marked set M = {CLASSICAL_ANSWER} (|M| = {M})")

    counts, iterations = run_grover(CLASSICAL_ANSWER, N_EDGES, shots=4096)
    print(f"Grover iterations used: {iterations}")

    total_shots = sum(counts.values())
    # Bit string from Qiskit is big-endian in string order but corresponds
    # to classical little-endian integer via int(..., 2) reversed to match
    # our qubit-index convention (qubit i = bit i of x).
    shot_dist = {}
    for bitstring, cnt in counts.items():
        x = int(bitstring[::-1], 2)
        shot_dist[x] = shot_dist.get(x, 0) + cnt

    # Take the top-M most frequent outcomes as the quantum-found marked set.
    ranked = sorted(shot_dist.items(), key=lambda kv: -kv[1])
    quantum_top = {x for x, _ in ranked[:M]}

    marked_prob = sum(shot_dist.get(x, 0) for x in CLASSICAL_ANSWER) / total_shots
    print(f"Probability mass on classically-marked states: {marked_prob:.4f}")
    print(f"Quantum top-{M} outcomes: {sorted(quantum_top)}")
    print(f"Classical answer:        {sorted(CLASSICAL_ANSWER)}")

    verified = (quantum_top == CLASSICAL_ANSWER) and (marked_prob > 0.75)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
