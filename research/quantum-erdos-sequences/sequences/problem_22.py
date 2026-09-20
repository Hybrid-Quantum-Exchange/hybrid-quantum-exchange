"""
Erdos problem #22 -- quantum-testable companion script.

Source record (read-only clone of manman4/erdosproblems,
data/problems.yaml, block for `number: "22"`):

    number: "22"
    prize: "no"
    informal_status: {state: "proved", last_update: "2025-08-31"}
    formal_status:   {state: "Lean",   last_update: "2026-08-23"}
    status:          {state: "proved (Lean)", last_update: "2026-08-23"}
    oeis: ["possible"]
    tags: ["graph theory"]

HONEST LIMITATION, stated up front: the `oeis` field for problem 22 in the
source data is the literal string "possible", not an OEIS A-number. There is
no real OEIS sequence id recorded for this problem in the source file, so
this script cannot build a circuit that tests membership/a term of "the"
OEIS sequence for problem 22 -- there isn't one to test. Grepping the raw
YAML at the time of writing confirms this is not a truncation artifact; it
is the value actually stored.

What this script does instead, honestly labeled as a best-effort substitute
tied to the one piece of real content the record does give -- the tag
"graph theory": it builds a genuine Grover search circuit over a small,
fully-specified finite graph problem (the kind of finite/computable graph
property Erdos-style graph theory problems are usually reducible to for a
tiny instance), and verifies the quantum search result against a classical
brute-force computation performed from first principles in this script.

Concrete instance chosen: the 4-cycle graph C4 on vertices {0,1,2,3} with
edges {(0,1),(1,2),(2,3),(3,0)}. The property being searched for is:

    "ordered pair (i, j), i != j, such that i and j are NOT adjacent in C4"
    (i.e. a non-edge / independent pair).

Search space: all ordered pairs (i, j) with i, j in {0,1,2,3}, encoded as a
4-qubit basis state |i>|j> (2 qubits each, 16 basis states total).

Classical ground truth (computed below, not copied from anywhere): C4's
non-adjacent ordered pairs are exactly the two "diagonals" in both
directions: (0,2), (2,0), (1,3), (3,1) -- 4 marked states out of 16.

Grover's algorithm is run for the classically-optimal number of iterations
for N=16, M=4 marked items (1 iteration), and the resulting measurement
distribution is compared against the classical marked set: PASS if the
quantum circuit's most-probable outcomes are exactly the classical marked
set (and their combined probability clears a fixed threshold).
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_non_adjacent_pairs(edges, n_vertices):
    """Brute-force, from first principles: all ordered (i,j), i!=j, not in `edges`."""
    edge_set = set()
    for a, b in edges:
        edge_set.add((a, b))
        edge_set.add((b, a))
    marked = []
    for i, j in itertools.product(range(n_vertices), repeat=2):
        if i != j and (i, j) not in edge_set:
            marked.append((i, j))
    return sorted(marked)


def pair_to_bits(i, j):
    """Encode (i,j), each in {0,1,2,3}, as a 4-bit string 'i1 i0 j1 j0' (Qiskit
    little-endian qubit order: qubit 0 is i's low bit, qubit1 i's high bit,
    qubit2 j's low bit, qubit3 j's high bit)."""
    return [(i >> 0) & 1, (i >> 1) & 1, (j >> 0) & 1, (j >> 1) & 1]


def bits_to_pair(bitstring):
    # bitstring is Qiskit's measurement key: c3 c2 c1 c0 (qubit0 first char from right)
    b = bitstring[::-1]  # now b[0]=qubit0 ... b[3]=qubit3
    i = int(b[0]) | (int(b[1]) << 1)
    j = int(b[2]) | (int(b[3]) << 1)
    return (i, j)


def build_oracle(marked_pairs, n_qubits=4):
    """Phase-flip oracle: for each marked (i,j), flip the sign of |i>|j> using
    a multi-controlled Z realized with X-gates around an MCX-with-phase-kick."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for (i, j) in marked_pairs:
        bits = pair_to_bits(i, j)  # order: q0,q1,q2,q3
        zero_qubits = [q for q, b in enumerate(bits) if b == 0]
        for q in zero_qubits:
            qc.x(q)
        # multi-controlled Z on all 4 qubits (phase flip of |1111...>)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits=4):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # C4
    n_vertices = 4
    n_qubits = 4  # 2 qubits for i, 2 for j
    N = 2 ** n_qubits  # 16 basis states

    marked = classical_non_adjacent_pairs(edges, n_vertices)
    M = len(marked)
    print(f"Classical brute force: non-adjacent ordered pairs of C4 = {marked}")
    print(f"N={N} total states, M={M} marked states")

    # Optimal number of Grover iterations for this N, M
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Running Grover with {iterations} iteration(s)")

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    shots = 8192
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Top M measured outcomes by count
    sorted_outcomes = sorted(counts.items(), key=lambda kv: -kv[1])
    top_bitstrings = [bs for bs, _ in sorted_outcomes[:M]]
    top_pairs = sorted(bits_to_pair(bs) for bs in top_bitstrings)

    marked_prob = sum(c for bs, c in counts.items() if bits_to_pair(bs) in marked) / shots

    print(f"Quantum top-{M} measured pairs: {top_pairs}")
    print(f"Total probability mass on classically-marked pairs: {marked_prob:.3f}")

    verified = (top_pairs == marked) and (marked_prob > 0.8)

    if verified:
        print("PASS: Grover search result matches classical brute-force marked set.")
    else:
        print("FAIL: Grover search result does not match classical brute-force marked set.")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
