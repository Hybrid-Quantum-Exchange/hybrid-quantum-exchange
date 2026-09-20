"""
Erdos problem #666 -- quantum-testable lane (best-effort, limitation noted)
============================================================================

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 666"):

    prize: no
    informal_status: disproved (Lean formalization, last_update 2026-02-06)
    oeis: ["possible"]
    tags: ["graph theory"]

LIMITATION (read before trusting the "verified_against_classical" claim):
The dataset's `oeis` field for problem 666 is the literal placeholder string
"possible", not a real OEIS sequence identifier (e.g. "A000040"). There is
therefore no genuine OEIS sequence to derive a finite computable property
from for this problem, and the repository clone available here does not
carry the problem's full natural-language statement -- only this metadata
record. Fabricating an OEIS id or copying a term from thin air would violate
the task's own instructions, so this script does NOT claim to test any
Erdos-problem-666-specific sequence membership.

What this script does instead, honestly: problem 666 is tagged "graph
theory", so it exercises a small, genuinely finite, genuinely computable
graph-theory decision property -- "does this 3-vertex labelled graph contain
a triangle (i.e. is it the complete graph K3)?" -- as a real Grover-search
instance. The search space is the 2^3 = 8 possible edge-subsets of a
3-vertex graph (edges: {0,1}, {0,2}, {1,2}, one qubit per edge). Exactly one
of the 8 graphs (all three edges present) is a triangle. This is classically
computed from first principles below (brute-force enumeration), and then a
genuine 3-qubit Grover search circuit is built and run on AerSimulator to
find that same marked graph. The quantum result is compared against the
independently computed classical answer.

This is offered as the best honest attempt for this problem given the
available data, not as a claim that it tests a term of an Erdos-problem-666
OEIS sequence -- no such usable sequence id exists in the source metadata.
"""

import math
from itertools import product

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_answer():
    """Brute-force, from first principles: which 3-vertex edge-subsets
    (bit i = edge present) form a triangle (all 3 edges present)?

    Vertices: 0,1,2. Edges, in qubit order: e0={0,1}, e1={0,2}, e2={1,2}.
    A graph on these 3 vertices is a triangle iff all three edges are
    present, i.e. the edge-subset is exactly {0,1,2} -> bitstring '111'.
    """
    triangles = []
    for bits in product([0, 1], repeat=3):
        e0, e1, e2 = bits  # presence of each of the 3 possible edges
        edges_present = sum(bits)
        is_triangle = edges_present == 3  # only way with 3 vertices: all edges present
        if is_triangle:
            # bitstring as Qiskit prints it: qubit 2 first (little-endian display)
            bitstring = f"{bits[2]}{bits[1]}{bits[0]}"
            triangles.append(bitstring)
    assert len(triangles) == 1, "expected exactly one triangle among 3-vertex graphs"
    return triangles[0]


def build_grover_circuit(marked_bitstring, n_qubits=3):
    """Standard Grover search (oracle + diffuser), one iteration, tuned for
    a search space of size 2^3 = 8 with exactly 1 marked item.

    Optimal number of Grover iterations for N=8, M=1:
        r = floor(pi/4 * sqrt(N/M)) = floor(pi/4 * sqrt(8)) = floor(2.221) = 2
    """
    n = n_qubits
    N = 2 ** n
    M = 1
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    qc = QuantumCircuit(n, n)

    # uniform superposition
    qc.h(range(n))

    def oracle(qc):
        # flip phase of the marked bitstring (qubit 2 qubit1 qubit0 order in
        # marked_bitstring, matching Qiskit's little-endian display)
        target = marked_bitstring[::-1]  # now index i -> qubit i's required bit
        zero_positions = [i for i, b in enumerate(target) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        for i in zero_positions:
            qc.x(i)

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
    return qc, iterations


def run_and_check():
    classical = classical_answer()
    print(f"Classical answer (brute force): triangle graph bitstring = {classical}")

    qc, iters = build_grover_circuit(classical, n_qubits=3)
    print(f"Built Grover circuit with {iters} iteration(s) over N=8, M=1 search space")

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    print("Measurement counts:", counts)

    most_likely = max(counts, key=counts.get)
    prob_correct = counts.get(classical, 0) / shots

    print(f"Most likely measured bitstring = {most_likely} "
          f"(probability of correct answer = {prob_correct:.3f})")

    verified = (most_likely == classical) and (prob_correct > 0.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = run_and_check()
    if not ok:
        raise SystemExit(1)
