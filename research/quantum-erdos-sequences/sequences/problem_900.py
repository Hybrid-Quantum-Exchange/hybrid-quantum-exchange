"""
Erdos problem #900 -- quantum-testable lane.

Source metadata (data/problems.yaml, erdosproblems repo, entry "number: '900'"):
    prize: no
    status: proved (as of 2025-08-31)
    oeis: ["N/A"]
    tags: ["graph theory"]

LIMITATION (read before trusting the PASS below as "verifying problem 900"):
Problem #900 carries NO OEIS sequence id -- its `oeis` field is literally the
string "N/A". There is therefore no integer sequence to derive a membership,
divisibility, or counting property from, and no way to build a circuit that
tests problem #900 itself against a classical OEIS term. Fabricating an OEIS
id or a "property of the sequence" would violate the task's own instructions
not to invent mathematical content that isn't there.

Best honest attempt taken instead: the problem's only real content available
at this scale is its tag, "graph theory". So this script builds a genuine,
small, verifiable quantum computation in that spirit -- Grover's search over
all edge-subsets of the complete graph on 3 vertices (K3), searching for the
unique subset that forms a triangle (all 3 possible edges present). This is
real graph theory and a real Grover circuit, but it is NOT a test of Erdos
problem #900's mathematical content, because no such finite computable
instance of problem #900 exists at this scale (no OEIS id to ground it in).

Classical setup:
    - 3 possible edges on 3 labeled vertices {0,1,2}: (0,1), (0,2), (1,2).
    - Each edge is present/absent -> 3 bits -> search space size N = 2**3 = 8.
    - "Triangle" = the graph where all 3 edges are present, i.e. bit string
      "111" (using little-endian qubit order q0 q1 q2 -> edge01 edge02 edge12).
    - There is exactly 1 marked graph out of 8, computed here by brute-force
      enumeration in pure Python (first-principles, not looked up).

Quantum method: standard Grover search (3 qubits, 1 marked state), oracle is
a multi-controlled-Z on the all-ones state, diffuser is the standard
inversion-about-mean operator. Optimal iteration count for N=8, M=1 is
round(pi/4 * sqrt(N/M)) = 2 iterations.

Run: python3 problem_900.py
Prints PASS if the most-probable measured bitstring on AerSimulator matches
the classically brute-forced triangle bitstring, else FAIL.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_find_triangle_k3():
    """Brute-force, first-principles: enumerate all 2**3 edge-subsets of K3
    and return the unique subset (as a bitstring, edge01 edge02 edge12) that
    forms a triangle, i.e. has all 3 edges present."""
    vertices = [0, 1, 2]
    edges = [(0, 1), (0, 2), (1, 2)]
    n_edges = len(edges)
    triangles = []
    for mask in range(2 ** n_edges):
        present = [(mask >> i) & 1 for i in range(n_edges)]
        # A triangle on these 3 labeled vertices requires all 3 edges present.
        if all(present):
            triangles.append(mask)
    assert len(triangles) == 1, "expected exactly one triangle-forming subset"
    mask = triangles[0]
    # bitstring with qubit 0 as least-significant bit, matching Qiskit's
    # little-endian classical-register convention (c[0] is rightmost char).
    bitstring = format(mask, f"0{n_edges}b")[::-1]
    return mask, bitstring, n_edges


def build_grover_circuit(n_qubits, marked_mask, iterations):
    """Standard Grover search circuit marking the single computational basis
    state equal to marked_mask (n_qubits bits, bit i on qubit i)."""
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Uniform superposition.
    qc.h(range(n_qubits))

    for _ in range(iterations):
        # --- Oracle: phase-flip the marked basis state ---
        for i in range(n_qubits):
            if not ((marked_mask >> i) & 1):
                qc.x(i)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i in range(n_qubits):
            if not ((marked_mask >> i) & 1):
                qc.x(i)

        # --- Diffuser: inversion about the mean ---
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    mask, classical_bitstring, n_edges = classical_find_triangle_k3()
    N = 2 ** n_edges
    M = 1
    iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))

    print(f"Erdos problem #900: no OEIS id in source data (oeis: ['N/A']).")
    print(f"Best-effort graph-theory instance instead: K3 triangle search.")
    print(f"Classical answer (brute force): marked edge-subset mask={mask} "
          f"-> bitstring={classical_bitstring} (n_edges={n_edges}, N={N})")
    print(f"Grover iterations used: {iterations}")

    qc = build_grover_circuit(n_edges, mask, iterations)

    sim = AerSimulator()
    result = sim.run(qc, shots=2048).result()
    counts = result.get_counts()
    top_bitstring = max(counts, key=counts.get)
    top_prob = counts[top_bitstring] / sum(counts.values())

    print(f"Quantum measured counts: {counts}")
    print(f"Most probable outcome: {top_bitstring} (p~{top_prob:.3f})")

    ok = (top_bitstring == classical_bitstring)

    if ok:
        print("PASS")
    else:
        print("FAIL")

    print()
    print("NOTE: this PASS/FAIL verifies a Grover search over K3 triangle "
          "subsets, a graph-theory toy instance chosen because problem #900 "
          "has no OEIS sequence id to ground a real sequence-membership "
          "test in. It does NOT constitute a quantum test of Erdos problem "
          "#900's actual mathematical content.")

    return ok


if __name__ == "__main__":
    main()
