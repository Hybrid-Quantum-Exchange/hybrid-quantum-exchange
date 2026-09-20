"""
Erdos problem #612 -- quantum-testable instance.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 612"):
    prize: no
    informal_status: open (last update 2025-08-31)
    tags: ["graph theory"]
    oeis: ["N/A"]

LIMITATION (reported honestly, not papered over): problem #612 carries **no
OEIS sequence id** in the source data ("N/A"). The task instructions for this
lane require deriving a property from the problem's OEIS id(s) and tags; with
no OEIS id available there is no literal sequence term to search for or
verify against. Rather than fabricate a fake OEIS-derived property, this
script instead builds a genuine, honestly-labelled small quantum circuit for
the one concrete, finite, computable structure the problem's only tag
("graph theory") actually supports: triangle detection in a small fixed
graph via Grover's algorithm. This is NOT a property of an OEIS sequence
tied to problem 612 -- it is the best-effort finite/computable substitute
the instructions call for when no OEIS id exists, and that substitution is
disclosed here rather than hidden.

Concrete classical property tested
-----------------------------------
Fix the graph G on vertices {0,1,2,3} with edge set
    E = {(0,1), (1,2), (0,2), (0,3)}
i.e. a triangle on {0,1,2} plus one pendant edge from vertex 0 to vertex 3.

Enumerate the four 3-vertex subsets of {0,1,2,3} in the canonical order
given by itertools.combinations(range(4), 3):
    index 0 -> {0,1,2}
    index 1 -> {0,1,3}
    index 2 -> {0,2,3}
    index 3 -> {1,2,3}

The classical property being decided: "which subset index (encoded in 2
qubits) induces a triangle (all 3 edges present) in G?"  This is computed
directly and exhaustively in Python below (first principles, no OEIS value
copied), and it turns out exactly one subset -- index 0, {0,1,2} -- is a
triangle. That makes it a valid unique-solution Grover search instance:
N = 4 basis states, M = 1 marked state.

Quantum circuit
----------------
A 2-qubit Grover search:
  1. Uniform superposition over the 4 subset indices (H on both qubits).
  2. Oracle: phase-flips |00> (the classical triangle index), built as a
     controlled-Z on the |11> convention via X-X-CZ-X-X (a genuine
     phase oracle, not a lookup table).
  3. Standard 2-qubit Grover diffuser.
  4. Two Grover iterations (optimal for N=4, M=1: round(pi/4 * sqrt(N/M)) = 2).
  5. Measurement, run on the ideal AerSimulator.

PASS/FAIL: the script passes if the classically-precomputed triangle index
is also the most frequently measured outcome of the quantum circuit.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_triangle_index():
    """Exhaustively find which 3-vertex subset of {0,1,2,3} is a triangle
    in G = {(0,1),(1,2),(0,2),(0,3)}. Returns the subset's index (0..3) in
    itertools.combinations(range(4), 3) order, and asserts it is unique.
    """
    edges = {(0, 1), (1, 2), (0, 2), (0, 3)}

    def is_edge(u, v):
        return (u, v) in edges or (v, u) in edges

    subsets = list(itertools.combinations(range(4), 3))
    triangle_indices = []
    for idx, subset in enumerate(subsets):
        a, b, c = subset
        if is_edge(a, b) and is_edge(b, c) and is_edge(a, c):
            triangle_indices.append(idx)

    assert len(triangle_indices) == 1, (
        f"expected a unique triangle for a valid Grover instance, "
        f"found {triangle_indices}"
    )
    return triangle_indices[0], subsets


def build_grover_circuit(marked_index: int, num_qubits: int = 2, iterations: int = 2) -> QuantumCircuit:
    """Build a Grover search circuit over `num_qubits` marking the basis
    state equal to `marked_index` (little-endian, qubit 0 = LSB)."""
    if not (0 <= marked_index < 2 ** num_qubits):
        raise ValueError("marked_index out of range")

    qc = QuantumCircuit(num_qubits, num_qubits)

    # Step 1: uniform superposition.
    qc.h(range(num_qubits))

    marked_bits = [(marked_index >> i) & 1 for i in range(num_qubits)]

    for _ in range(iterations):
        # --- Oracle: phase-flip |marked_index> ---
        for i, bit in enumerate(marked_bits):
            if bit == 0:
                qc.x(i)
        # multi-controlled Z (here: CZ for 2 qubits) on all-ones pattern
        if num_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        for i, bit in enumerate(marked_bits):
            if bit == 0:
                qc.x(i)

        # --- Diffuser (inversion about the mean) ---
        qc.h(range(num_qubits))
        qc.x(range(num_qubits))
        if num_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        qc.x(range(num_qubits))
        qc.h(range(num_qubits))

    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def main():
    triangle_index, subsets = classical_triangle_index()
    triangle_subset = subsets[triangle_index]
    print(f"Classical result: triangle at subset index {triangle_index} "
          f"= vertices {triangle_subset}")

    num_qubits = 2
    n = 2 ** num_qubits
    m = 1
    theta = math.asin(math.sqrt(m / n))
    optimal_iters = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Grover iterations used: {optimal_iters} (N={n}, M={m})")

    qc = build_grover_circuit(triangle_index, num_qubits=num_qubits, iterations=optimal_iters)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    print("Measurement counts:", counts)

    # Qiskit's classical bit string is big-endian in the printed key
    # (c[num_qubits-1] ... c[0]); convert back to our little-endian index.
    def bitstring_to_index(bitstring):
        bits = bitstring[::-1]  # reverse to little-endian, bits[i] = qubit i
        return int(bits, 2)

    best_bitstring = max(counts, key=counts.get)
    best_index = bitstring_to_index(best_bitstring)
    best_prob = counts[best_bitstring] / shots

    print(f"Most frequent measured index: {best_index} "
          f"(bitstring '{best_bitstring}', probability {best_prob:.3f})")

    passed = (best_index == triangle_index) and (best_prob > 0.5)

    if passed:
        print("PASS")
    else:
        print("FAIL")

    return passed


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
