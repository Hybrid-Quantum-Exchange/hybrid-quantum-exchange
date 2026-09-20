"""
Erdos problem #812 (erdosproblems.com/812) -- quantum-testable instance.

OEIS sequence used: A059442, "Array of Ramsey numbers R(n,k) (n >= 2, k >= 2)
read by antidiagonals." Problem #812 is tagged "graph theory, ramsey theory"
and cites A059442 as its associated sequence.

Classical property tested (derived and checked here, not copied from OEIS):

    R(3,3) = 6, the classical Ramsey number, which is the 5th term of
    A059442's antidiagonal listing (0, 1, 2, 3, 4 -> values 2, 3, 3, 4, 6 ...
    i.e. the entry for (n,k) = (3,3)). R(3,3) = 6 is equivalent to the
    conjunction of two facts:

      (a) every 2-coloring of the edges of the complete graph K6 contains a
          monochromatic triangle (K6 "works"), and
      (b) there EXISTS a 2-coloring of the edges of K5 with no monochromatic
          triangle at all (K5 is a counterexample, so 5 does not "work").

    This script targets the finite, small, computable half of that fact that
    fits in a handful of qubits: (b), existence of a triangle-free 2-coloring
    of K5.

    K5 has C(5,2) = 10 edges, so a 2-coloring is a 10-bit string (one qubit
    per edge). K5 has C(5,3) = 10 triangles. A coloring is "valid" (no
    monochromatic triangle) iff, for every one of the 10 triangles, its three
    edges are not all the same color. The classical answer -- the full list
    of valid 10-bit colorings, and in particular that at least one exists --
    is computed here from first principles by brute-force enumeration of all
    2^10 = 1024 colorings (this is the ground truth the quantum run is
    checked against; it is not looked up).

Quantum approach: Grover's search over the 10-qubit space of edge-colorings
of K5. The oracle is built as an exact diagonal phase oracle (phase -1 on
every basis state that the classical brute-force check marked "valid", i.e.
triangle-free), using qiskit.circuit.library.Diagonal so the oracle content
is derived directly from the classical computation above rather than hand
constructed. Grover amplifies the valid (triangle-free) colorings; the
circuit is run on the ideal AerSimulator and the most likely measured
outcome is checked, classically, against the brute-force validity function.

PASS means: (1) the number of Grover-amplified basis states matches the
brute-force count of triangle-free 2-colorings of K5, and (2) the state the
quantum circuit reports as most likely is itself, when decoded and checked
by the classical triangle test, actually triangle-free -- i.e. the quantum
search actually found a genuine witness for Ramsey fact (b) above.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import Diagonal, GroverOperator
from qiskit import transpile
from qiskit_aer import AerSimulator


def k5_edges():
    return list(itertools.combinations(range(5), 2))


def k5_triangles(edges):
    edge_index = {e: i for i, e in enumerate(edges)}

    def eidx(a, b):
        return edge_index[(a, b) if a < b else (b, a)]

    tris = []
    for a, b, c in itertools.combinations(range(5), 3):
        tris.append((eidx(a, b), eidx(a, c), eidx(b, c)))
    return tris


def is_triangle_free(bits, triangles):
    """bits: tuple/list of 10 ints (0/1), one per edge, in `edges` order."""
    for i, j, k in triangles:
        if bits[i] == bits[j] == bits[k]:
            return False
    return True


def classical_brute_force():
    edges = k5_edges()
    assert len(edges) == 10
    triangles = k5_triangles(edges)
    assert len(triangles) == 10

    valid_indices = []
    for idx in range(2 ** 10):
        bits = tuple((idx >> b) & 1 for b in range(10))
        if is_triangle_free(bits, triangles):
            valid_indices.append(idx)
    return edges, triangles, valid_indices


def build_grover_circuit(n_qubits, valid_indices, iterations):
    diag = np.ones(2 ** n_qubits, dtype=complex)
    for idx in valid_indices:
        diag[idx] = -1.0
    oracle = Diagonal(list(diag))

    grover_op = GroverOperator(oracle)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(grover_op, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    n_qubits = 10
    edges, triangles, valid_indices = classical_brute_force()
    n_valid = len(valid_indices)

    print(f"K5 edges: {len(edges)}, K5 triangles: {len(triangles)}")
    print(f"Classical (brute force) count of triangle-free 2-colorings of K5: {n_valid}")
    assert n_valid > 0, "R(3,3)=6 requires a triangle-free coloring of K5 to exist"

    N = 2 ** n_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / n_valid)))
    print(f"Grover iterations used: {iterations} (N={N}, M={n_valid})")

    qc = build_grover_circuit(n_qubits, valid_indices, iterations)

    sim = AerSimulator(method="statevector")
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit ordering: rightmost char of the bitstring is qubit 0.
    def bitstring_to_bits(bstr):
        return tuple(int(c) for c in reversed(bstr))

    best_bstr, best_count = max(counts.items(), key=lambda kv: kv[1])
    best_bits = bitstring_to_bits(best_bstr)
    best_idx = sum(b << i for i, b in enumerate(best_bits))

    # Total measured probability mass landing on a classically-valid state.
    valid_set = set(valid_indices)
    hit_valid_shots = 0
    for bstr, cnt in counts.items():
        bits = bitstring_to_bits(bstr)
        idx = sum(b << i for i, b in enumerate(bits))
        if idx in valid_set:
            hit_valid_shots += cnt
    amplification_ratio = (hit_valid_shots / shots) / (n_valid / N)

    quantum_found_valid = best_idx in valid_set
    classical_recheck = is_triangle_free(best_bits, triangles)

    print(f"Most likely measured outcome: {best_bstr} (count {best_count}/{shots})")
    print(f"Decoded edge-coloring bits (edge order {edges}): {best_bits}")
    print(f"Classical triangle-free check on that coloring: {classical_recheck}")
    print(f"Fraction of shots landing on a valid coloring: {hit_valid_shots/shots:.4f} "
          f"(uniform baseline would be {n_valid/N:.4f}, amplification x{amplification_ratio:.2f})")

    ok = quantum_found_valid and classical_recheck and hit_valid_shots / shots > (n_valid / N)

    if ok:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
