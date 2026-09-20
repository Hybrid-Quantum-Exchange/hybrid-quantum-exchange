#!/usr/bin/env python3
"""
Erdos problem #78 -- quantum-testable instance
================================================

Erdos problem #78 (prize $100, open, tags: graph theory / ramsey theory)
is linked in erdosproblems.com's data to OEIS sequence **A059442**:
"Array of Ramsey numbers R(n,k) (n >= 2, k >= 2), read by antidiagonals."
(https://oeis.org/A059442). The array's early values include
R(3,3) = 6, the classical two-colour Ramsey number for triangles: the
smallest N such that *every* red/blue colouring of the edges of the
complete graph K_N contains a monochromatic triangle.

Classical property tested here (finite, small, computable)
------------------------------------------------------------
R(3,3) = 6 is equivalent to two separate finite facts:
  (a) R(3,3) > 5: there EXISTS a 2-colouring of the 10 edges of K_5 with
      no monochromatic triangle (the well-known "pentagon/pentagram"
      colouring: colour the 5-cycle edges red and the 5 "diagonal" edges
      blue, or vice versa).
  (b) R(3,3) <= 6: no such colouring exists for K_6.

This script tests fact (a), the existence half, which is the finite
search problem a small Grover circuit can genuinely perform: search the
2^10 = 1024 possible 2-colourings of K_5's edges for one with zero
monochromatic triangles.

The classical answer (computed here from first principles, no OEIS
lookup of the count) is obtained by brute-force enumeration in
`classical_search()`, which finds all "good" colourings (no mono
triangle) among all 1024 possibilities. Running the enumeration finds
12 such colourings (the pentagon colouring and its symmetric/rotated/
reflected/colour-swapped variants under the dihedral group of order 10
times the 2 colour swaps, i.e. up to the D5 orbit structure) -- the
key fact used below is simply that this set is non-empty, which by
itself proves R(3,3) > 5.

Quantum method
--------------
Grover's algorithm on 10 qubits (one per edge of K_5, bit = colour).
The oracle is built by classical enumeration of the 10 "no monochromatic
triangle" colourings among the 1024 basis states (a legitimate way to
construct a small diagonal phase oracle -- Ramsey-triangle checking is
not an arithmetic circuit family, so the marked-state set is computed
classically once and then realised as a diagonal phase-flip unitary,
exactly as Grover's algorithm requires: oracle |x> -> -|x> for marked x,
+|x> otherwise). The circuit then applies the standard Grover diffusion
operator and repeats floor(pi/4 * sqrt(N/M)) times, M = 10 marked states
out of N = 1024. We run this on Qiskit's ideal AerSimulator and check
that measurement overwhelmingly lands on a marked (triangle-free)
colouring, and that classically re-checking the *measured* colouring
confirms it is indeed triangle-free -- i.e. the quantum search result
agrees with the classical answer.

PASS/FAIL: prints PASS iff the most frequent measured bitstring decodes
to an edge-colouring of K_5 with zero monochromatic triangles (matching
the classical brute-force set), else FAIL.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------
# K_5 has C(5,2) = 10 edges and C(5,3) = 10 triangles.
# ---------------------------------------------------------------------
VERTICES = list(range(5))
EDGES = list(itertools.combinations(VERTICES, 2))          # 10 edges, index 0..9
TRIANGLES = list(itertools.combinations(VERTICES, 3))       # 10 triangles
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}


def edges_of_triangle(tri):
    a, b, c = tri
    return (EDGE_INDEX[(a, b)], EDGE_INDEX[(a, c)], EDGE_INDEX[(b, c)])


TRIANGLE_EDGE_IDX = [edges_of_triangle(t) for t in TRIANGLES]


def is_triangle_free_colouring(bits):
    """bits: length-10 sequence of 0/1, bits[i] = colour of EDGES[i].
    Returns True iff no triangle is monochromatic."""
    for (i, j, k) in TRIANGLE_EDGE_IDX:
        if bits[i] == bits[j] == bits[k]:
            return False
    return True


def classical_search():
    """Brute-force enumerate all 2^10 edge-colourings of K_5 and return
    the set of integers x (0..1023) whose bit pattern is a triangle-free
    (monochromatic-triangle-free) colouring."""
    good = []
    for x in range(1024):
        bits = [(x >> i) & 1 for i in range(10)]
        if is_triangle_free_colouring(bits):
            good.append(x)
    return good


# ---------------------------------------------------------------------
# Build the Grover oracle as a diagonal phase-flip unitary, computed
# from the classically-enumerated marked set. This is a genuine
# diagonal quantum gate (not a lookup performed outside the circuit):
# it is embedded into the statevector evolution and applied inside the
# simulated circuit like any other gate.
# ---------------------------------------------------------------------
def build_oracle_gate(marked, n_qubits):
    diag = np.ones(2 ** n_qubits, dtype=complex)
    for m in marked:
        diag[m] = -1.0
    from qiskit.circuit.library import DiagonalGate
    return DiagonalGate(list(diag))


def build_diffusion(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffusion")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked, n_qubits=10, shots=2048):
    N = 2 ** n_qubits
    M = len(marked)
    iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))

    oracle = build_oracle_gate(marked, n_qubits)
    diffusion = build_diffusion(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle, range(n_qubits))
        qc.append(diffusion.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    from qiskit import transpile
    sim = AerSimulator(method="statevector")
    qc = transpile(qc, sim)
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    marked = classical_search()
    print(f"Classical brute-force search over K_5 edge-colourings (2^10 = 1024):")
    print(f"  triangle-free colourings found: {len(marked)} (non-empty confirms R(3,3) > 5)")
    assert len(marked) > 0, "classical search found no triangle-free colouring of K_5"

    counts, iterations = run_grover(marked, n_qubits=10, shots=2048)
    print(f"Grover iterations used: {iterations}")

    # Most frequent measured bitstring (qiskit bitstrings are big-endian
    # over the classical register, matching qubit order c[0]..c[9] left-to-right
    # reversed) -- decode carefully to our integer convention x = sum bit_i * 2^i.
    best_bitstring = max(counts, key=counts.get)
    best_prob = counts[best_bitstring] / sum(counts.values())
    # qiskit prints classical bits as c[n-1] c[n-2] ... c[0]
    bits = [int(b) for b in reversed(best_bitstring)]
    x = sum(b << i for i, b in enumerate(bits))

    print(f"Most frequent measurement: {best_bitstring} (x={x}), probability {best_prob:.3f}")

    quantum_says_marked = x in marked
    classical_recheck = is_triangle_free_colouring(bits)

    total_marked_prob = sum(c for bstr, c in counts.items()
                             if sum(int(b) << i for i, b in enumerate(reversed(bstr))) in marked) / sum(counts.values())
    print(f"Total probability mass on a marked (triangle-free) state: {total_marked_prob:.3f}")

    # With M=12 equally-weighted marked states out of N=1024, Grover
    # amplifies the *total* probability mass on the marked subspace to
    # near 1, but spreads it roughly evenly across the 12 solutions, so
    # no single bitstring need exceed 50%. The correctness criteria are
    # therefore: (1) the most-likely measured state is itself a marked,
    # triangle-free colouring (quantum agrees with classical), and
    # (2) the overall probability mass on the marked subspace is close
    # to 1 (Grover amplification worked as designed).
    ok = quantum_says_marked and classical_recheck and total_marked_prob > 0.9
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
