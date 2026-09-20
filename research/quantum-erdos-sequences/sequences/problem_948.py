"""
Erdos problem #948 (data/problems.yaml, entry `number: "948"`):
    tags: ["number theory", "ramsey theory"]
    oeis: ["N/A"]  -- no OEIS sequence id is associated with this problem.

Because problem #948 carries no OEIS id, there is no specific integer
sequence to test membership/terms of. This script is the documented
best-honest-attempt for that case: rather than fabricate an OEIS value,
it builds a genuine finite, computable property from the problem's own
tag ("ramsey theory") that a small quantum circuit can search over --
the classical fact behind the Ramsey number R(3,3) = 6:

    Property tested: does there exist a 2-coloring of the edges of the
    complete graph K5 (5 vertices, C(5,2) = 10 edges) with NO
    monochromatic triangle?

    Classical answer (well known, and independently re-derived here by
    brute force over all 2^10 = 1024 edge-colorings before any quantum
    code runs): YES -- such colorings exist. Brute force finds exactly 12
    of the 1024 colorings have no monochromatic triangle (the canonical
    "pentagon/pentagram" 2-coloring of K5 and its rotations/reflections
    and color swap). This directly witnesses R(3,3) > 5, i.e. R(3,3) = 6
    combined with the (well known, not re-derived here) fact that every
    2-coloring of K6 does contain a monochromatic triangle.

Quantum approach: Grover's algorithm over the 10 edge-color qubits.
  1. Classically enumerate all 1024 colorings and mark the "good" ones
     (no monochromatic triangle among the C(5,3) = 10 triangles of K5).
  2. Build a diagonal phase oracle (qiskit.circuit.library.Diagonal)
     whose diagonal is -1 exactly on the good marked states and +1
     elsewhere -- built directly from the classical computation, not
     hand-picked.
  3. Run standard Grover diffusion for the optimal number of iterations
     given 20 good states out of 1024, on the ideal AerSimulator
     (statevector method).
  4. Measure. PASS iff the measured 10-bit string decodes to a coloring
     that the classical checker (re-run independently on the sampled
     result) confirms has no monochromatic triangle -- i.e. the quantum
     search actually found a valid instance of the classical property.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Diagonal
from qiskit_aer import AerSimulator

N_VERTICES = 5
EDGES = list(itertools.combinations(range(N_VERTICES), 2))  # 10 edges
TRIANGLES = list(itertools.combinations(range(N_VERTICES), 3))  # 10 triangles
N_EDGES = len(EDGES)
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}


def edge_color(coloring_bits, u, v):
    """coloring_bits: int, bit i = color (0/1) of EDGES[i]. Returns 0/1 color of edge (u,v)."""
    e = (u, v) if u < v else (v, u)
    i = EDGE_INDEX[e]
    return (coloring_bits >> i) & 1


def has_mono_triangle(coloring_bits):
    for (a, b, c) in TRIANGLES:
        c1 = edge_color(coloring_bits, a, b)
        c2 = edge_color(coloring_bits, b, c)
        c3 = edge_color(coloring_bits, a, c)
        if c1 == c2 == c3:
            return True
    return False


def classical_brute_force():
    """Enumerate all 2^10 edge-colorings of K5; return sorted list of the
    ones with no monochromatic triangle (the classical answer)."""
    good = []
    for bits in range(2 ** N_EDGES):
        if not has_mono_triangle(bits):
            good.append(bits)
    return good


def build_grover_circuit(good_states, n_qubits):
    dim = 2 ** n_qubits
    diag = np.ones(dim, dtype=complex)
    for s in good_states:
        diag[s] = -1.0

    oracle = Diagonal(diag.tolist())

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    # Standard diffuser (inversion about the mean) over n_qubits.
    def diffuser():
        d = QuantumCircuit(n_qubits, name="diffuser")
        d.h(range(n_qubits))
        d.x(range(n_qubits))
        d.h(n_qubits - 1)
        d.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        d.h(n_qubits - 1)
        d.x(range(n_qubits))
        d.h(range(n_qubits))
        return d

    m = len(good_states)
    theta = math.asin(math.sqrt(m / dim))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    diff = diffuser()
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diff.to_instruction(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, iterations


def main():
    good_states = classical_brute_force()
    print(f"Classical brute force: {len(good_states)} of {2 ** N_EDGES} "
          f"colorings of K5 have no monochromatic triangle.")
    assert len(good_states) == 12, (
        f"unexpected classical count {len(good_states)}, expected 20"
    )
    good_set = set(good_states)

    qc, iterations = build_grover_circuit(good_states, N_EDGES)
    print(f"Grover circuit built: {N_EDGES} qubits, {iterations} iteration(s).")

    backend = AerSimulator(method="statevector")
    tqc = transpile(qc, backend)
    shots = 2000
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Sum probability mass landing on classically-verified good states.
    hit_shots = 0
    best_bits = None
    best_count = -1
    for bitstring, cnt in counts.items():
        bits = int(bitstring, 2)
        if bits in good_set:
            hit_shots += cnt
        if cnt > best_count:
            best_count = cnt
            best_bits = bits

    hit_fraction = hit_shots / shots
    most_likely_is_good = best_bits in good_set
    # Independently re-check the most likely measured outcome classically.
    reverified = (best_bits is not None) and (not has_mono_triangle(best_bits))

    print(f"Most likely measured outcome: {best_bits:010b} "
          f"(classically re-verified good = {reverified})")
    baseline = len(good_states) / (2 ** N_EDGES)
    print(f"Fraction of shots landing on a classically-verified good state: "
          f"{hit_fraction:.3f} (uniform-random baseline = {baseline:.3f})")

    passed = most_likely_is_good and reverified and hit_fraction > 5 * baseline

    if passed:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
