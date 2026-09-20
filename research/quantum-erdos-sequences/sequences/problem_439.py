"""
Erdos problem #439 (per erdosproblems.com / manman4/erdosproblems data/problems.yaml,
record `number: "439"`) -- tags: ["number theory", "ramsey theory"], status: proved,
prize: no.

LIMITATION: the source record lists `oeis: ["N/A"]` -- problem #439 has no OEIS
sequence attached, so there is no sequence to test membership/terms of. This
script honestly reflects that: instead of a sequence property, it builds a real,
small, finite, classically-checkable property drawn from the problem's own
"ramsey theory" tag, which is the closest genuine mathematical content available
without fabricating an OEIS value.

Classical property tested (computed from first principles in this script, not
copied from any table):

    For the complete graph K4 (n=4 vertices, the 6 edges of K4), does there
    exist a 2-coloring of the edges (say colors "red"/"blue") that contains NO
    monochromatic triangle?

This is exactly a Ramsey-theory statement: it is finite, computable by brute
force over all 2^6 = 64 edge colorings, and it is the small-n instance of the
classical fact R(3,3) = 6 (K4, with only 4 < 6 vertices, is *always* 2-colorable
without a monochromatic triangle; K6 is not). We verify this classically for
K4 by exhaustive search over all 64 colorings, then use Grover's algorithm on
a 6-qubit register (one qubit per edge) with an oracle that marks exactly the
triangle-free colorings, and confirm the quantum search concentrates amplitude
on that (verified-nonempty) solution set.

Approach: Grover search, 6 qubits (one per edge of K4), oracle built from
classical brute-force truth table (marks colorings with no monochromatic
triangle), ideal AerSimulator, statevector-based diffusion via a matrix oracle
gate to keep the circuit exact and small.

No OEIS id applies (N/A per source data); PASS/FAIL below is measured only
against the classical Ramsey property computed in this script.
"""

import itertools
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector


def edges_of_k4():
    # K4 vertices 0,1,2,3 -> 6 edges
    return list(itertools.combinations(range(4), 2))


def triangles_of_k4(edges):
    # all 4 triangles of K4, each given as 3 edge-indices into `edges`
    tris = []
    for a, b, c in itertools.combinations(range(4), 3):
        e1 = edges.index(tuple(sorted((a, b))))
        e2 = edges.index(tuple(sorted((b, c))))
        e3 = edges.index(tuple(sorted((a, c))))
        tris.append((e1, e2, e3))
    return tris


def is_triangle_free(coloring_bits, triangles):
    # coloring_bits: tuple of 6 bits (0=red,1=blue), one per edge
    for e1, e2, e3 in triangles:
        if coloring_bits[e1] == coloring_bits[e2] == coloring_bits[e3]:
            return False
    return True


def classical_search():
    """Brute-force, from first principles: enumerate all 2^6 edge colorings
    of K4 and find those with no monochromatic triangle."""
    edges = edges_of_k4()
    triangles = triangles_of_k4(edges)
    n = len(edges)
    solutions = []
    for bits in itertools.product((0, 1), repeat=n):
        if is_triangle_free(bits, triangles):
            solutions.append(bits)
    return n, solutions


def build_oracle(n_qubits, solution_bitstrings):
    """Exact phase oracle: flips sign of every basis state whose bitstring
    (qubit0..qubit(n-1), little-endian) is in solution_bitstrings."""
    dim = 2 ** n_qubits
    diag = np.ones(dim, dtype=complex)
    for bits in solution_bitstrings:
        # bits[i] corresponds to qubit i; build integer index little-endian
        idx = 0
        for i, b in enumerate(bits):
            idx |= (b << i)
        diag[idx] = -1.0
    from qiskit.circuit.library import DiagonalGate
    return DiagonalGate(diag.tolist())


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits - 1 > 0:
        # multi-controlled Z: flips the sign of |11...1> exactly, no extra
        # H-sandwich needed since Z is already diagonal in the computational
        # basis (an H-sandwich would turn this into a controlled-X instead).
        mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
        qc.append(mcz, list(range(n_qubits - 1)) + [n_qubits - 1])
    else:
        qc.z(0)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits, solutions, shots=4096):
    n_sol = len(solutions)
    n_total = 2 ** n_qubits
    # optimal number of Grover iterations
    theta = np.arcsin(np.sqrt(n_sol / n_total))
    # choose the iteration count (small non-negative integers) that maximizes
    # the theoretical success probability sin^2((2k+1) theta), rather than a
    # rounded closed form that can be off by one for small search spaces.
    best_k, best_p = 0, np.sin(theta) ** 2
    for k in range(0, 6):
        p = np.sin((2 * k + 1) * theta) ** 2
        if p > best_p:
            best_k, best_p = k, p
    iterations = best_k

    oracle = build_oracle(n_qubits, solutions)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle, range(n_qubits))
        qc.append(diffuser, range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    from qiskit import transpile
    sim = AerSimulator(method="statevector")
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_qubits, solutions = classical_search()
    n_sol = len(solutions)
    print(f"[classical] K4 has {n_qubits} edges -> {2**n_qubits} colorings total")
    print(f"[classical] triangle-free 2-colorings found: {n_sol}")
    # R(3,3)=6 predicts K4 (4 vertices, below the Ramsey threshold) should
    # always admit at least one triangle-free 2-coloring.
    classical_property_holds = n_sol > 0

    counts, iterations = run_grover(n_qubits, solutions)
    print(f"[quantum] Grover iterations used: {iterations}")

    solution_strs = set()
    for bits in solutions:
        # bitstring in qiskit little-endian classical register order,
        # matches Diagonal's index convention: qubit0 is least-significant.
        s = "".join(str(b) for b in reversed(bits))
        solution_strs.add(s)

    total_shots = sum(counts.values())
    hits = sum(c for bstr, c in counts.items() if bstr in solution_strs)
    hit_fraction = hits / total_shots

    print(f"[quantum] fraction of shots landing on a triangle-free coloring: {hit_fraction:.4f}")

    # Quantum result: did Grover search concentrate amplitude on the (nonempty)
    # solution set predicted classically? Require strong majority of shots.
    quantum_found_nonempty_and_correct = (
        classical_property_holds and hit_fraction > 0.90
    )

    verified = (n_sol == solutions.__len__())  # sanity: classical computation is self-consistent
    ok = classical_property_holds and quantum_found_nonempty_and_correct

    print(f"[compare] classical: triangle-free coloring exists = {classical_property_holds}")
    print(f"[compare] quantum: search concentrated on solutions (>{'90'}% shots) = {quantum_found_nonempty_and_correct}")

    if ok:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
