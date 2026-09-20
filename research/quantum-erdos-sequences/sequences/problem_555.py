"""
Erdos problem #555 (erdosproblems.com), quantum-testable instance.

Source metadata (data/problems.yaml in teorth/erdosproblems, entry "555"):
    tags: ["graph theory", "ramsey theory"]
    oeis: ["A389313", "possible"]   # the "possible" flag in the source data
                                     # itself marks this OEIS association as
                                     # tentative, not confirmed by the problem
                                     # authors.
A389313 is (per available OEIS metadata) associated with diagonal/cycle
Ramsey-type numbers r(C_n, C_n) for cycles. Because the precise defining
formula of A389313 could not be independently re-derived here with
confidence (the "possible" tag on the source data itself signals the same
uncertainty), this script does not gamble on reproducing a specific OEIS
term. Instead -- consistent with the task's fallback instructions -- it
targets the smallest classical fact that is unambiguously inside the same
mathematical territory (diagonal Ramsey numbers for triangles, the
best-known and most elementary object in Ramsey theory) and builds a real,
non-trivial quantum circuit that verifies it:

    CLASSICAL PROPERTY TESTED
    --------------------------
    R(3,3) = 6, equivalently: there EXISTS a 2-colouring of the edges of the
    complete graph K5 (5 vertices, C(5,2) = 10 edges) that contains no
    monochromatic triangle, while no such colouring exists for K6.

    This script only needs the K5 half (the existence witness), encoded as
    a 10-bit search space (one bit per edge, bit = colour of that edge).
    K5 has C(5,3) = 10 triangles. A colouring is "good" (a witness for
    R(3,3) > 5, i.e. for R(3,3) >= 6) iff none of its 10 triangles is
    monochromatic.

    The classical answer is computed from first principles in this script
    by brute force over all 2^10 = 1024 colourings, both to (a) confirm at
    least one good colouring exists and (b) know the true count M of good
    colourings, needed to pick the optimal number of Grover iterations.

    QUANTUM CIRCUIT
    ----------------
    A Grover search over the 10-qubit colouring space, with a phase oracle
    (qiskit.circuit.library.PhaseOracle) built directly from the boolean
    formula "no triangle among the 10 is monochromatic", then the standard
    Grover diffusion operator, run on the ideal AerSimulator. The circuit
    genuinely computes/searches -- the oracle is derived symbolically from
    the graph's triangle structure, not hard-coded to a specific answer.

    PASS/FAIL
    ---------
    After running the optimal number of Grover iterations, the script takes
    the most frequently measured 10-bit string and checks classically that
    it is indeed a valid triangle-free-monochromatic-free colouring of K5.
    PASS iff the quantum-selected string is in the classically-computed set
    of good colourings.
"""

import itertools
import math

from qiskit import transpile
from qiskit.circuit.library import PhaseOracle
from qiskit.circuit.library import GroverOperator
from qiskit_aer import AerSimulator


def edges_of_k5():
    return list(itertools.combinations(range(5), 2))


def triangles_of_k5(edges):
    edge_index = {e: i for i, e in enumerate(edges)}
    tris = []
    for a, b, c in itertools.combinations(range(5), 3):
        e1 = edge_index[tuple(sorted((a, b)))]
        e2 = edge_index[tuple(sorted((a, c)))]
        e3 = edge_index[tuple(sorted((b, c)))]
        tris.append((e1, e2, e3))
    return tris


def is_good_coloring(bits, triangles):
    """bits: tuple of 0/1, length 10 (one per edge). True iff no
    monochromatic triangle."""
    for (e1, e2, e3) in triangles:
        v1, v2, v3 = bits[e1], bits[e2], bits[e3]
        if v1 == v2 == v3:
            return False
    return True


def classical_brute_force(triangles, n_edges):
    good = []
    for bits in itertools.product([0, 1], repeat=n_edges):
        if is_good_coloring(bits, triangles):
            good.append(bits)
    return good


def build_boolean_expression(triangles, n_edges):
    var = [f"e{i}" for i in range(n_edges)]
    mono_terms = []
    for (e1, e2, e3) in triangles:
        a, b, c = var[e1], var[e2], var[e3]
        mono_terms.append(f"({a} & {b} & {c}) | (~{a} & ~{b} & ~{c})")
    mono_any = " | ".join(f"({t})" for t in mono_terms)
    # "good" (no monochromatic triangle) = NOT(mono_any)
    expr = f"~({mono_any})"
    return expr, var


def main():
    edges = edges_of_k5()
    n_edges = len(edges)
    assert n_edges == 10

    triangles = triangles_of_k5(edges)
    assert len(triangles) == 10

    # --- classical ground truth, computed from first principles ---
    good_colorings = classical_brute_force(triangles, n_edges)
    good_set = set(good_colorings)
    m = len(good_set)
    n = 2 ** n_edges
    print(f"Classical brute force over K5 (10 edges, 2^10={n} colourings):")
    print(f"  triangle-monochromatic-free colourings found: {m}")
    assert m > 0, "R(3,3) > 5 witness must exist classically (known fact)"

    # --- quantum circuit: Grover search for a good colouring ---
    expr, var_order = build_boolean_expression(triangles, n_edges)
    oracle = PhaseOracle(expr)

    grover_op = GroverOperator(oracle)

    iterations = max(1, round((math.pi / 4) * math.sqrt(n / m)))
    print(f"Grover iterations used: {iterations} (N={n}, M={m})")

    qc = grover_op.decompose().copy_empty_like()
    # Rebuild full circuit: H on all qubits, then `iterations` Grover steps,
    # then measurement.
    from qiskit import QuantumCircuit

    num_qubits = oracle.num_qubits
    full = QuantumCircuit(num_qubits, num_qubits)
    full.h(range(num_qubits))
    for _ in range(iterations):
        full.compose(grover_op, inplace=True)
    full.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    tqc = transpile(full, sim)
    result = sim.run(tqc, shots=2048).result()
    counts = result.get_counts()

    # Most frequent measured bitstring.
    best_bitstring = max(counts, key=counts.get)
    # Qiskit's PhaseOracle exposes the oracle's own variable ordering
    # (oracle.variables), which is what maps qubit index -> edge name; use
    # it directly instead of assuming little/big-endian conventions.
    oracle_vars = [str(v) for v in oracle.boolean_expression.args]
    # Qiskit classical-register bitstrings are printed with qubit (n-1) first.
    bits_by_qubit = list(reversed(best_bitstring))
    var_to_bit = {oracle_vars[q]: int(bits_by_qubit[q]) for q in range(num_qubits)}
    measured_bits = tuple(var_to_bit[f"e{i}"] for i in range(n_edges))

    quantum_says_good = is_good_coloring(measured_bits, triangles)
    print(f"Most frequent measurement (edge-bit order e0..e9): {measured_bits}")
    print(f"  measured coloring is monochromatic-triangle-free: {quantum_says_good}")
    print(f"  classical set agrees: {measured_bits in good_set}")

    passed = quantum_says_good and (measured_bits in good_set)

    if passed:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
