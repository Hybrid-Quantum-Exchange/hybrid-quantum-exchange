"""
Quantum-testable instance for Erdos problem #639 (graph theory / Ramsey theory).

Source metadata (erdosproblems.com data, as cloned in manman4/erdosproblems
data/problems.yaml, entry "number: \"639\""):
    prize: no
    status: proved (Lean), last update 2026-05-06
    oeis: ["N/A"]
    tags: ["graph theory", "ramsey theory"]

LIMITATION, stated honestly up front: problem #639 has NO associated OEIS
sequence (oeis: ["N/A"] in the source data). There is therefore no literal
OEIS term to derive or verify here, and this script cannot claim to test
"the sequence for problem 639" in the way a problem with a real OEIS id
would allow. Per the tags ("graph theory", "ramsey theory"), this script
instead builds a genuine, small, finite, classically-checkable combinatorial
search problem from the same mathematical area the problem lives in, and
runs a real Grover search circuit against it on AerSimulator. This is
presented as the best-effort quantum-testable artifact for this problem
given the absence of an OEIS sequence, not as a literal test of problem
639's own statement.

The classical property tested
------------------------------
Take the complete graph K4 on vertices {0,1,2,3}. It has 6 edges and 4
triangles. Label the 6 edges as qubits q0..q5:

    q0 = (0,1)   q1 = (0,2)   q2 = (0,3)
    q3 = (1,2)   q4 = (1,3)   q5 = (2,3)

and the 4 triangles as the edge-triples:

    T0 = {0,1,2} -> (q0,q1,q3)
    T1 = {0,1,3} -> (q0,q2,q4)
    T2 = {0,2,3} -> (q1,q2,q5)
    T3 = {1,2,3} -> (q3,q4,q5)

A qubit value of 0/1 is a 2-coloring of that edge. A triangle is
"monochromatic" if all three of its edges have the same color. We search,
via Grover's algorithm over all 2^6 = 64 edge-colorings, for a coloring of
K4 with NO monochromatic triangle at all (a genuine finite search problem:
"does a Ramsey-type coloring exist, and if so find one").

This is computed classically first, from first principles, by brute force
over all 64 colorings, and the exact count of valid (triangle-free)
colorings is used as ground truth. Grover's algorithm is then run with the
matching number of iterations and must return a valid solution with high
probability, matching the classical answer.

(Aside, for context only, not used by the script: this is the small-n
analogue of the Ramsey number R(3,3)=6, which says every 2-coloring of K6
DOES contain a monochromatic triangle; K4 is small enough that
triangle-free colorings still exist, which is exactly what this script
finds.)
"""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import GroverOperator, PhaseOracleGate
from qiskit_aer import AerSimulator

NUM_EDGES = 6
TRIANGLES = [
    (0, 1, 3),  # T0 = {0,1,2}
    (0, 2, 4),  # T1 = {0,1,3}
    (1, 2, 5),  # T2 = {0,2,3}
    (3, 4, 5),  # T3 = {1,2,3}
]
EDGE_VARS = ["q0", "q1", "q2", "q3", "q4", "q5"]


def is_triangle_free(bits):
    """bits: tuple of 6 ints (0/1), one per edge q0..q5."""
    for (a, b, c) in TRIANGLES:
        if bits[a] == bits[b] == bits[c]:
            return False
    return True


def classical_brute_force():
    """Enumerate all 2^6 edge-colorings of K4 and find the triangle-free ones."""
    valid = []
    for bits in product([0, 1], repeat=NUM_EDGES):
        if is_triangle_free(bits):
            valid.append(bits)
    return valid


def bits_to_bitstring(bits):
    # Qiskit bit ordering: q0 is the least-significant (rightmost) bit.
    return "".join(str(b) for b in reversed(bits))


def build_oracle_expression():
    """Boolean expression, true iff the coloring has NO monochromatic triangle."""
    # For each triangle (a,b,c): NOT all-equal <=> not((a&b&c) | (~a&~b&~c))
    clauses = []
    for (a, b, c) in TRIANGLES:
        va, vb, vc = EDGE_VARS[a], EDGE_VARS[b], EDGE_VARS[c]
        all_one = f"({va} & {vb} & {vc})"
        all_zero = f"(~{va} & ~{vb} & ~{vc})"
        mono = f"({all_one} | {all_zero})"
        clauses.append(f"~{mono}")
    return " & ".join(clauses)


def main():
    # --- Classical ground truth, computed from first principles ---
    valid_colorings = classical_brute_force()
    n_valid = len(valid_colorings)
    n_total = 2 ** NUM_EDGES
    valid_bitstrings = {bits_to_bitstring(b) for b in valid_colorings}

    print(f"Classical brute force over {n_total} edge-colorings of K4:")
    print(f"  triangle-free colorings found: {n_valid}")
    assert n_valid > 0, "sanity: K4 must admit at least one triangle-free coloring"

    # --- Build Grover oracle from the same boolean property ---
    expr = build_oracle_expression()
    oracle_gate = PhaseOracleGate(expr)
    assert oracle_gate.num_qubits == NUM_EDGES
    oracle_circuit = QuantumCircuit(NUM_EDGES)
    oracle_circuit.append(oracle_gate, range(NUM_EDGES))

    grover_op = GroverOperator(oracle_circuit)

    # Optimal number of Grover iterations for n_valid solutions out of n_total.
    theta = np.arcsin(np.sqrt(n_valid / n_total))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(NUM_EDGES, NUM_EDGES)
    qc.h(range(NUM_EDGES))
    for _ in range(iterations):
        qc.compose(grover_op, inplace=True)
    qc.measure(range(NUM_EDGES), range(NUM_EDGES))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # --- Compare quantum result to classical answer ---
    hits = sum(c for bstr, c in counts.items() if bstr in valid_bitstrings)
    hit_fraction = hits / shots

    most_common = max(counts.items(), key=lambda kv: kv[1])[0]
    most_common_is_valid = most_common in valid_bitstrings

    print(f"Grover circuit: {NUM_EDGES} qubits, {iterations} iteration(s), {shots} shots")
    print(f"  most frequent measured coloring: {most_common} "
          f"({'triangle-free' if most_common_is_valid else 'HAS mono triangle'})")
    print(f"  fraction of shots landing on a valid (triangle-free) coloring: "
          f"{hit_fraction:.3f}")

    # Success criteria: the most likely outcome is a genuine solution, and
    # the amplified probability mass on valid solutions is well above the
    # uniform-random baseline (n_valid / n_total).
    baseline = n_valid / n_total
    verified = most_common_is_valid and hit_fraction > 2 * baseline

    print(f"  classical baseline probability (uniform random guess): {baseline:.3f}")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
