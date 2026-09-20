"""
Erdos problem #112 -- quantum-testable instance.

Source: erdosproblems.com problem 112 (graph theory / Ramsey theory).
Per data/problems.yaml in the manman4/erdosproblems repo, problem #112 has
oeis: ["possible"] -- i.e. no real OEIS sequence id is recorded for it (the
string "possible" is a placeholder, not an id), and its tags are
["graph theory", "ramsey theory"]. Because there is no usable OEIS sequence
to build a membership/search property from, this script does not test an
OEIS sequence directly. Instead it honors the "ramsey theory" tag with the
smallest genuine, finite, computable fact that sits directly underneath
Ramsey-type problems of this kind: the existence of a 2-colouring of the
edges of the complete graph K4 that contains no monochromatic triangle
(the classical witness that the Ramsey number R(3,3) = 6 is > 4, i.e. that
4 vertices are not enough to force a monochromatic triangle no matter how
the edges are 2-coloured).

Classical property under test
------------------------------
K4 has 6 edges. Label them q0..q5:
    q0 = (0,1)  q1 = (0,2)  q2 = (0,3)
    q3 = (1,2)  q4 = (1,3)  q5 = (2,3)
K4 has exactly 4 triangles, each triangle being "monochromatic" if all three
of its edges get the same colour (bit value):
    T1 = {0,1,3}  T2 = {0,2,4}  T3 = {1,2,5}  T4 = {3,4,5}
(indices into the q-labels above). A colouring x in {0,1}^6 is VALID iff
none of T1..T4 is monochromatic under x.

This script:
  1. Computes the classical answer from first principles: brute-forces all
     2^6 = 64 colourings and counts/collects the valid ones. (For K4 this is
     well known to be nonzero -- since R(3,3)=6, a valid colouring must
     exist for n=4 -- but the exact count and set of solutions is computed
     here, not asserted.)
  2. Builds a genuine Grover search circuit over the 6-qubit colouring space
     whose oracle marks exactly the valid colourings (phase oracle built
     directly from the same monochromatic-triangle boolean expression used
     classically), with the standard number of Grover iterations for a
     6-qubit space with M solutions.
  3. Runs the circuit on the ideal AerSimulator, takes the most frequent
     measured bitstrings, and checks that all of them are members of the
     classically-computed valid set (and that the search amplified valid
     colourings well above the 1/64 uniform baseline).
  4. Prints PASS if the quantum search result is consistent with the
     classical answer, FAIL otherwise.

Honesty note: this is not a test of an OEIS sequence (problem #112 records
no real OEIS id), so it does not claim OEIS verification. It is a genuine
quantum search over a small, explicit, classically-checked boolean
satisfiability instance drawn directly from the Ramsey-theory tag on this
problem.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import GroverOperator, PhaseOracleGate
from qiskit_aer import AerSimulator

NUM_EDGES = 6  # edges of K4
# Each triangle lists the 3 edge-qubit indices that form it.
TRIANGLES = [(0, 1, 3), (0, 2, 4), (1, 2, 5), (3, 4, 5)]


def is_valid_colouring(bits):
    """bits: tuple of 6 ints (0/1), one per edge q0..q5.
    Returns True iff no triangle in TRIANGLES is monochromatic."""
    for t in TRIANGLES:
        v0, v1, v2 = bits[t[0]], bits[t[1]], bits[t[2]]
        if v0 == v1 == v2:
            return False
    return True


def classical_valid_colourings():
    """Brute-force, from first principles, every 2-colouring of K4's edges
    and return the ones with no monochromatic triangle."""
    valid = []
    for bits in itertools.product((0, 1), repeat=NUM_EDGES):
        if is_valid_colouring(bits):
            valid.append(bits)
    return valid


def bits_to_bitstring(bits):
    """Qiskit orders classical register bits with qubit 0 as the
    least-significant (rightmost) character."""
    return "".join(str(b) for b in reversed(bits))


def build_oracle_expression():
    """Boolean expression, true iff the colouring is VALID (no monochromatic
    triangle among TRIANGLES), built directly from the same definition used
    in is_valid_colouring above."""

    def mono_expr(t):
        a, b, c = (f"x{i}" for i in t)
        all_zero = f"(~{a} & ~{b} & ~{c})"
        all_one = f"({a} & {b} & {c})"
        return f"({all_zero} | {all_one})"

    mono_terms = " | ".join(mono_expr(t) for t in TRIANGLES)
    return f"~({mono_terms})"


def build_grover_circuit(num_solutions, num_qubits=NUM_EDGES, shots_hint=True):
    var_order = [f"x{i}" for i in range(num_qubits)]
    oracle_gate = PhaseOracleGate(build_oracle_expression(), var_order=var_order)
    oracle_circuit = QuantumCircuit(num_qubits)
    oracle_circuit.append(oracle_gate, range(num_qubits))

    grover_op = GroverOperator(oracle_circuit)

    # Standard optimal iteration count for Grover's algorithm.
    n_total = 2 ** num_qubits
    theta = math.asin(math.sqrt(num_solutions / n_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.compose(grover_op, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc, iterations


def main():
    valid = classical_valid_colourings()
    valid_bitstrings = {bits_to_bitstring(b) for b in valid}
    num_solutions = len(valid)
    total = 2 ** NUM_EDGES

    print(f"Classical brute force: {num_solutions} valid colourings out of {total}")
    print(f"(uniform-random baseline hit rate would be {num_solutions/total:.4f})")

    qc, iterations = build_grover_circuit(num_solutions)
    print(f"Grover circuit built with {iterations} Grover iteration(s) on "
          f"{NUM_EDGES} qubits, {qc.size()} gates after construction")

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Consider the measured outcomes whose cumulative frequency covers the
    # bulk of the shots (i.e. the amplified peaks Grover found).
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_k = num_solutions  # Grover should spread amplitude ~uniformly over all solutions
    top_outcomes = sorted_counts[:top_k]
    top_total = sum(c for _, c in top_outcomes)
    hit_rate_in_valid_set = sum(c for bs, c in counts.items() if bs in valid_bitstrings) / shots

    all_top_are_valid = all(bs in valid_bitstrings for bs, _ in top_outcomes)

    print(f"Top-{top_k} measured outcomes account for {top_total}/{shots} shots")
    print(f"Fraction of all shots landing on a classically-valid colouring: "
          f"{hit_rate_in_valid_set:.4f}")
    print(f"All of the top-{top_k} most frequent outcomes are valid colourings: "
          f"{all_top_are_valid}")

    # Success criteria: Grover must be amplifying real solutions well above
    # the uniform baseline, and the most frequent outcomes must all be
    # genuine classical solutions (i.e. the quantum search result agrees
    # with the classical answer).
    baseline = num_solutions / total
    amplified = hit_rate_in_valid_set > 3 * baseline
    verified = all_top_are_valid and amplified

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    main()
