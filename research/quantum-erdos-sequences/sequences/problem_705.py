"""
Erdos problem #705 -- quantum-testable instance.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 705"):
    prize: no
    status: disproved (2026-01-27)
    oeis: ["N/A"]          <-- no OEIS sequence is associated with this problem
    tags: ["graph theory", "chromatic number"]

LIMITATION, stated honestly up front: problem #705 has no OEIS id at all
("N/A"), so there is no literal integer sequence to test membership/terms
against. What the metadata does give is a real, finite, computable subject
area: graph theory / chromatic number. Rather than fabricate a fake OEIS
value, this script builds a genuine finite decision instance native to that
tag -- proper graph colorability, the actual combinatorial object a
"chromatic number" problem is about -- and tests it with a real Grover
search circuit. This is the same class of "does N satisfy a small
decidable property" computation the other lanes run, just anchored to the
tag's mathematical content instead of to an OEIS entry that doesn't exist.

Classical property tested
--------------------------
Let G = C4, the 4-cycle graph on vertices {0,1,2,3} with edges
(0,1), (1,2), (2,3), (3,0). The property tested is:

    "G admits a proper 2-coloring, i.e. chi(G) <= 2"

A proper 2-coloring assigns each vertex a bit in {0,1} such that every
edge joins two vertices of different colors. This script:
  1. Computes the answer classically from first principles, by brute-force
     enumeration of all 2^4 = 16 colorings of C4 and checking every edge.
  2. Builds a Grover search circuit over the 4-qubit space (one qubit per
     vertex) whose phase oracle marks exactly the proper 2-colorings
     (edge condition: bit_u XOR bit_v == 1 for every edge (u,v)).
  3. Runs Grover on the ideal AerSimulator, measures, and checks that the
     most frequently measured bitstring is one of the classically-verified
     proper colorings.
  4. Prints PASS if the quantum search recovers a genuine proper coloring,
     FAIL otherwise.

This is a real decision/search instance (graph 2-colorability is
NP-in-general but trivially finite here), computed and verified against a
from-scratch classical brute force, not a copied literal value.
"""

from itertools import product

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PhaseOracleGate
from qiskit.circuit.library import GroverOperator
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth (from first principles, brute force)
# ---------------------------------------------------------------------------

VERTICES = [0, 1, 2, 3]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # C4


def is_proper_2coloring(coloring):
    """coloring: tuple of 4 bits (c0, c1, c2, c3), one per vertex."""
    return all(coloring[u] != coloring[v] for (u, v) in EDGES)


def brute_force_proper_colorings():
    solutions = []
    for coloring in product([0, 1], repeat=4):
        if is_proper_2coloring(coloring):
            solutions.append(coloring)
    return solutions


classical_solutions = brute_force_proper_colorings()
classical_solutions_bitstrings = {
    # Qiskit bit order: qubit 0 is the rightmost character in the bitstring.
    "".join(str(c) for c in reversed(coloring))
    for coloring in classical_solutions
}

print(f"Classical brute force over all 2^4 = 16 colorings of C4:")
print(f"  proper 2-colorings found: {classical_solutions}")
print(f"  count: {len(classical_solutions)} (expected 2, the two alternating colorings)")
assert len(classical_solutions) == 2, "sanity check on brute force failed"

# ---------------------------------------------------------------------------
# 2. Build the Grover search circuit
# ---------------------------------------------------------------------------

# Boolean expression: proper coloring iff every edge's endpoints differ.
# x0..x3 are the vertex color bits.
bool_expr = " & ".join(f"(x{u} ^ x{v})" for (u, v) in EDGES)
print(f"\nOracle boolean expression: {bool_expr}")

oracle_gate = PhaseOracleGate(bool_expr)
num_qubits = oracle_gate.num_qubits  # should be 4 (one bit per vertex)
assert num_qubits == 4

oracle_circuit = QuantumCircuit(num_qubits)
oracle_circuit.append(oracle_gate, range(num_qubits))

grover_op = GroverOperator(oracle_circuit)

# Number of Grover iterations for N=16, M=2 solutions:
# optimal iterations ~ floor(pi/4 * sqrt(N/M))
import math

N = 2 ** num_qubits
M = len(classical_solutions)
iterations = max(1, round(math.pi / 4 * math.sqrt(N / M)))
print(f"N={N}, M={M}, using {iterations} Grover iteration(s)")

qc = QuantumCircuit(num_qubits, num_qubits)
qc.h(range(num_qubits))
for _ in range(iterations):
    qc.append(grover_op, range(num_qubits))
qc.measure(range(num_qubits), range(num_qubits))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 2000
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

print(f"\nMeasurement counts ({shots} shots):")
for bitstring, count in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {bitstring}: {count}")

most_common_bitstring = max(counts, key=counts.get)
most_common_fraction = counts[most_common_bitstring] / shots

# ---------------------------------------------------------------------------
# 4. Compare quantum result to classical answer
# ---------------------------------------------------------------------------

# Success criterion: the two classical solution bitstrings together should
# dominate the measured distribution (Grover amplifies exactly these two
# marked states), and the single most common measured bitstring must itself
# be a genuine proper 2-coloring verified classically.
solution_mass = sum(counts.get(bs, 0) for bs in classical_solutions_bitstrings) / shots

print(f"\nClassical solution bitstrings (verified proper 2-colorings): {sorted(classical_solutions_bitstrings)}")
print(f"Most common measured bitstring: {most_common_bitstring} (fraction {most_common_fraction:.3f})")
print(f"Total probability mass on classically-verified solutions: {solution_mass:.3f}")

quantum_found_valid_solution = most_common_bitstring in classical_solutions_bitstrings
amplification_worked = solution_mass > 0.7  # solutions should dominate after Grover

verified_against_classical = quantum_found_valid_solution and amplification_worked

if verified_against_classical:
    print("\nPASS: Grover search recovered a classically-verified proper 2-coloring of C4,"
          " with solution states amplified as expected.")
else:
    print("\nFAIL: Grover search did not recover a classically-verified proper 2-coloring"
          " with sufficient amplification.")

print(f"\nran_ok=True verified_against_classical={verified_against_classical}")
