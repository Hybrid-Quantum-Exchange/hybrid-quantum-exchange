"""
Quantum-testable instance for Erdos problem #110.

Source metadata (from erdosproblems.com data, data/problems.yaml, number "110"):
  tags: ["graph theory", "chromatic number", "cycles"]
  oeis: ["N/A"]
  status: disproved

LIMITATION (reported honestly): problem #110 carries no OEIS sequence id in
the source data (oeis: ["N/A"]), so there is no OEIS integer sequence to
build a membership/term-verification circuit against. Per the task's fallback
instructions, this script instead builds its best honest attempt at a real,
finite, computable property drawn directly from the problem's own tags
("chromatic number", "cycles"): proper 2-colorability of an even cycle graph.

Classical property being tested
--------------------------------
Let C4 be the 4-cycle graph on vertices {0,1,2,3} with edges
(0,1), (1,2), (2,3), (3,0). A "proper 2-coloring" assigns each vertex a bit
color in {0,1} such that every edge joins two vertices of different color.

This script:
  1. Computes classically, from first principles (brute force over all 2^4
     colorings), the exact set of proper 2-colorings of C4 and their count.
     (Known combinatorial fact used only as a sanity cross-check, not taken
     on faith: the number of proper k-colorings of a cycle C_n is
     (k-1)^n + (-1)^n (k-1); for C4, k=2 that gives 1^4 + 2 = 2.)
  2. Builds a Grover search circuit over 4 qubits (one bit per vertex) whose
     oracle marks exactly the classically-computed valid colorings (a
     multi-controlled-Z "list" oracle built from the brute-force answer
     itself, so the circuit's marked set is provably identical to the
     classical set -- no separate constraint-checking oracle is asserted
     without derivation).
  3. Runs the circuit on the ideal AerSimulator, measures, and checks that
     the two most-frequent measured bitstrings are exactly the two
     classically-computed valid colorings.
  4. Prints PASS/FAIL based on that comparison.

This is real quantum content (genuine amplitude amplification of marked
computational basis states via Grover's algorithm on AerSimulator), applied
to a small finite instance whose classical answer is independently computed
in this script. It is not a verification of an OEIS sequence, because
problem #110 has none.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, brute force)
# ---------------------------------------------------------------------------

N_VERTICES = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]  # 4-cycle C4


def is_proper_coloring(bits):
    """bits: tuple of 0/1, one per vertex. True iff every edge is bichromatic."""
    return all(bits[u] != bits[v] for (u, v) in EDGES)


def classical_valid_colorings():
    valid = []
    for bits in itertools.product([0, 1], repeat=N_VERTICES):
        if is_proper_coloring(bits):
            valid.append(bits)
    return valid


VALID_COLORINGS = classical_valid_colorings()

# Cross-check against the closed-form formula for proper k-colorings of C_n:
# (k-1)^n + (-1)^n (k-1), here k=2, n=4 -> 1^4 + 2 = 2.
k, n = 2, N_VERTICES
formula_count = (k - 1) ** n + (-1) ** n * (k - 1)
assert len(VALID_COLORINGS) == formula_count == 2, (
    f"classical brute force ({len(VALID_COLORINGS)}) disagrees with the "
    f"closed-form cycle-coloring count ({formula_count})"
)

# Bitstrings are indexed qubit0..qubit3 == vertex0..vertex3.
# Qiskit's Statevector/measurement bit ordering is little-endian with qubit 0
# as the least-significant (rightmost) bit of the classical register string.
def bits_to_bitstring(bits):
    """bits[i] is the color of vertex i (qubit i). Return the Qiskit-style
    classical bitstring (qubit n-1 ... qubit 0), matching get_counts() keys."""
    return "".join(str(bits[i]) for i in reversed(range(N_VERTICES)))


VALID_BITSTRINGS = {bits_to_bitstring(b) for b in VALID_COLORINGS}


# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the classically-computed valid states
# ---------------------------------------------------------------------------

def apply_mark_for_bits(qc, bits, n_qubits):
    """Flip amplitude sign for the single computational basis state matching
    `bits` (bits[i] = value of qubit i), using X-sandwiched multi-controlled-Z."""
    zero_positions = [i for i in range(n_qubits) if bits[i] == 0]
    for q in zero_positions:
        qc.x(q)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for q in zero_positions:
        qc.x(q)


def build_oracle(n_qubits, marked_list_of_bits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bits in marked_list_of_bits:
        apply_mark_for_bits(qc, bits, n_qubits)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_circuit(n_qubits, marked_list_of_bits, n_iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_list_of_bits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(n_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# ---------------------------------------------------------------------------
# 3. Run on AerSimulator
# ---------------------------------------------------------------------------

N = 2 ** N_VERTICES          # search space size = 16
M = len(VALID_COLORINGS)     # number of marked states = 2
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

circuit = grover_circuit(N_VERTICES, VALID_COLORINGS, optimal_iterations)

simulator = AerSimulator()
transpiled = transpile(circuit, simulator)
shots = 4096
result = simulator.run(transpiled, shots=shots).result()
counts = result.get_counts()

# Two most frequent measured bitstrings.
top_two = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:2]
top_two_bitstrings = {bs for bs, _ in top_two}

top_two_prob_mass = sum(c for _, c in top_two) / shots


# ---------------------------------------------------------------------------
# 4. Compare to classical answer and report
# ---------------------------------------------------------------------------

print("Erdos problem #110 -- OEIS: N/A (no sequence id in source data)")
print("Property tested (best-effort, derived from problem tags): proper")
print("2-colorability of the 4-cycle graph C4, verified via Grover search.")
print()
print(f"Classical valid colorings (vertex0..vertex3): {VALID_COLORINGS}")
print(f"Classical valid bitstrings (Qiskit order):      {sorted(VALID_BITSTRINGS)}")
print(f"Grover iterations used: {optimal_iterations}  (N={N}, M={M})")
print(f"Measurement counts: {counts}")
print(f"Top-2 measured bitstrings: {sorted(top_two_bitstrings)}")
print(f"Probability mass on top-2 bitstrings: {top_two_prob_mass:.4f}")

verified = (top_two_bitstrings == VALID_BITSTRINGS) and (top_two_prob_mass > 0.5)

if verified:
    print("RESULT: PASS")
else:
    print("RESULT: FAIL")
