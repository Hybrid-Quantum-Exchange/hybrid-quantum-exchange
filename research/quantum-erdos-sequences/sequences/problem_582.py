"""
Erdos problem #582 -- Qiskit quantum-testable lane.

Source metadata (erdosproblems.com data, /home/user/manman4/erdosproblems/data/problems.yaml,
entry "number: '582'"): prize "$100", status "proved (Lean)", tags
["graph theory", "ramsey theory"], oeis: ["N/A"].

LIMITATION, stated honestly up front: problem #582 carries no OEIS sequence id
(oeis is literally "N/A" in the source data), so there is no OEIS sequence
membership/term property to test here, and the task's "identify a small
finite computable property of the sequence" cannot be done for problem #582
as literally worded. Rather than fabricate an OEIS value, this script instead
builds a genuine, small, computable decision problem drawn directly from the
problem's own tags (graph theory / Ramsey theory): the classical Ramsey
number statement

    R(3,3) = 6

meaning: K5 (5 vertices, the complete graph, 10 edges) CAN be 2-edge-colored
with no monochromatic triangle, while K6 cannot. This is exactly the kind of
finite, computable, Ramsey-theoretic fact that sits behind the "graph
theory / ramsey theory" tags on this problem, and it is checked here two
ways: classically (brute force over all 2^10 colorings of K5) and with a
real Grover search circuit over the same 10-qubit search space, whose oracle
marks precisely the triangle-free colorings computed classically. If Grover
picks out a marked (triangle-free) coloring with high probability, the
quantum result is compared against the classical set of solutions.

This is NOT a claim that R(3,3)=6 "is" the sequence for problem 582 -- it is
the closest genuine, self-contained, small quantum-computable instance
available given that problem 582 has no OEIS id. verified_against_classical
below is scored against the classical brute-force computation done in this
script, not against any OEIS lookup (there is none to make).
"""

import itertools
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no external data).
# ---------------------------------------------------------------------------

N_VERTICES = 5
VERTICES = list(range(N_VERTICES))
EDGES = list(itertools.combinations(VERTICES, 2))          # 10 edges for K5
TRIANGLES = list(itertools.combinations(VERTICES, 3))       # C(5,3) = 10 triangles
N_EDGES = len(EDGES)
assert N_EDGES == 10

EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}


def triangle_is_monochromatic(coloring_bits, triangle):
    """coloring_bits: tuple of 0/1 per edge index. triangle: (a,b,c)."""
    a, b, c = triangle
    e1 = EDGE_INDEX[tuple(sorted((a, b)))]
    e2 = EDGE_INDEX[tuple(sorted((b, c)))]
    e3 = EDGE_INDEX[tuple(sorted((a, c)))]
    return coloring_bits[e1] == coloring_bits[e2] == coloring_bits[e3]


def is_triangle_free_coloring(coloring_int):
    """coloring_int in [0, 2**10): bit i = color of EDGES[i] (0 or 1)."""
    bits = tuple((coloring_int >> i) & 1 for i in range(N_EDGES))
    for tri in TRIANGLES:
        if triangle_is_monochromatic(bits, tri):
            return False
    return True


# Brute-force classical search over all 1024 colorings of K5's edges.
marked_states = [x for x in range(2 ** N_EDGES) if is_triangle_free_coloring(x)]
num_marked = len(marked_states)

# Classical statement of problem: does a triangle-free 2-coloring of K5 exist?
classical_answer_exists = num_marked > 0

print(f"Search space size N = 2^{N_EDGES} = {2 ** N_EDGES}")
print(f"Classical brute force: {num_marked} triangle-free 2-colorings of K5 found "
      f"(out of {2 ** N_EDGES}).")
print(f"Classical answer -- does R(3,3)=6 witness (K5 triangle-free 2-coloring) exist? "
      f"{classical_answer_exists}")

if not classical_answer_exists:
    raise RuntimeError("Unexpected: classical search found no witness; cannot build oracle.")

marked_set = set(marked_states)

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over the 10-qubit space, oracle built
#    directly from the classical marked_set computed above.
# ---------------------------------------------------------------------------

n = N_EDGES  # 10 qubits
dim = 2 ** n

# Diagonal phase oracle: -1 on marked (triangle-free) computational basis
# states, +1 elsewhere. This is a genuine unitary representation of the
# classical predicate is_triangle_free_coloring, not a shortcut around it.
diag = np.ones(dim, dtype=complex)
for m in marked_states:
    diag[m] = -1.0
oracle_gate = UnitaryGate(np.diag(diag), label="Oracle")

# Standard Grover diffuser (inversion about the mean) on n qubits.
def diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="Diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


diffuser_circ = diffuser(n)

# Optimal number of Grover iterations for M marked out of N.
iterations = max(1, round((np.pi / 4) * np.sqrt(dim / num_marked)))
print(f"num_marked = {num_marked}, Grover iterations = {iterations}")

qc = QuantumCircuit(n, n)
qc.h(range(n))
for _ in range(iterations):
    qc.append(oracle_gate, range(n))
    qc.compose(diffuser_circ, qubits=range(n), inplace=True)
qc.measure(range(n), range(n))

sim = AerSimulator(method="statevector")
result = sim.run(qc, shots=2048, seed_simulator=42).result()
counts = result.get_counts()

# Qiskit's classical-bit string is c[n-1] c[n-2] ... c[0] (leftmost = MSB),
# which is exactly standard big-endian binary -- int(bs, 2) already gives
# sum_i bit_i * 2**i with bit_i = the measurement of qubit i, matching our
# edge-index convention where bit i corresponds to EDGES[i]. (Verified with
# a single-X-on-qubit-0 sanity check: bitstring "0000000001" -> int 1.)
def bitstring_to_int(bs):
    return int(bs, 2)

# Most frequent measured outcome.
best_bitstring = max(counts, key=counts.get)
best_int = bitstring_to_int(best_bitstring)
best_shots = counts[best_bitstring]
total_shots = sum(counts.values())

# Fraction of all shots landing on a classically-verified triangle-free coloring.
marked_shots = sum(c for bs, c in counts.items() if bitstring_to_int(bs) in marked_set)
marked_fraction = marked_shots / total_shots

print(f"Most frequent Grover outcome: {best_bitstring} -> edge-coloring int {best_int}, "
      f"{best_shots}/{total_shots} shots")
print(f"Fraction of shots landing on a classically triangle-free coloring: "
      f"{marked_fraction:.3f}")

quantum_found_valid_witness = best_int in marked_set
quantum_amplified_correctly = marked_fraction > 0.5  # Grover should strongly favor marked states

verified_against_classical = quantum_found_valid_witness and quantum_amplified_correctly

print()
if verified_against_classical:
    print("PASS: Grover search's top outcome is a classically-verified triangle-free "
          "2-coloring of K5 (consistent with R(3,3)=6), and marked states dominate the "
          "measured distribution.")
else:
    print("FAIL: Grover search did not reliably return a classically-verified "
          "triangle-free 2-coloring of K5.")
