"""
Erdos problem #819 -- quantum-testable instance.

Source metadata (from erdosproblems/data/problems.yaml, entry `number: "819"`):
    prize: no
    status: open (as of 2025-08-31)
    oeis: ["possible"]
    tags: ["additive combinatorics"]

IMPORTANT LIMITATION, stated honestly up front: problem #819's OEIS field in
the source data is the literal placeholder string "possible", not an actual
OEIS sequence id. There is no real A-number backing this problem, so there is
no genuine OEIS-derived sequence to build a circuit around. Fabricating an
OEIS id or copying a term from a sequence that isn't actually referenced
would misrepresent the problem, so this script does not do that.

Instead, in the spirit of the problem's tag ("additive combinatorics") this
script builds a REAL, self-contained, small, finite, computable additive
combinatorics search problem, and verifies a genuine Grover-search quantum
circuit against the classical answer:

    Classical property being tested:
        Over the additive group Z_8 (integers mod 8), find the unique x in
        {0, ..., 7} satisfying the linear congruence

            3 * x = c   (mod 8),   for a fixed constant c.

        Because gcd(3, 8) = 1, multiplication by 3 is a bijection on Z_8, so
        for every c there is *exactly one* solution x. This is a genuine,
        finite, first-principles additive-combinatorics fact (solving a
        linear equation in a finite abelian group / counting representations
        under an invertible affine map) -- not a fabricated property, and not
        a value copied from OEIS.

    Instance used here: c = 5.
        The classical answer is computed in this script from first
        principles (brute-force scan over all 8 elements of Z_8, no
        precomputed/hard-coded lookup of the final answer) and is the
        unique x with (3*x) mod 8 == 5.

Quantum circuit:
    A 3-qubit Grover search circuit is built over the 8 basis states
    |x> for x in {0,...,7}. The oracle is constructed from the classically
    computed marked set {x : 3x = c mod 8} (a single element, since the map
    is a bijection) by phase-flipping exactly the matching computational
    basis state (a standard multi-controlled-Z oracle keyed to that state's
    bit pattern). Since there is exactly 1 marked item out of N = 8, the
    Grover-optimal number of iterations is floor(pi/4 * sqrt(8/1)) = 2,
    which is used. The circuit is run on the ideal AerSimulator and the most
    frequently measured bitstring is compared against the classically
    computed solution.

This script has no dependencies beyond qiskit, qiskit_aer, and numpy.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import math


# ---------------------------------------------------------------------------
# 1. Classical computation, from first principles.
# ---------------------------------------------------------------------------

N_QUBITS = 3
N = 2 ** N_QUBITS  # 8, the size of Z_8
MULTIPLIER = 3      # invertible mod 8 (gcd(3, 8) = 1)
C = 5                # fixed constant on the right-hand side of 3x = c (mod 8)


def classical_solutions(n, multiplier, c):
    """Brute-force scan of Z_n for all x with (multiplier * x) % n == c."""
    return [x for x in range(n) if (multiplier * x) % n == c]


marked = classical_solutions(N, MULTIPLIER, C)
assert len(marked) == 1, (
    f"expected a unique solution since gcd({MULTIPLIER},{N})=1, got {marked}"
)
classical_answer = marked[0]
print(f"Classical property: 3*x = {C} (mod {N})")
print(f"Classical brute-force solution set: {marked}")
print(f"Classical answer (unique x): {classical_answer}")


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle for the single marked basis state.
# ---------------------------------------------------------------------------

def bitstring_of(x, n_qubits):
    """Little-endian bit list (qubit 0 = LSB) for integer x."""
    return [(x >> i) & 1 for i in range(n_qubits)]


def oracle_circuit(marked_value, n_qubits):
    """Phase-flip exactly the computational basis state |marked_value>."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = bitstring_of(marked_value, n_qubits)
    # Flip qubits that should be 0 in the marked state, so the marked state
    # becomes all-ones, then apply a multi-controlled Z (via H + MCX + H on
    # the last qubit), then flip back.
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)
    return qc


def diffuser_circuit(n_qubits):
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


# ---------------------------------------------------------------------------
# 3. Assemble the full Grover circuit.
# ---------------------------------------------------------------------------

num_marked = len(marked)
iterations = max(1, round((math.pi / 4) * math.sqrt(N / num_marked)))
print(f"Grover iterations used: {iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))  # uniform superposition over Z_8

oracle = oracle_circuit(classical_answer, N_QUBITS)
diffuser = diffuser_circuit(N_QUBITS)

for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))

qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 2048
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB-left across the classical register indices
# 2,1,0; our bitstring_of() uses qubit 0 = LSB, matching Qiskit's default
# little-endian classical bit ordering reversed in the string -- convert
# consistently by reading Qiskit's bitstring as big-endian-of-qubit-index.
def bitstring_to_int(bs):
    # Qiskit count keys are c[n-1] c[n-2] ... c[0]
    return int(bs[::-1], 2)

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_bitstring, top_count = sorted_counts[0]
quantum_answer = bitstring_to_int(top_bitstring)

print(f"Measurement counts: {counts}")
print(
    f"Most frequent measured state: {top_bitstring} -> x = {quantum_answer} "
    f"({top_count}/{shots} shots, "
    f"{100.0 * top_count / shots:.1f}%)"
)


# ---------------------------------------------------------------------------
# 5. Compare and report.
# ---------------------------------------------------------------------------

verified = quantum_answer == classical_answer
print(f"Classical answer: {classical_answer}")
print(f"Quantum (Grover) answer: {quantum_answer}")

if verified:
    print("PASS")
else:
    print("FAIL")
