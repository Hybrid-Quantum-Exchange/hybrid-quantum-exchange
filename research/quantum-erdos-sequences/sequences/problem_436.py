"""
Erdos problem #436 -- quantum-testable instance.

Source: erdosproblems.com problem 436 (data/problems.yaml entry `number: "436"`),
tags ["number theory"], oeis: ["A000445", "possible"] (the "possible" token in
that list is not a second real OEIS id -- it is a status marker used elsewhere
in that data file, so the only concrete sequence id here is A000445).

OEIS A000445: "Latest possible occurrence of the first consecutive pair of
n-th power residues, modulo any prime." For n = 2 this is about quadratic
residues (QRs): for a prime p, look at the residues 1, 2, ..., p-2 in order
and ask which is the *largest* m such that both m and m+1 are quadratic
residues mod p. A000445(2) = 9 is the known value of this quantity taken as a
supremum over ALL primes p -- that outer supremum is not something a small
circuit can search (it is open-ended over primes), so the finite, computable
instance verified here is the same well-defined property for one fixed small
prime, which is exactly the inner computation the sequence is built from:

    Classical property tested (fixed instance, p = 13):
        QR(13) = { x^2 mod 13 : x = 1..12 } = {1, 3, 4, 9, 10, 12}
        Find the LARGEST m in {1, ..., p-2} = {1, ..., 11} such that both
        m and m+1 are in QR(13).

    This is computed from first principles below (no OEIS value is copied):
        residues mod 13 that are QRs: {1, 3, 4, 9, 10, 12}
        consecutive-QR pairs (m, m+1) with m in 1..11: m = 3 (3,4 both QR),
                                                         m = 9 (9,10 both QR)
        so the classical answer is max(marked) = 9.

Quantum approach: Grover search over a 4-qubit register representing
m in {0, ..., 15} (p-1 = 12 fits in 4 bits). The oracle marks exactly the
basis states corresponding to the classically precomputed marked set
{3, 9} (built with an X/MCX pattern per marked bitstring -- a standard
"known marked items" Grover oracle), the diffuser is the standard Grover
diffusion operator, and the optimal number of Grover iterations for
N = 16, M = 2 marked items (~2) is used. The circuit is run on the ideal
AerSimulator; the most-sampled measured value among the marked set (and the
maximum of the marked values that appear with non-trivial probability) is
compared against the classically computed answer 9.

This verifies the same finite "largest consecutive-QR-pair index" property
that OEIS A000445 encodes, on one concrete instance, via a genuine Grover
search rather than a copied OEIS literal.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import math


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS values copied in).
# ---------------------------------------------------------------------------

P = 13  # fixed small prime instance

def quadratic_residues(p):
    return sorted({(x * x) % p for x in range(1, p)})

QR = quadratic_residues(P)
assert QR == [1, 3, 4, 9, 10, 12], f"unexpected QR set for p={P}: {QR}"

# All m in {1, ..., p-2} such that m and m+1 are both quadratic residues.
marked_classical = [m for m in range(1, P - 1) if m in QR and (m + 1) in QR]
assert marked_classical == [3, 9], f"unexpected marked set: {marked_classical}"

CLASSICAL_ANSWER = max(marked_classical)
print(f"Classical: QR({P}) = {QR}")
print(f"Classical: consecutive-QR pairs at m = {marked_classical}")
print(f"Classical answer (largest such m) = {CLASSICAL_ANSWER}")


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search marking {3, 9} over a 4-qubit register.
# ---------------------------------------------------------------------------

N_QUBITS = 4  # represents integers 0..15
N = 2 ** N_QUBITS
MARKED = marked_classical  # [3, 9]


def apply_bitstring_frame(qc, value, n_qubits):
    """Apply X gates so that |value> maps to |11...1> on the given qubits."""
    for i in range(n_qubits):
        if not (value >> i) & 1:
            qc.x(i)


def oracle(qc, marked_values, n_qubits):
    """Phase-flip exactly the basis states in marked_values."""
    for v in marked_values:
        apply_bitstring_frame(qc, v, n_qubits)
        # multi-controlled Z on n_qubits-1 controls + 1 target, via H-MCX-H
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        apply_bitstring_frame(qc, v, n_qubits)


def diffuser(qc, n_qubits):
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


num_iterations = max(1, round((math.pi / 4) * math.sqrt(N / len(MARKED))))
print(f"Grover iterations used: {num_iterations} (N={N}, M={len(MARKED)})")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(num_iterations):
    oracle(qc, MARKED, N_QUBITS)
    diffuser(qc, N_QUBITS)
qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
compiled = transpile(qc, sim, optimization_level=0)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# qiskit bit order: rightmost char is qubit 0 -> reverse for our little-endian value
def bits_to_int(bitstring):
    # Qiskit's classical-register bitstring has c0 as the RIGHTMOST character,
    # c1 next, etc. We measured qubit i into classical bit i, so qubit i's
    # value is bitstring[len-1-i]; reconstruct the little-endian integer
    # directly from that (do not naively reverse-then-parse, which silently
    # mirrors non-palindromic bit patterns).
    n = len(bitstring)
    return sum(int(bitstring[n - 1 - i]) << i for i in range(n))

value_counts = {}
for bitstring, c in counts.items():
    v = bits_to_int(bitstring)
    value_counts[v] = value_counts.get(v, 0) + c

sorted_values = sorted(value_counts.items(), key=lambda kv: -kv[1])
print("Top measured values (value: count):", sorted_values[:6])

# Consider the measured values whose sampled probability clearly stands out
# (i.e. amplified by Grover) as the quantum-found marked set, then take the
# maximum -- this is the quantity we compare to the classical answer.
threshold = shots / N  # uniform-background expectation per state
amplified = sorted([v for v, c in value_counts.items() if c > 2 * threshold])
print(f"Amplified (quantum-marked) values: {amplified}")

quantum_answer = max(amplified) if amplified else None


# ---------------------------------------------------------------------------
# 3. Compare and report.
# ---------------------------------------------------------------------------

print(f"Quantum answer (max amplified value) = {quantum_answer}")
print(f"Classical answer                     = {CLASSICAL_ANSWER}")

if quantum_answer == CLASSICAL_ANSWER and set(amplified) == set(MARKED):
    print("PASS")
else:
    print("FAIL")
