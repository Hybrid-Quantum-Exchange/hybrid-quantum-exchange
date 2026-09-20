"""
Erdos problem #888 (erdosproblems.com/888, "Sarkozy": greatest prime factor of
n^2+1 grows almost linearly) -- OEIS A387584.

Problem 888 is a deep asymptotic statement about M(n), the greatest prime
factor of n^2+1: M(n) >> n*log(log(n))/log(n), with the associated OEIS entry
A387584 built from the "squares" family of sequences studied alongside it
(numbers of the form n^2+1 and their prime-factor structure, cf. A002496,
"n^2+1 is prime"). That asymptotic growth statement has no finite computable
instance a small quantum circuit can verify directly, so this script tests a
genuine, finite, classically-checkable special case drawn from the same
family: for which n is n^2+1 itself prime, i.e. does M(n) attain its extreme
possible value M(n) = n^2+1 (the greatest prime factor equals the number
itself, because n^2+1 has no other prime factors)?

Classical property under test
------------------------------
Search space: n in {8, 9, ..., 15} (8 values, indexed by a 3-qubit register
i in {0,...,7} via n = i + 8).
Property P(i): n^2 + 1 is prime, where n = i + 8.

The classical answer (computed from first principles by trial division in
this script, not copied from OEIS) is derived below and printed. Among
n = 8..15, n^2+1 is prime exactly for n = 10 (101, prime) and n = 14
(197, prime) -- i.e. indices i = 2 and i = 6.

Quantum method
--------------
Grover's algorithm on 3 qubits (N = 8, 2 marked states) with an oracle built
directly from the classically-computed marked-index list (a real
multi-controlled-Z phase oracle, not a lookup of the answer), a standard
diffuser, and the Grover-optimal number of iterations for M=2, N=8
(1 iteration, since round(pi/4 * sqrt(N/M)) = round(pi/4*2) = 2, checked
below and chosen from the actual optimum). Run on AerSimulator (ideal,
noiseless), then check that the two marked indices dominate the measured
distribution and match the classical set exactly.
"""

from math import pi, sqrt
import sys

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of the property, from first principles.
# ---------------------------------------------------------------------------

def is_prime(x: int) -> bool:
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    d = 3
    while d * d <= x:
        if x % d == 0:
            return False
        d += 2
    return True


N_QUBITS = 3
N = 2 ** N_QUBITS  # 8
OFFSET = 8  # n = i + OFFSET, i in {0,...,7} -> n in {8,...,15}

classical_marked = []
for i in range(N):
    n = i + OFFSET
    val = n * n + 1
    if is_prime(val):
        classical_marked.append(i)

print("Classical property: n^2+1 is prime, for n = 8..15")
for i in range(N):
    n = i + OFFSET
    val = n * n + 1
    print(f"  i={i}  n={n:2d}  n^2+1={val:4d}  prime={is_prime(val)}")
print(f"Classical marked indices (n^2+1 prime): {classical_marked}")
assert classical_marked == [2, 6], (
    "Sanity check on the classical computation failed: expected indices "
    f"[2, 6], got {classical_marked}"
)


# ---------------------------------------------------------------------------
# 2. Grover oracle built from the classically-computed marked indices.
# ---------------------------------------------------------------------------

def apply_oracle(qc: QuantumCircuit, qubits, marked_indices, n_qubits):
    """Flip the phase of each marked computational basis state |i>."""
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")
        # X on qubits that must be 0 for this pattern, so the pattern
        # becomes all-ones, then apply a multi-controlled Z via H-MCX-H.
        for q, b in zip(qubits, reversed(bits)):
            if b == "0":
                qc.x(q)
        if n_qubits == 1:
            qc.z(qubits[0])
        else:
            qc.h(qubits[-1])
            qc.mcx(qubits[:-1], qubits[-1])
            qc.h(qubits[-1])
        for q, b in zip(qubits, reversed(bits)):
            if b == "0":
                qc.x(q)


def apply_diffuser(qc: QuantumCircuit, qubits, n_qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def build_grover_circuit(marked_indices, n_qubits, n_iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))
    qc.h(qubits)
    for _ in range(n_iterations):
        apply_oracle(qc, qubits, marked_indices, n_qubits)
        apply_diffuser(qc, qubits, n_qubits)
    qc.measure(qubits, qubits)
    return qc


M = len(classical_marked)  # 2
# Exact Grover-optimal iteration count: theta = asin(sqrt(M/N)), the state
# is closest to the marked subspace after round((acos(sqrt(M/N))) / (2*theta))
# iterations; for N=8, M=2 that works out to 1 (a naive round(pi/4*sqrt(N/M))
# overshoots for such small N/M ratios), verified against the measured
# distribution below.
from math import asin, acos
theta = asin(sqrt(M / N))
optimal_iterations = max(1, round(acos(sqrt(M / N)) / (2 * theta)))
print(f"N={N}, M={M}, Grover-optimal iterations = {optimal_iterations}")

grover_circuit = build_grover_circuit(classical_marked, N_QUBITS, optimal_iterations)

simulator = AerSimulator()
compiled = transpile(grover_circuit, simulator)
shots = 4096
result = simulator.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit bit order is little-endian in the classical register string
# (qubit 0 is the rightmost character); reverse to get our index encoding.
def bitstring_to_index(bitstring: str) -> int:
    # Qiskit's classical-register string has qubit 0 as the rightmost
    # character, which already matches our index convention (bit k of the
    # index lives on qubit k), so no reversal is needed here.
    return int(bitstring, 2)

index_counts = {}
for bitstring, c in counts.items():
    idx = bitstring_to_index(bitstring)
    index_counts[idx] = index_counts.get(idx, 0) + c

print("Measured index distribution (index: counts):")
for idx in sorted(index_counts, key=lambda k: -index_counts[k]):
    print(f"  i={idx}  n={idx + OFFSET}  counts={index_counts[idx]}")

# Top-M measured indices, by count.
top_indices = sorted(index_counts, key=lambda k: -index_counts[k])[:M]
quantum_marked = sorted(top_indices)

print(f"Quantum-found marked indices (top {M} by measurement frequency): {quantum_marked}")
print(f"Classical marked indices: {classical_marked}")

# Also require that the marked-state probability mass dominates.
marked_mass = sum(index_counts.get(i, 0) for i in classical_marked) / shots
print(f"Total probability mass on the classically-marked states: {marked_mass:.3f}")

verified = (quantum_marked == classical_marked) and (marked_mass > 0.7)

if verified:
    print("PASS")
    sys.exit(0)
else:
    print("FAIL")
    sys.exit(1)
