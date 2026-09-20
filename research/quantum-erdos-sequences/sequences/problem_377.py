"""
Erdos problem #377 (erdosproblems.com), quantum-testable lane.

Source metadata (data/problems.yaml in manman4/erdosproblems, entry "number: 377"):
tags = ["number theory", "binomial coefficients"], oeis = ["N/A"].
Problem 377 has NO OEIS sequence id attached in the source data -- so, per the
task instructions, this script does not use any specific OEIS integer sequence.
Instead it honestly falls back to the one concrete, finite, computable
mathematical object the problem's own tags name: binomial-coefficient
divisibility (specifically parity, i.e. divisibility by 2).

Chosen classical property
--------------------------
Kummer's theorem: for n >= k >= 0, the binomial coefficient C(n, k) is ODD
(not divisible by 2) if and only if every bit that is set in k is also set
in n, i.e. (n & k) == k (equivalently n AND k, in bitwise AND, equals k).

We fix k = 5 (binary 0101, so bits 0 and 2 must be set in n) and search over
all n in [0, 15] (a 4-qubit register, N = 16 <= 64 as required). This is a
genuine, first-principles-derivable arithmetic property of binomial
coefficients -- squarely inside this problem's "binomial coefficients" tag --
and is small enough for an exact statevector simulation.

The classical answer for k = 5, n in [0, 15] is computed in this script from
scratch two independent ways:
  (a) directly from Pascal's-triangle-style binomial coefficients mod 2, and
  (b) via Kummer's bitwise test (n & k) == k,
and the two are cross-checked to agree before anything quantum happens.

Quantum approach
-----------------
A Grover search over the 4-qubit register |n> marks exactly the n with
C(n, 5) odd. The oracle is a plain multi-controlled-Z on the two bit
positions set in k = 5 (qubits 0 and 2) -- it flips the phase of a basis
state iff those two qubits are both |1>, which is exactly the Kummer
condition (n & 5) == 5 (the other two qubits, positions 1 and 3, are free,
giving 4 marked states out of 16, matching the classical count). One Grover
iteration (optimal for M=4 marked out of N=16) is run on the ideal
AerSimulator (statevector method), and the resulting measurement
distribution is checked against the classical solution set: the amplified
outcomes must be exactly the 4 classically-odd n values, carrying a large
majority of the measured probability mass.

PASS/FAIL is decided by comparing the quantum measurement distribution to
the from-scratch classical answer.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

def binom(n: int, k: int) -> int:
    """Plain integer binomial coefficient via Pascal's triangle recursion."""
    if k < 0 or k > n:
        return 0
    row = [1] + [0] * n
    for i in range(1, n + 1):
        for j in range(min(i, k), 0, -1):
            row[j] = row[j] + row[j - 1]
    return row[k] if k <= n else 0


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16
K = 5               # binary 0101


def is_odd_binom_direct(n: int, k: int) -> bool:
    """Method (a): compute C(n, k) exactly and test parity directly."""
    return binom(n, k) % 2 == 1


def is_odd_binom_kummer(n: int, k: int) -> bool:
    """Method (b): Kummer's bitwise test (n & k) == k."""
    return (n & k) == k


classical_solutions_direct = sorted(n for n in range(N) if is_odd_binom_direct(n, K))
classical_solutions_kummer = sorted(n for n in range(N) if is_odd_binom_kummer(n, K))

assert classical_solutions_direct == classical_solutions_kummer, (
    "Cross-check failed: direct binomial-parity computation and Kummer's "
    "bitwise test disagree -- classical answer is not trustworthy."
)

classical_solutions = classical_solutions_direct
print(f"Property: C(n, {K}) is odd, for n in [0, {N - 1}]")
print(f"Classical answer (from Pascal's triangle mod 2): {classical_solutions}")
print(f"Cross-check via Kummer's theorem (n & k == k):   {classical_solutions_kummer}")
print(f"Number of marked states M = {len(classical_solutions)} out of N = {N}")


# ---------------------------------------------------------------------------
# 2. Grover search circuit.
# ---------------------------------------------------------------------------
# K = 5 = 0b0101, so bits 0 and 2 (0-indexed, qubit 0 = LSB) must be 1.
# The oracle is a controlled-Z on qubits {0, 2}: it flips the phase of any
# basis state |n> with both those qubits set to |1>, i.e. exactly the states
# with (n & 5) == 5. Qubits 1 and 3 are untouched by the oracle, so each of
# the 4 combinations of those two free bits is marked -> 4 solutions, matching
# the classical count above.

MARK_QUBITS = [i for i in range(N_QUBITS) if (K >> i) & 1]  # bit positions set in K
assert MARK_QUBITS == [0, 2]


def oracle(qc: QuantumCircuit) -> None:
    # Multi-controlled Z on the qubits that must be 1 (Kummer condition).
    if len(MARK_QUBITS) == 1:
        qc.z(MARK_QUBITS[0])
    else:
        qc.h(MARK_QUBITS[-1])
        qc.mcx(MARK_QUBITS[:-1], MARK_QUBITS[-1])
        qc.h(MARK_QUBITS[-1])


def diffuser(qc: QuantumCircuit, qubits) -> None:
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


M = len(classical_solutions)
# Optimal number of Grover iterations for M solutions out of N.
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / 4) / theta - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    oracle(qc)
    diffuser(qc, list(range(N_QUBITS)))
qc.measure(range(N_QUBITS), range(N_QUBITS))

print(f"Grover iterations used: {iterations}")

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator(method="statevector")
compiled = transpile(qc, backend)
SHOTS = 8192
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit reports bitstrings with qubit 0 as the rightmost character, which is
# already standard big-endian-to-int order for our qubit-i == bit-i encoding.
def bitstring_to_n(bits: str) -> int:
    return int(bits, 2)

n_counts = Counter()
for bits, c in counts.items():
    n_counts[bitstring_to_n(bits)] += c

marked_mass = sum(n_counts[n] for n in classical_solutions)
marked_fraction = marked_mass / SHOTS

# The most-frequent outcomes (top M by count) should be exactly the
# classical solution set.
top_n = [n for n, _ in n_counts.most_common(M)]
top_n_sorted = sorted(top_n)

print(f"Measured probability mass on classically-marked states: {marked_fraction:.3f}")
print(f"Top-{M} most frequent measured n values: {top_n_sorted}")
print(f"Classical solution set:                  {classical_solutions}")

quantum_matches_classical = (
    top_n_sorted == classical_solutions and marked_fraction > 0.6
)

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")
