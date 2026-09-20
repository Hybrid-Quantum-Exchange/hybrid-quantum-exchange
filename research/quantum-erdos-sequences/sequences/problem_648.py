"""
Erdos problem #648 -- quantum-testable instance.

OEIS sequence used: A391750.

A391750(n) is defined (Thomas Bloom / erdosproblems.com #648, and the OEIS
entry itself) as the maximum length of a strictly increasing sequence of
integers 2 <= x_1 < x_2 < ... < x_k <= n whose largest-prime-factor (gpf)
values form a strictly DECREASING sequence gpf(x_1) > gpf(x_2) > ... >
gpf(x_k). Equivalently: build the array g[i] = gpf(i) for i = 2..n, in
increasing-index order, and A391750(n) is the length of the longest
strictly decreasing subsequence (LDS) of that array. The OEIS b-data is
1,1,2,2,2,2,3,3,3,3,3,3,3,3,4,... for n = 2,3,4,...; the docstring below
reproduces and checks the first 8 of these terms (n = 2..9) from first
principles inside this script.

Classical property tested here (small, finite, computable):
    For n in 2..9, mark exactly the n for which A391750(n) == 1.
    Classically this set is {2, 3} (the sequence only reaches 1 for the
    very first two terms; for n=4 it jumps to 2, e.g. via 3,2 at
    indices 3,4 since gpf(3)=3 > gpf(4)=2).

We encode n = 2 + k for k in {0,...,7} using 3 qubits (k = q2 q1 q0,
q2 most significant). The classically-marked set {2,3} corresponds to
k in {0,1}, i.e. exactly the computational-basis states with the two
high qubits (q2,q1) both equal to 0 (q0 is free). This is a genuine,
non-trivial, sparse subset (2 of 8 basis states) derived from real
number-theoretic computation, not hard-coded from OEIS -- it is derived
and verified in this script by direct enumeration of decreasing
subsequences of the gpf array.

We then run Grover's algorithm (built as a real Qiskit circuit -- a
diagonal-phase oracle constructed from the classically-computed marked
set, plus the standard Grover diffuser) on the ideal AerSimulator to
search for these marked states among all 8, and check that the states
returned with high probability are exactly {000, 001} (n = 2, 3).

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of A391750(n) for n = 2..9, from first principles.
# ---------------------------------------------------------------------------

def greatest_prime_factor(m: int) -> int:
    """Largest prime factor of m >= 2, by trial division."""
    assert m >= 2
    x = m
    gpf = 1
    d = 2
    while d * d <= x:
        while x % d == 0:
            gpf = d
            x //= d
        d += 1
    if x > 1:
        gpf = x
    return gpf


def a391750_upto(n_max: int):
    """
    Return a dict {n: A391750(n)} for n = 2..n_max, computed as the length
    of the longest strictly decreasing subsequence (LDS) of the array
    g[2], g[3], ..., g[n] where g[i] = greatest_prime_factor(i), restricted
    at each prefix length n (so a(n) only looks at indices 2..n).
    """
    indices = list(range(2, n_max + 1))
    g = [greatest_prime_factor(i) for i in indices]

    # dp[j] = length of the longest strictly decreasing subsequence of
    # g[0..j] that ends exactly at position j.
    dp = [1] * len(g)
    for j in range(len(g)):
        for i in range(j):
            if g[i] > g[j]:
                dp[j] = max(dp[j], dp[i] + 1)

    result = {}
    best_so_far = 0
    for idx, n in enumerate(indices):
        best_so_far = max(best_so_far, dp[idx])
        result[n] = best_so_far
    return result


N_MAX = 9
a_values = a391750_upto(N_MAX)

# Sanity check against the OEIS b-file data for A391750, offset 2:
# 1,1,2,2,2,2,3,3,... for n = 2,3,4,5,6,7,8,9
expected_oeis_prefix = [1, 1, 2, 2, 2, 2, 3, 3]
computed_prefix = [a_values[n] for n in range(2, N_MAX + 1)]
assert computed_prefix == expected_oeis_prefix, (
    f"classical A391750 computation disagrees with OEIS A391750 data: "
    f"{computed_prefix} != {expected_oeis_prefix}"
)

# The property to search for: n in 2..9 such that A391750(n) == 1.
TARGET_VALUE = 1
marked_n = sorted(n for n in range(2, N_MAX + 1) if a_values[n] == TARGET_VALUE)
assert marked_n == [2, 3], f"unexpected marked set {marked_n}"

NUM_QUBITS = 3  # encodes k = n - 2 for n in 2..9 (8 values)
marked_k = sorted(n - 2 for n in marked_n)  # -> [0, 1]


# ---------------------------------------------------------------------------
# 2. Build the Grover circuit: diagonal phase oracle for {marked_k} + diffuser.
# ---------------------------------------------------------------------------

def build_oracle(num_qubits: int, marked_states):
    """Diagonal phase-flip oracle: multiplies amplitude of each marked
    computational basis state (little-endian integer index) by -1."""
    dim = 2 ** num_qubits
    diag = np.ones(dim, dtype=complex)
    for m in marked_states:
        diag[m] = -1.0
    qc = QuantumCircuit(num_qubits, name="Oracle")
    qc.unitary(Operator(np.diag(diag)), range(num_qubits), label="Oracle")
    return qc


def build_diffuser(num_qubits: int):
    qc = QuantumCircuit(num_qubits, name="Diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def grover_iterations(n_items: int, n_marked: int) -> int:
    theta = math.asin(math.sqrt(n_marked / n_items))
    return max(1, round((math.pi / (4 * theta)) - 0.5))


oracle = build_oracle(NUM_QUBITS, marked_k)
diffuser = build_diffuser(NUM_QUBITS)
iterations = grover_iterations(2 ** NUM_QUBITS, len(marked_k))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
shots = 4096
job = backend.run(qc, shots=shots)
counts = job.result().get_counts()

# Qiskit's classical-bit string is big-endian in the printed order but the
# bit at string position -1-i corresponds to qubit i; since we measured
# qubit i -> classical bit i in order, int(bitstring, 2) with the string
# reversed gives the little-endian integer k that matches build_oracle's
# indexing (Operator.diag index = integer with qubit 0 as least
# significant bit, matching Qiskit's default little-endian convention,
# which is exactly how the bitstring already prints qubit0 as the
# rightmost character) -- so int(bitstring, 2) is directly the state index.
observed = {int(bits, 2): freq for bits, freq in counts.items()}

# Keep only outcomes that occur meaningfully more than the uniform-noise
# floor, then take the top len(marked_k) most frequent outcomes.
sorted_outcomes = sorted(observed.items(), key=lambda kv: kv[1], reverse=True)
top_outcomes = sorted(k for k, _ in sorted_outcomes[: len(marked_k)])

marked_fraction = sum(observed.get(k, 0) for k in marked_k) / shots

print(f"Erdos problem #648 / OEIS A391750")
print(f"Classical A391750(n) for n=2..{N_MAX}: {computed_prefix}")
print(f"Marked n with A391750(n) == {TARGET_VALUE}: {marked_n} (k = {marked_k})")
print(f"Grover iterations used: {iterations}")
print(f"Measurement counts (state index -> count): {observed}")
print(f"Top {len(marked_k)} most frequent outcomes: {top_outcomes}")
print(f"Fraction of shots landing on marked states: {marked_fraction:.4f}")

classical_answer = marked_k
quantum_answer = top_outcomes

verified = (quantum_answer == classical_answer) and (marked_fraction > 0.75)

if verified:
    print("PASS")
else:
    print("FAIL")
