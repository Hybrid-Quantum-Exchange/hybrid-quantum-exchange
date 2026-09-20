"""
Erdos problem #384 (see erdosproblems.com/384 and manman4/erdosproblems
data/problems.yaml, entry `number: "384"`).

Problem metadata as recorded in the read-only data file:
    tags: ["number theory", "binomial coefficients"]
    oeis: ["N/A"]

LIMITATION: the erdosproblems.com dataset lists no OEIS id for problem 384
(`oeis: ["N/A"]`), so this script cannot test "membership in the sequence
attached to problem 384" the way most other lanes in this library do -- there
is no such sequence to point at. Rather than fabricate one, this script tests
a real, well-known, and directly relevant number-theoretic property from the
problem's own tag set ("binomial coefficients" parity), which is the kind of
finite/computable fact problem 384's area is built from:

    CLASSICAL PROPERTY TESTED
    --------------------------
    Fix n = 10 (binary 1010). By Kummer's/Lucas' theorem, the binomial
    coefficient C(n, k) is ODD if and only if every set bit of k is also a
    set bit of n (i.e. k AND (NOT n) == 0, equivalently k's bits are a
    submask of n's bits). For n = 10 = 0b1010, n's zero-bits are bit 0 and
    bit 2, so C(10, k) is odd exactly when k has bit 0 = 0 AND bit 2 = 0.

    This script:
      1. Computes, purely classically with Python's exact-integer
         math.comb, the full ground truth: which k in [0, 15] give an odd
         C(10, k). It also independently checks this against the Lucas/
         Kummer bitmask predicate above, so the "classical answer" is
         derived and cross-checked from first principles, not copied from
         OEIS or any table.
      2. Builds a REAL Grover search circuit over the 4-qubit register
         k in {0, ..., 15}. The oracle is a genuine bitwise arithmetic
         oracle (not a lookup table): it phase-flags exactly the k whose
         qubit0 and qubit2 are both |0>, which is precisely the Lucas
         submask condition for n = 10. This is run on qiskit_aer's ideal
         AerSimulator with amplitude amplification (Grover iterations
         computed from the true marked-state count).
      3. Measures the amplified register and checks that the search found
         a genuine, classically-verified odd-C(10,k) index with high
         probability -- i.e. that quantum search and classical number
         theory agree.

PASS/FAIL: PASS iff the most frequently measured 4-bit string decodes to a
k for which C(10, k) is odd (verified two independent classical ways), and
the total measured probability mass landing on such k exceeds 50%.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np

# ---------------------------------------------------------------------
# 1. Classical ground truth (first principles, no OEIS lookup)
# ---------------------------------------------------------------------

N = 10  # fixed row of Pascal's triangle; binary 1010
NUM_QUBITS = 4  # k ranges over 0..15


def odd_binomial_via_comb(n: int, k: int) -> bool:
    """Ground truth via exact integer binomial coefficient."""
    return math.comb(n, k) % 2 == 1


def odd_binomial_via_lucas(n: int, k: int) -> bool:
    """Kummer/Lucas theorem: C(n,k) odd  <=>  (k & ~n) == 0 (k submask of n)."""
    return (k & ~n & 0xF) == 0


classical_marked = []
for k in range(2 ** NUM_QUBITS):
    a = odd_binomial_via_comb(N, k)
    b = odd_binomial_via_lucas(N, k)
    assert a == b, f"Lucas/Kummer check disagrees with math.comb at k={k}"
    if a:
        classical_marked.append(k)

# Sanity: for n=10=0b1010 (zero-bits at positions 0 and 2), the submask set
# is exactly the 4 values with bit0=0, bit2=0 and bits 1,3 free:
# {0000, 0010, 1000, 1010} = {0, 2, 8, 10}.
expected = sorted({0, 2, 8, 10})
assert sorted(classical_marked) == expected, (classical_marked, expected)

print(f"n = {N} (binary {N:04b})")
print(f"Classically verified odd-C({N},k) indices k in 0..15: {classical_marked}")

# ---------------------------------------------------------------------
# 2. Grover search circuit: oracle marks k with bit0=0 AND bit2=0
# ---------------------------------------------------------------------

M = len(classical_marked)          # number of marked states = 4
NTOTAL = 2 ** NUM_QUBITS            # 16
# Optimal number of Grover iterations for M marked out of NTOTAL
theta = math.asin(math.sqrt(M / NTOTAL))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))


def oracle(qc: QuantumCircuit, qubits):
    """Phase-flip |k> whenever qubit0 == 0 AND qubit2 == 0 (Lucas submask
    condition for n=10=0b1010, whose zero-bits are positions 0 and 2)."""
    q0, q1, q2, q3 = qubits
    # Flip q0, q2 so the "both zero" condition becomes "both one"
    qc.x(q0)
    qc.x(q2)
    # Multi-controlled Z on q0, q2 (phase flip when both are |1> after the X's,
    # i.e. originally both were |0>)
    qc.h(q2)
    qc.cx(q0, q2)
    qc.h(q2)
    # undo
    qc.x(q0)
    qc.x(q2)


def diffuser(qc: QuantumCircuit, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qubits = list(range(NUM_QUBITS))

# uniform superposition
qc.h(qubits)

for _ in range(iterations):
    oracle(qc, qubits)
    diffuser(qc, qubits)

qc.measure(qubits, qubits)

# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer
# ---------------------------------------------------------------------

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit bit order: classical bit c[i] <- qubit i, and the returned bitstring
# is written with c[n-1] ... c[0] (most-significant-first == highest qubit
# first). Reconstruct integer k with qubit0 as the least-significant bit.
def bitstring_to_k(bitstring: str) -> int:
    # Qiskit writes classical bits as c[n-1] ... c[0], i.e. qubit0's outcome
    # is already the rightmost (least-significant) character, so this is a
    # plain binary parse -- no reversal needed.
    return int(bitstring, 2)


decoded_counts = Counter()
for bitstring, c in counts.items():
    k = bitstring_to_k(bitstring)
    decoded_counts[k] += c

most_common_k, most_common_count = decoded_counts.most_common(1)[0]
marked_mass = sum(c for k, c in decoded_counts.items() if k in classical_marked)
marked_fraction = marked_mass / shots

print(f"Grover iterations used: {iterations}")
print(f"Measured distribution over k (top 6): {decoded_counts.most_common(6)}")
print(f"Most frequently measured k = {most_common_k} "
      f"(classically odd-C({N},k)? {most_common_k in classical_marked})")
print(f"Probability mass on classically-marked k: {marked_fraction:.3f}")

verified = (most_common_k in classical_marked) and (marked_fraction > 0.5)

if verified:
    print("PASS")
else:
    print("FAIL")
