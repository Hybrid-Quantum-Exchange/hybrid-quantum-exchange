"""
Erdos problem #854 -- quantum-testable instance.

Erdos problem 854 (erdosproblems.com/854) is tagged "number theory" and its
metadata lists OEIS ids A389839 and A048670. A048670 is the well documented
sequence relevant here: a(n) = J(P(n)), the Jacobsthal function of the n-th
primorial P(n) = product of the first n primes. (A389839 is a companion/
newer sequence on the same underlying problem -- the Jacobsthal function of
primorials -- and is not needed to state the finite computable property
below.)

The Jacobsthal function g(m) of an integer m > 1 is the smallest L such that
every L consecutive integers contain at least one integer coprime to m.
Equivalently, g(m) - 1 is the length of the longest run of *consecutive*
integers that are ALL non-coprime to m (i.e. each one shares a prime factor
with m). Because "coprime to m" only depends on the residue mod m, this run
can always be found inside one period {0, 1, ..., m-1} (cyclically).

Finite computable property tested here (derived from A048670, not copied
from the OEIS b-file):

    For m = 2*3*5 = 30 (the 3rd primorial, so this instance is A048670(3)),
    find every residue s in Z_30 that starts the longest run of consecutive
    non-coprime-to-30 integers, i.e. s such that s, s+1, s+2, s+3, s+4
    (mod 30) are ALL divisible by 2, 3, or 5.

The script first computes this classically from first principles (checking
every residue mod 30, exactly the definition above -- no value is copied
from OEIS). It finds the run starts s = 2 and s = 24 (the same run shape,
{2,3,4,5,6} and {24,...,28}, occurring twice in one period of 30) and
confirms the resulting run length + 1 equals A048670(3) = 6, matching the
known sequence value 2, 4, 6, ... at n=3 (this is only used as a sanity
cross-check, the classical answer is derived independently by brute force).

It then builds a genuine Grover search circuit over the 5-qubit space
{0, ..., 31} (padding 30, 31 as non-solutions) whose oracle marks exactly
the classically-derived solution set, runs it on the ideal AerSimulator, and
checks that the most frequent measured outcome equals the classical answer.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation, from first principles.
# ---------------------------------------------------------------------------

def first_n_primes(n):
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes


def primorial(n):
    p = 1
    for prime in first_n_primes(n):
        p *= prime
    return p


N = 3
M = primorial(N)          # 2*3*5 = 30
assert M == 30

RUN_LEN = 5                # length of the run we search for (candidate g(M)-1)
NUM_QUBITS = 5              # 2**5 = 32 >= M, enough to index every residue mod 30


def is_non_coprime(x, m):
    return math.gcd(x, m) != 1


def classical_jacobsthal(m):
    """Brute-force g(m): smallest L such that every window of L consecutive
    integers contains one coprime to m. Returns (g(m), run_start, run_len)
    where run_start is a starting residue of a maximal non-coprime run of
    length g(m) - 1."""
    best_run = 0
    best_start = None
    for start in range(m):
        length = 0
        while is_non_coprime((start + length) % m, m):
            length += 1
            if length > m:  # m has no integer coprime to it -- degenerate, not our case
                break
        if length > best_run:
            best_run = length
            best_start = start
    return best_run + 1, best_start, best_run


g_m, run_start, run_len = classical_jacobsthal(M)

# Cross-check against the known A048670 value at n=3 without ever trusting
# it blindly -- it must match what we just derived independently.
assert g_m == 6, f"expected A048670(3) = 6, got {g_m}"
assert run_len == RUN_LEN

# The classical solution set for our circuit: residues s in [0, M) such that
# s, s+1, ..., s+RUN_LEN-1 (mod M) are all non-coprime to M.
solutions = []
for s in range(M):
    if all(is_non_coprime((s + i) % M, M) for i in range(RUN_LEN)):
        solutions.append(s)

assert run_start in solutions
CLASSICAL_ANSWERS = sorted(solutions)
print(f"Classical: A048670({N}) = g({M}) = {g_m}; maximal non-coprime runs "
      f"of length {RUN_LEN} start at residues {CLASSICAL_ANSWERS} "
      f"(mod {M}, cyclically -- these are the same run shape wrapping the "
      f"period, e.g. 2..6 and 24..28/29 both consist entirely of multiples "
      f"of 2, 3 or 5)")


# ---------------------------------------------------------------------------
# 2. Grover search circuit over the 5-qubit index space marking the
#    classically-derived solution set CLASSICAL_ANSWERS.
# ---------------------------------------------------------------------------

def marked_bitstring(value, num_qubits):
    return format(value, f"0{num_qubits}b")


def apply_oracle(qc, marked_value, num_qubits):
    """Phase-flip the single computational basis state |marked_value>."""
    bits = marked_bitstring(marked_value, num_qubits)[::-1]  # little-endian per qubit index
    zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
    if zero_qubits:
        qc.x(zero_qubits)
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    if zero_qubits:
        qc.x(zero_qubits)


def apply_diffuser(qc, num_qubits):
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))


N_STATES = 2 ** NUM_QUBITS
NUM_SOLUTIONS = len(CLASSICAL_ANSWERS)
iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / NUM_SOLUTIONS)))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    for marked in CLASSICAL_ANSWERS:
        apply_oracle(qc, marked, NUM_QUBITS)
    apply_diffuser(qc, NUM_QUBITS)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

backend = AerSimulator()
compiled = transpile(qc, backend)
result = backend.run(compiled, shots=2048).result()
counts = result.get_counts()

# Qiskit reports classical bit c_{n-1}...c_0 (big-endian string) matching
# our little-endian qubit-to-bit assignment above, so int(key, 2) recovers
# the integer index directly.
top_bitstrings = sorted(counts, key=counts.get, reverse=True)[:NUM_SOLUTIONS]
quantum_answers = sorted(int(b, 2) for b in top_bitstrings)
top_probability = sum(counts[b] for b in top_bitstrings) / sum(counts.values())

print(f"Quantum: {iterations} Grover iteration(s) over {NUM_QUBITS} qubits "
      f"({N_STATES} states, {NUM_SOLUTIONS} marked), top outcomes = "
      f"{quantum_answers} (combined probability {top_probability:.3f})")

passed = (quantum_answers == CLASSICAL_ANSWERS) and (top_probability > 0.5)

print("PASS" if passed else "FAIL")
if not passed:
    raise SystemExit(1)
