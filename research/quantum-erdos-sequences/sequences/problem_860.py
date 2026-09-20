"""
Erdos problem #860 (Erdos-Pomerance distinct-multiples problem, primes case)
https://www.erdosproblems.com/860

OEIS sequence used: A058989.
  A058989(n) = the largest number L such that there exist L consecutive
  integers each of which is divisible by some prime p <= p_n (the n-th
  prime). Equivalently: the longest run of consecutive integers that is
  "covered" by the prime set {p_1, ..., p_n}, where an integer is covered
  if it has a prime factor <= p_n. This is the same combinatorial quantity
  studied by Jacobsthal's function (A048670 is essentially the Jacobsthal
  function of the n-th primorial), which is exactly what problem #860 is
  about: how long a run of consecutive integers can avoid being coprime to
  the primorial P_n = p_1*p_2*...*p_n.

  Known OEIS values: A058989 = 1, 3, 5, 9, 13, 21, 25, ...
  so A058989(1) = 1, A058989(2) = 3.

Classical property tested here (computed from first principles below, not
copied from OEIS):
  For n = 2, the relevant prime set is {2, 3} (the first two primes).
  We search over all starting positions s in a small finite window
  W = {1, ..., 8} for the LONGEST run of L = 3 consecutive integers
  s, s+1, s+2 that are ALL divisible by 2 or 3 (i.e. covered by {2, 3}).
  This is exactly the defining search for A058989(2).

  We first verify classically (by brute force over the whole window) that:
    (a) L = 3 is achievable (matching the known OEIS value A058989(2) = 3),
    (b) L = 4 is NOT achievable anywhere in the window,
  so 3 is truly the maximum run length in this instance, and we record the
  full classical set of starting positions s for which the run of length 3
  works.

Quantum circuit:
  We then build a genuine Grover search circuit over the 3-qubit space
  s in {0, ..., 7} (representing starting positions 1..8, with s_qubit + 1
  = actual starting integer) whose oracle marks exactly the starting
  positions found classically above (i.e. the set of s such that s+1, s+2,
  s+3 are all covered by {2,3}). The oracle is built as a multi-controlled-Z
  "mark these specific classically-verified basis states" oracle (the
  standard technique for Grover search over an explicitly known marked
  set), followed by the standard Grover diffusion operator, run for the
  optimal number of iterations for this instance. We run it on the ideal
  AerSimulator and check that the most probable measured outcome(s) are
  exactly the classically-computed marked starting positions -- i.e. that
  Grover search successfully recovers a starting position realizing
  A058989(2) = 3.

PASS/FAIL: the script prints PASS if the AerSimulator's most-likely
measured basis state(s) are exactly the classically verified marked set of
starting positions for the run-of-3 property; else it prints FAIL.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# 1. Classical computation (first principles): find longest run of
#    consecutive integers in a small window, each divisible by 2 or 3.
# ---------------------------------------------------------------------

def covered(x, primes):
    """True if x has a prime factor in `primes` (x divisible by some p in primes)."""
    return any(x % p == 0 for p in primes)


PRIMES_N2 = [2, 3]          # first n = 2 primes
WINDOW = list(range(1, 9))  # search window {1, ..., 8}, fits in 3 qubits (s = 0..7)

# Brute-force classical search over all possible run lengths and all
# starting positions in the window.
best_len = 0
for start in WINDOW:
    length = 0
    x = start
    while x in WINDOW and covered(x, PRIMES_N2):
        length += 1
        x += 1
    best_len = max(best_len, length)

assert best_len == 3, f"expected classical A058989(2) = 3, got {best_len}"

TARGET_LEN = 3

# Classical set of starting positions (1-indexed integers) s such that
# s, s+1, s+2 are all covered by {2,3}, restricted to the search window
# used for the quantum circuit (starting positions 1..6 so that s+2 <= 8).
marked_starts = []
for start in range(1, 7):
    if all(covered(start + k, PRIMES_N2) for k in range(TARGET_LEN)):
        marked_starts.append(start)

assert marked_starts, "no classical starting position found for run length 3"
# Sanity: confirm no length-4 run exists anywhere in the window (so 3 is
# really the max, matching A058989(2) = 3).
for start in range(1, 6):
    assert not all(covered(start + k, PRIMES_N2) for k in range(4)), (
        "found an unexpected run of length 4 -- contradicts A058989(2) = 3"
    )

print("Classical brute-force result:")
print(f"  primes used (first 2 primes): {PRIMES_N2}")
print(f"  window: {WINDOW}")
print(f"  longest covered run found: {best_len} (matches OEIS A058989(2) = 3)")
print(f"  starting positions (1-indexed) giving a run of length 3: {marked_starts}")

# Encode starting positions as 3-qubit basis states s_qubit = start - 1,
# i.e. s_qubit in {0, ..., 7} represents starting integer start = s_qubit+1.
marked_qubit_values = sorted(s - 1 for s in marked_starts)
N_QUBITS = 3
N_STATES = 2 ** N_QUBITS  # 8

print(f"  marked basis states (0-indexed, for the quantum search): {marked_qubit_values}")


# ---------------------------------------------------------------------
# 2. Quantum circuit: Grover search over the 3-qubit space for the
#    classically-verified marked starting positions.
# ---------------------------------------------------------------------

def multi_controlled_z_on_state(qc, qubits, state_bits):
    """Apply a phase flip (-1) to the single computational basis state
    given by state_bits (tuple of 0/1, little-endian matching `qubits`
    order) using X gates to map that state to |11...1> and a
    multi-controlled-Z (built from H + MCX + H on the last qubit)."""
    flip_qubits = [q for q, b in zip(qubits, state_bits) if b == 0]
    for q in flip_qubits:
        qc.x(q)

    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])

    for q in flip_qubits:
        qc.x(q)


def build_oracle(n_qubits, marked_values):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for value in marked_values:
        bits = tuple((value >> i) & 1 for i in range(n_qubits))  # little-endian
        multi_controlled_z_on_state(qc, list(range(n_qubits)), bits)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


n_marked = len(marked_qubit_values)
# Optimal number of Grover iterations for N_STATES items, n_marked marked.
theta = math.asin(math.sqrt(n_marked / N_STATES))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

oracle = build_oracle(N_QUBITS, marked_qubit_values)
diffuser = build_diffuser(N_QUBITS)

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

print(f"\nGrover circuit: {N_QUBITS} qubits, {n_marked} marked state(s) out of {N_STATES}, "
      f"{iterations} iteration(s)")

# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 4096
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB-first ("c2 c1 c0"); convert to the
# little-endian integer value used above.
def bitstring_to_value(bs):
    # Qiskit's classical-register bitstrings are already written with the
    # lowest-index bit (qubit 0, our LSB) as the rightmost character, so a
    # plain binary parse recovers the little-endian integer value.
    return int(bs, 2)

value_counts = {}
for bitstring, c in counts.items():
    v = bitstring_to_value(bitstring)
    value_counts[v] = value_counts.get(v, 0) + c

max_count = max(value_counts.values())
most_likely_values = sorted(v for v, c in value_counts.items() if c == max_count)

print(f"\nMeasurement counts (by 0-indexed starting value): {dict(sorted(value_counts.items()))}")
print(f"Most likely measured value(s): {most_likely_values}")
print(f"Classically expected marked value(s): {marked_qubit_values}")

# ---------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------

verified = set(most_likely_values) == set(marked_qubit_values)

if verified:
    print("\nPASS: Grover search recovered exactly the classically verified "
          "starting position(s) realizing A058989(2) = 3.")
else:
    print("\nFAIL: Grover search result does not match the classical answer.")

print(f"\nran_ok=True verified_against_classical={verified}")
