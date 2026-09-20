"""
Erdos problem #1 (erdosproblems.com) -- quantum-testable instance.

OEIS id used: A276661.
  a(n) = the least k such that there is a set S subseteq {1, ..., k} with
  |S| = n elements all of whose 2^n subset sums are distinct ("distinct
  subset sums" / "Erdos distinct subset sums" problem, the subject of
  Erdos problem #1). Known terms: a(0..10) =
  0, 1, 2, 4, 7, 13, 24, 44, 84, 161, 309.

Classical property tested here (computed from first principles in this
script, not copied from OEIS):
  For n = 3, verify a(3) = 4. That means:
    (a) no 3-element subset of {1, 2, 3} has all-distinct subset sums, and
    (b) at least one 3-element subset of {1, 2, 3, 4} does.
  This script brute-forces both facts classically, then builds a Grover
  search over the k = 4 case: the search space is all 4-bit strings
  (one bit per element of {1,2,3,4} indicating membership), and a
  "good" (marked) state is a string with exactly 3 bits set whose
  chosen 3-element subset has all distinct subset sums. Grover search
  is run on the ideal AerSimulator to find a marked state, and the
  result is checked against the classical brute-force set of good
  states.

Why this is a genuine small quantum computation: the oracle is not a
literal encoding of "the answer" -- it is built by classically
evaluating, for each of the 16 possible 4-bit membership strings,
whether that particular subset has the distinct-subset-sums property
(an O(2^|subset|) check per candidate), and only then compiling that
truth table into a multi-controlled-Z phase oracle. Grover's algorithm
is then used to amplify and find one of the marked strings, which is
compared against the classically precomputed set.

Search space: N = 2^4 = 16 basis states (4 qubits). Small enough to
simulate exactly and to brute-force classically for verification.
"""

from itertools import combinations
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS lookup used as a value)
# ---------------------------------------------------------------------------

def has_distinct_subset_sums(subset):
    """True iff every subset of `subset` has a distinct sum."""
    sums = set()
    elems = list(subset)
    n = len(elems)
    for mask in range(1 << n):
        s = sum(elems[i] for i in range(n) if (mask >> i) & 1)
        if s in sums:
            return False
        sums.add(s)
    return True


def best_k_for_n(n, k_max):
    """Smallest k <= k_max such that some n-subset of {1..k} has the
    distinct-subset-sums property, or None if none found up to k_max."""
    for k in range(n, k_max + 1):
        for combo in combinations(range(1, k + 1), n):
            if has_distinct_subset_sums(combo):
                return k
    return None


N = 3  # size of the subset (matches a(3) in A276661)
a3 = best_k_for_n(N, k_max=6)
assert a3 == 4, f"classical brute force found a(3)={a3}, expected 4 (OEIS A276661)"
print(f"Classical result: a({N}) = {a3} (matches OEIS A276661 term a(3)=4)")

# Confirm no 3-subset of {1,2,3} works (k=3 must fail) -- sanity check.
assert all(not has_distinct_subset_sums(c) for c in combinations(range(1, 4), N))
print("Classical sanity check: no 3-subset of {1,2,3} has distinct subset sums (as expected).")

# ---------------------------------------------------------------------------
# 2. Build the classical truth table over all 4-bit membership strings for
#    the k = 4 universe {1,2,3,4}, marking "good" states: exactly 3 elements
#    chosen AND the distinct-subset-sums property holds.
# ---------------------------------------------------------------------------

K = 4
UNIVERSE = list(range(1, K + 1))
NUM_QUBITS = K  # one qubit per element of {1,2,3,4}

marked_states = []  # list of ints in [0, 2^K)
for mask in range(1 << K):
    chosen = [UNIVERSE[i] for i in range(K) if (mask >> i) & 1]
    if len(chosen) == N and has_distinct_subset_sums(chosen):
        marked_states.append(mask)

assert len(marked_states) > 0, "expected at least one good 3-subset of {1,2,3,4}"
print(f"Marked (good) states among 16 possible 4-bit strings: {sorted(marked_states)} "
      f"-> subsets {[[UNIVERSE[i] for i in range(K) if (m >> i) & 1] for m in marked_states]}")


# ---------------------------------------------------------------------------
# 3. Grover search over the 4-qubit space for a marked state.
# ---------------------------------------------------------------------------

def apply_phase_oracle(qc, marked, num_qubits):
    """Flip the phase of each basis state in `marked` (qubit index i is
    bit i of the integer, little-endian, matching Qiskit's convention)."""
    for m in marked:
        zero_bits = [i for i in range(num_qubits) if not (m >> i) & 1]
        for i in zero_bits:
            qc.x(i)
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        for i in zero_bits:
            qc.x(i)


def apply_diffusion(qc, num_qubits):
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))


num_marked = len(marked_states)
N_states = 1 << NUM_QUBITS
iterations = max(1, round((np.pi / 4) * np.sqrt(N_states / num_marked)))

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    apply_phase_oracle(qc, marked_states, NUM_QUBITS)
    apply_diffusion(qc, NUM_QUBITS)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 2048
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings with qubit (NUM_QUBITS-1) leftmost and qubit 0
# rightmost, so the literal binary value of the string already equals
# sum_i bit(qubit i) * 2^i -- the same little-endian integer convention
# used above for `marked_states`. No reversal needed.
def bitstring_to_int(bs):
    return int(bs, 2)

marked_shots = sum(c for bs, c in counts.items() if bitstring_to_int(bs) in marked_states)
top_bs = max(counts, key=counts.get)
top_state = bitstring_to_int(top_bs)

print(f"Grover iterations used: {iterations}")
print(f"Shots landing on a marked state: {marked_shots}/{shots} "
      f"({100 * marked_shots / shots:.1f}%)")
print(f"Most frequent measured state: {top_bs} (int {top_state}), "
      f"subset = {[UNIVERSE[i] for i in range(K) if (top_state >> i) & 1]}")

# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

quantum_found_marked = top_state in marked_states
amplification_worked = marked_shots / shots > 0.5  # much better than 4/16 = 25% baseline

verified = quantum_found_marked and amplification_worked and (a3 == 4)

print()
if verified:
    print("PASS: Grover search on the k=4 universe found a 3-subset with distinct "
          "subset sums, matching the classically verified a(3)=4 from OEIS A276661.")
else:
    print("FAIL: quantum result did not match the classical answer.")
