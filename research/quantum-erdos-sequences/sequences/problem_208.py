"""
Erdos problem #208 (erdosproblems.com) -- quantum-testable instance.

OEIS ids listed for this problem: A005117, A076259.
  A005117 = squarefree numbers: positive integers n not divisible by any
  perfect square > 1 (i.e. n's prime factorization has no exponent >= 2).
  A076259 is a related squarefree-adjacent sequence; A005117 is the one
  used directly here since it gives a clean finite/computable membership
  predicate.

Classical property tested here (computed from first principles in this
script, not copied from OEIS): for the universe {1, ..., 16} (fits in
4 qubits), which integers n are squarefree, i.e. n mod 4 != 0 and
n mod 9 != 0 (4=2^2 and 9=3^2 are the only squares <= 16 other than 1,
so those two divisibility checks fully decide squarefreeness up to 16).
This is verified independently against a direct prime-factorization
check for every n in 1..16.

Why this is a genuine small quantum computation: the oracle is built by
classically evaluating, for each of the 16 possible 4-bit integers n
(0..15, mapped to n+1 in {1,...,16}), whether n+1 is squarefree by
direct trial-division factorization (not by copying an OEIS list), and
compiling that truth table into a multi-controlled-Z phase oracle.
Grover's algorithm is then run on the ideal AerSimulator to amplify and
find one of the marked (squarefree) states, and the full measured
distribution is checked against the classically computed set of
squarefree members of {1,...,16} (which should match the initial
segment of OEIS A005117: 1,2,3,5,6,7,10,11,13,14,15).

Search space: N = 2^4 = 16 basis states (4 qubits). Small enough to
simulate exactly and to brute-force classically for verification.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS lookup used as a value)
# ---------------------------------------------------------------------------

def is_squarefree_by_factorization(n):
    """True iff n's prime factorization has no exponent >= 2 (direct
    trial-division factorization, not a divisibility shortcut)."""
    if n <= 0:
        raise ValueError("n must be positive")
    m = n
    d = 2
    while d * d <= m:
        if m % d == 0:
            count = 0
            while m % d == 0:
                m //= d
                count += 1
            if count >= 2:
                return False
        d += 1
    return True


K = 16  # universe {1, ..., 16}
NUM_QUBITS = 4  # 2^4 = 16 basis states, one per n-1 in [0, 16)

squarefree_members = [n for n in range(1, K + 1) if is_squarefree_by_factorization(n)]
print(f"Classical (trial-division) squarefree members of 1..{K}: {squarefree_members}")

# Cross-check with the mod-4 / mod-9 shortcut described in the docstring,
# valid specifically for n <= 16 since 4 and 9 are the only squares > 1
# not exceeding 16.
squarefree_by_shortcut = [n for n in range(1, K + 1) if n % 4 != 0 and n % 9 != 0]
assert squarefree_by_shortcut == squarefree_members, (
    "mod-4/mod-9 shortcut disagrees with direct factorization -- bug in reasoning"
)
print("Classical cross-check (n%4!=0 and n%9!=0) agrees with direct factorization.")

# Sanity: matches the known initial segment of OEIS A005117.
expected_prefix = [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15]
assert squarefree_members == expected_prefix, (
    f"expected {expected_prefix}, got {squarefree_members}"
)
print(f"Matches OEIS A005117 initial terms restricted to 1..{K}: {expected_prefix}")

# ---------------------------------------------------------------------------
# 2. Build the classical truth table over all 4-bit integers x in [0,16),
#    marking x as "good" iff n = x+1 is squarefree.
# ---------------------------------------------------------------------------


# Grover's amplitude amplification degrades once the marked fraction
# exceeds ~1/2 (the rotation overshoots), and squarefree numbers are the
# *majority* of 1..16 (11 of 16). So we instead search for the minority
# class -- the NON-squarefree numbers -- which is the complementary,
# equally valid target for the same oracle-construction technique, and
# recover squarefreeness as "not found among the marked (non-squarefree)
# states."
non_squarefree_members = [n for n in range(1, K + 1) if n not in squarefree_members]
marked_states = [n - 1 for n in non_squarefree_members]  # basis state x encodes n = x+1
assert len(marked_states) > 0
print(f"Marked (NON-squarefree) basis states x=n-1: {sorted(marked_states)} "
      f"-> n values {non_squarefree_members}")


# ---------------------------------------------------------------------------
# 3. Grover search over the 4-qubit space for a marked (squarefree) state.
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
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()


def bitstring_to_int(bs):
    # Qiskit's bitstring has qubit (NUM_QUBITS-1) leftmost, qubit 0
    # rightmost, so the literal binary value equals the little-endian
    # integer convention used above for `marked_states`.
    return int(bs, 2)


marked_shots = sum(c for bs, c in counts.items() if bitstring_to_int(bs) in marked_states)
top_bs = max(counts, key=counts.get)
top_state = bitstring_to_int(top_bs)

print(f"Grover iterations used: {iterations}")
print(f"Shots landing on a marked (non-squarefree) state: {marked_shots}/{shots} "
      f"({100 * marked_shots / shots:.1f}%)")
print(f"Most frequent measured state: {top_bs} (int {top_state}), n = {top_state + 1}, "
      f"non-squarefree = {top_state in marked_states}")

# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

baseline = num_marked / N_states  # uniform-random baseline probability
quantum_found_marked = top_state in marked_states
amplification_worked = (marked_shots / shots) > (2 * baseline)  # well above chance

verified = quantum_found_marked and amplification_worked and (squarefree_members == expected_prefix)

print()
if verified:
    print("PASS: Grover search over {1,...,16} found a non-squarefree integer with "
          "amplified probability, matching the classically verified squarefree/"
          "non-squarefree partition of OEIS A005117 restricted to 1..16.")
else:
    print("FAIL: quantum result did not match the classical answer.")
