"""
Erdos problem #140 (erdosproblems.com), status "proved (Lean)".

OEIS id used: A003002 -- the Stanley sequence starting 0,1: the
lexicographically-earliest increasing sequence of nonnegative integers
containing no 3-term arithmetic progression. (0, 1, 3, 4, 9, 10, 12, 13, ...)
Erdos problem #140 concerns 3-term-AP-free sets / arithmetic progressions
(tags: "additive combinatorics", "arithmetic progressions"), and A003002 is
the canonical greedy example of such a set.

Classical fact checked from first principles in this script (not copied from
OEIS): a nonnegative integer m belongs to this greedy AP-free sequence if and
only if its base-3 representation uses only the digits 0 and 1 (no digit 2).
We verify this equivalence classically for all m in [0, 64) by:
  (a) computing the sequence directly via the greedy "no 3-term AP" rule, and
  (b) computing the "no digit 2 in base 3" set independently,
and checking the two sets are identical over that range -- this is the
"small, finite, computable property" used as the oracle criterion below.

Quantum part: Grover's algorithm over 6 qubits (search space 0..63) whose
oracle marks exactly the integers m in [0,64) with no digit 2 in base 3
(equivalently, m in the Stanley/A003002 sequence, per the check above). We
run Grover with the standard number of iterations for this marked-set size
on the ideal AerSimulator, then check that measurement mass concentrates on
the marked set: the most frequent outcomes returned by the circuit are all
true members of the sequence, and disjoint from non-members. This is a real
amplitude-amplification search, not a lookup: the oracle is built gate-by-
gate from the classically-computed marked set, and Grover's diffuser does
the amplification; nothing quantum is faked or pre-decided.

PASS/FAIL: PASS iff every one of the top-K most frequent measured strings
(K = number of marked elements) is indeed in the classically-computed marked
set, and the total probability mass on marked states clearly exceeds the
unmarked baseline.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

N_QUBITS = 6
N = 2 ** N_QUBITS  # 64


# ---------------------------------------------------------------------------
# Classical part: derive the property from first principles.
# ---------------------------------------------------------------------------

def greedy_ap_free_sequence(limit):
    """Lexicographically-earliest increasing sequence starting 0,1 with no
    3-term arithmetic progression, generated greedily up to `limit` (exclusive
    on the search range, but we only need membership within [0, limit))."""
    seq = []
    for candidate in range(limit):
        ok = True
        # check every pair already in seq: does candidate complete a 3-AP?
        # 3-AP conditions with candidate as any of the three positions.
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                a, b = seq[i], seq[j]
                # a, b, candidate is an AP if b - a == candidate - b
                if b - a == candidate - b:
                    ok = False
                    break
                # a, candidate, b is an AP if candidate - a == b - candidate
                if candidate - a == b - candidate:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            seq.append(candidate)
    return seq


def no_digit_2_in_base3(m):
    if m == 0:
        return True
    x = m
    while x > 0:
        if x % 3 == 2:
            return False
        x //= 3
    return True


greedy_seq = greedy_ap_free_sequence(N)
greedy_set = set(greedy_seq)
base3_set = {m for m in range(N) if no_digit_2_in_base3(m)}

assert greedy_set == base3_set, (
    "classical equivalence check failed: greedy AP-free sequence and "
    "'no digit 2 in base 3' sets differ over [0, 64) -- cannot proceed"
)

MARKED = sorted(base3_set)
print(f"Classical: {len(MARKED)} marked values in [0,{N}) "
      f"(A003002 terms in range) = {MARKED}")


# ---------------------------------------------------------------------------
# Quantum part: Grover search whose oracle marks exactly MARKED.
# ---------------------------------------------------------------------------

def build_oracle(n_qubits, marked_values):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


n_marked = len(MARKED)
theta = math.asin(math.sqrt(n_marked / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = build_oracle(N_QUBITS, MARKED)
diffuser = build_diffuser(N_QUBITS)
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
shots = 20000
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit bitstrings are printed as c[n-1] ... c[1] c[0] (leftmost = highest
# clbit index). Since measure(range(6), range(6)) maps qubit i -> clbit i,
# the leftmost character is qubit5 (most significant), so interpreting the
# string directly as a binary integer recovers the measured value.
value_counts = Counter()
for bitstring, c in counts.items():
    value = int(bitstring, 2)
    value_counts[value] += c

top_k = value_counts.most_common(n_marked)
top_k_values = {v for v, _ in top_k}

marked_mass = sum(c for v, c in value_counts.items() if v in base3_set)
unmarked_mass = shots - marked_mass

print(f"Grover iterations used: {iterations}")
print(f"Top-{n_marked} measured values: {sorted(top_k_values)}")
print(f"Probability mass on marked states: {marked_mass}/{shots} "
      f"({100.0 * marked_mass / shots:.1f}%)")
print(f"Probability mass on unmarked states: {unmarked_mass}/{shots} "
      f"({100.0 * unmarked_mass / shots:.1f}%)")

all_top_k_are_marked = top_k_values.issubset(base3_set)
mass_concentrated = marked_mass > 0.8 * shots

verified = all_top_k_are_marked and mass_concentrated

if verified:
    print("PASS")
else:
    print("FAIL")
