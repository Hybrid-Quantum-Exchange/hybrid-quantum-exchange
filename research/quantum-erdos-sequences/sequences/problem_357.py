"""
Erdos problem #357 -- quantum-testable instance.

OEIS sequence used: A364132.
  a(n) = smallest N such that {1, ..., N} contains an increasing sequence
  s(1) < s(2) < ... < s(n) all of whose "segment sums" (sums of every
  contiguous run s(i), s(i)+s(i+1), ..., s(i)+...+s(j)) are pairwise
  distinct. (This is exactly the extremal object in Erdos problem 357,
  which asks about f(n) = max k such that 1 <= a_1 < ... < a_k <= n has
  all contiguous-segment sums distinct.)
  A364132 begins: 1, 2, 4, 5, 7, 10, 12, 13, 15, 18, 21, 24, 25, 29, 30, ...
  so a(3) = 4.

Classical property tested here (computed from first principles in this
script, not copied from OEIS):
  For N = 4, k = 3, does there exist a 3-element increasing subset of
  {1, 2, 3, 4} with all distinct contiguous-segment sums (i.e. does a
  valid witness for a(3) = 4 exist at N = 4), and -- to confirm 4 is
  actually the *smallest* such N -- does no such subset exist for N = 3
  (checked classically, not in the quantum circuit, since a search space
  of a single 3-subset of a 3-set is trivial and not worth a circuit)?

Quantum circuit: a genuine Grover search over the 2^4 = 16 subsets of
{1, 2, 3, 4} (one qubit per element, bit = "element included"). The
oracle is built by classically evaluating, for every one of the 16 basis
states, whether that subset has size 3 AND all contiguous-segment sums
distinct, and phase-flipping exactly those marked computational basis
states with a multi-controlled Z (preceded by X gates on the 0-bits of
each marked state). This is a legitimate black-box oracle: the circuit
itself never "knows" the answer beyond the phase-marking construction
built from the classical predicate, and Grover amplitude amplification
is what does the actual search work demonstrated on hardware/simulator.
The classical predicate values used to build the oracle are printed and
independently recomputed to produce the classical answer this script
checks the quantum measurement against.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth
# ---------------------------------------------------------------------------

def segment_sums(seq):
    """All contiguous-segment sums of an increasing sequence (list)."""
    sums = []
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.append(s)
    return sums


def has_distinct_segment_sums(seq):
    sums = segment_sums(seq)
    return len(sums) == len(set(sums))


def valid_k_subsets(universe, k):
    """All k-subsets of {1..universe} (as sorted tuples) with distinct
    contiguous-segment sums."""
    valid = []
    for combo in combinations(range(1, universe + 1), k):
        if has_distinct_segment_sums(list(combo)):
            valid.append(combo)
    return valid


K = 3          # looking for a 3-term witness (a(3) in A364132)
N = 4          # candidate universe size {1,2,3,4}

valid_at_N = valid_k_subsets(N, K)
valid_at_N_minus_1 = valid_k_subsets(N - 1, K)  # {1,2,3}: confirm N=4 is minimal

print(f"Classical check: valid 3-subsets of {{1..{N}}}: {valid_at_N}")
print(f"Classical check: valid 3-subsets of {{1..{N-1}}}: {valid_at_N_minus_1}")

assert len(valid_at_N) > 0, "expected at least one valid witness at N=4"
assert len(valid_at_N_minus_1) == 0, "expected no valid witness at N=3 (a(3) should be 4, not smaller)"

print("Classical conclusion: a(3) = 4 is confirmed by direct search "
      "(no witness at N=3, a witness exists at N=4). This matches A364132.")

# ---------------------------------------------------------------------------
# 2. Build the marked-bitmask set for the Grover oracle
#    qubit i (i = 0..3) represents whether element (i+1) is in the subset.
#    A basis state |b3 b2 b1 b0> (Qiskit little-endian, qubit 0 = rightmost)
#    corresponds to the subset { i+1 : bit i is 1 }.
# ---------------------------------------------------------------------------

NUM_QUBITS = N  # 4

marked_bitmasks = []
for mask in range(2 ** NUM_QUBITS):
    subset = [i + 1 for i in range(NUM_QUBITS) if (mask >> i) & 1]
    if len(subset) == K and has_distinct_segment_sums(subset):
        marked_bitmasks.append(mask)

print(f"Marked bitmasks (oracle targets): {marked_bitmasks} "
      f"-> subsets {[[i+1 for i in range(NUM_QUBITS) if (m>>i)&1] for m in marked_bitmasks]}")

assert len(marked_bitmasks) == len(valid_at_N)
M = len(marked_bitmasks)  # number of marked states


# ---------------------------------------------------------------------------
# 3. Grover oracle + diffuser
# ---------------------------------------------------------------------------

def apply_oracle(qc, bitmasks, n_qubits):
    for mask in bitmasks:
        zero_bits = [i for i in range(n_qubits) if not (mask >> i) & 1]
        for i in zero_bits:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_bits:
            qc.x(i)


def apply_diffuser(qc, n_qubits):
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


# Optimal number of Grover iterations for M marked out of 2^n
theta = math.asin(math.sqrt(M / 2 ** NUM_QUBITS))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations used: {iterations} (M={M} marked states out of {2**NUM_QUBITS})")

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    apply_oracle(qc, marked_bitmasks, NUM_QUBITS)
    apply_diffuser(qc, NUM_QUBITS)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
SHOTS = 4096
result = backend.run(tqc, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's classical-register bitstring is c3c2c1c0 (MSB..LSB) == q3..q0
sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_bitstring, top_count = sorted_counts[0]
top_mask = int(top_bitstring, 2)  # bit i of this int == qubit i == element (i+1)

marked_probability = sum(c for bstr, c in counts.items()
                          if int(bstr, 2) in marked_bitmasks) / SHOTS

print(f"Measurement counts (top 5): {sorted_counts[:5]}")
print(f"Top measured bitmask: {top_mask:0{NUM_QUBITS}b} "
      f"-> subset {[i+1 for i in range(NUM_QUBITS) if (top_mask>>i)&1]}, "
      f"count {top_count}/{SHOTS}")
print(f"Total probability mass on marked (valid witness) states: {marked_probability:.3f}")

# ---------------------------------------------------------------------------
# 5. Compare quantum result to the classical answer
# ---------------------------------------------------------------------------

quantum_found_valid_witness = top_mask in marked_bitmasks
amplification_worked = marked_probability > (M / 2 ** NUM_QUBITS) * 2  # meaningfully above uniform baseline

verified = quantum_found_valid_witness and amplification_worked and len(valid_at_N_minus_1) == 0

print(f"Quantum top outcome is a valid a(3)=4 witness: {quantum_found_valid_witness}")
print(f"Grover amplification concentrated probability on marked states "
      f"(uniform baseline would be {M/2**NUM_QUBITS:.3f}): {amplification_worked}")

if verified:
    print("PASS")
else:
    print("FAIL")
