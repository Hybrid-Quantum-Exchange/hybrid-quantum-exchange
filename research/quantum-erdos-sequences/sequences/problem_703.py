"""
Erdos problem #703 -- quantum-testable instance
=================================================

OEIS sequence used: A390645
  "Triangle read by rows: T(n,r) is maximal such that there exists a
  family F of subsets of {1,...,n} of size T(n,r) such that the
  intersection of no two sets in F has r elements."
  (Extremal set-theory problem in the style of Frankl-Furedi
  intersection theorems; Erdos problem #703, status "proved",
  prize $250, tag "combinatorics".)

Classical property tested (small, finite, computable)
-------------------------------------------------------
We use the r = 0 column of the triangle, T(n, 0): the largest possible
size of a family F of subsets of an n-element ground set such that no
two (distinct) members of F have empty intersection -- i.e. F is an
"intersecting family". The classical extremal result (and the value
that A390645 records at r=0) is

        T(n, 0) = 2 ** (n - 1)      for n >= 1.

For the small instance n = 2 the ground set is {1, 2} and the power
set has the 4 subsets

        S0 = {}      S1 = {1}      S2 = {2}      S3 = {1,2}

A "selection" is a 4-bit string b3 b2 b1 b0 (bit i = 1 means Si is
included in the family). The predicate

        valid(b) := for every pair i != j with bits i and j both set,
                    Si intersect Sj is non-empty

is computed directly (bitwise AND on the subset masks), and

        M = max popcount(b) over all b with valid(b) == True

is the classical answer to be checked against 2**(n-1) = 2, and the
exact set TARGETS of 4-bit strings achieving this maximum is computed
by brute force over all 16 possible selections -- this is the ground
truth the quantum circuit is checked against, derived in this script,
not copied from OEIS.

Quantum circuit
----------------
A genuine Grover search over the 4-qubit selection register:
  * The oracle is built from TARGETS (computed classically above): for
    each target bitstring it flips the 0-bits with X gates, applies a
    multi-controlled Z (phase flip) on all 4 qubits, and undoes the X
    gates. This is the standard "mark these specific basis states"
    Grover oracle, built strictly from the classically-derived target
    set -- it is not hand-picked or guessed.
  * The diffuser is the standard Grover diffusion operator on 4
    qubits.
  * The optimal number of Grover iterations is computed from the
    standard formula floor(pi/4 * sqrt(N/M)) with N = 16, M =
    |TARGETS|.

The circuit is run on the ideal AerSimulator (statevector-based qasm
simulation, no noise). PASS is declared iff the most frequently
measured 4-bit string, when decoded back into a subset family and
re-checked with the *same* classical predicate, is valid, has size
exactly M, and M == 2 ** (n - 1) as the extremal theorem (and
A390645's r=0 column) predicts.
"""

import itertools
from collections import Counter

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical setup: ground set {1, 2}, its 4 subsets, and the intersecting
#    family predicate. All derived from first principles in this script.
# ---------------------------------------------------------------------------

N_GROUND = 2
SUBSETS = list(range(2 ** N_GROUND))  # bitmask representation: S0=0b00, S1=0b01, S2=0b10, S3=0b11
NUM_SUBSETS = len(SUBSETS)  # 4


def is_valid_family(selection_bits):
    """selection_bits: int in [0, 2**NUM_SUBSETS). Bit i set => subset i is in the family.

    Returns True iff every pair of distinct selected subsets has non-empty
    intersection (an "intersecting family").
    """
    selected = [i for i in range(NUM_SUBSETS) if (selection_bits >> i) & 1]
    for i, j in itertools.combinations(selected, 2):
        if (SUBSETS[i] & SUBSETS[j]) == 0:
            return False
    return True


def popcount(x, width):
    return bin(x).count("1")


# Brute-force over all 2**NUM_SUBSETS = 16 selections (small, finite, exact).
all_results = {}
for sel in range(2 ** NUM_SUBSETS):
    all_results[sel] = (is_valid_family(sel), popcount(sel, NUM_SUBSETS))

valid_sizes = [size for sel, (valid, size) in all_results.items() if valid]
M_CLASSICAL = max(valid_sizes)
TARGETS = sorted(sel for sel, (valid, size) in all_results.items() if valid and size == M_CLASSICAL)

EXPECTED_M = 2 ** (N_GROUND - 1)  # A390645 T(n,0) = 2**(n-1)

print(f"Classical brute force over ground set size n={N_GROUND}, {NUM_SUBSETS} subsets:")
print(f"  maximum intersecting-family size M = {M_CLASSICAL}")
print(f"  A390645 T(n,0) = 2**(n-1) predicts   = {EXPECTED_M}")
print(f"  target selections (4-bit, LSB=S0..S3 membership): {[format(t, '04b') for t in TARGETS]}")

assert M_CLASSICAL == EXPECTED_M, "classical brute force disagrees with the extremal formula"


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over the 4-qubit selection register,
#    oracle built from TARGETS (computed purely classically above).
# ---------------------------------------------------------------------------

NUM_QUBITS = NUM_SUBSETS  # one qubit per subset membership bit


def build_oracle(num_qubits, targets):
    qc = QuantumCircuit(num_qubits, name="oracle")
    for t in targets:
        bits = [(t >> i) & 1 for i in range(num_qubits)]
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z across all qubits (phase flip on |11...1>)
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


import math

N_STATES = 2 ** NUM_QUBITS
M_TARGETS = len(TARGETS)
num_iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / M_TARGETS)))

qreg = QuantumRegister(NUM_QUBITS, "sel")
grover = QuantumCircuit(qreg)
grover.h(range(NUM_QUBITS))

oracle = build_oracle(NUM_QUBITS, TARGETS)
diffuser = build_diffuser(NUM_QUBITS)

for _ in range(num_iterations):
    grover.compose(oracle, inplace=True)
    grover.compose(diffuser, inplace=True)

grover.measure_all()

print(f"\nGrover circuit: {NUM_QUBITS} qubits, {M_TARGETS} marked targets out of {N_STATES}, "
      f"{num_iterations} Grover iteration(s).")

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(grover, backend)
job = backend.run(transpiled, shots=4096)
result = job.result()
counts = result.get_counts()

# Qiskit's measure_all bit ordering: the returned bitstring has qubit (n-1)
# as the leftmost character, i.e. bitstring[::-1] gives qubit index -> bit.
def bitstring_to_selection(bitstring):
    bits = bitstring[::-1]
    sel = 0
    for i, ch in enumerate(bits):
        if ch == "1":
            sel |= (1 << i)
    return sel

decoded_counts = Counter()
for bitstring, cnt in counts.items():
    decoded_counts[bitstring_to_selection(bitstring)] += cnt

most_common_selection, most_common_count = decoded_counts.most_common(1)[0]
total_shots = sum(decoded_counts.values())

print(f"\nTop measured selection: {format(most_common_selection, '04b')} "
      f"({most_common_count}/{total_shots} shots = {most_common_count/total_shots:.1%})")

# Fraction of shots landing on any marked (classically-valid maximum) target.
hits_on_targets = sum(cnt for sel, cnt in decoded_counts.items() if sel in TARGETS)
print(f"Fraction of shots on a classically-valid maximum-size intersecting family: "
      f"{hits_on_targets/total_shots:.1%}")

quantum_valid, quantum_size = is_valid_family(most_common_selection), popcount(most_common_selection, NUM_QUBITS)

verified = (
    most_common_selection in TARGETS
    and quantum_valid
    and quantum_size == M_CLASSICAL
    and M_CLASSICAL == EXPECTED_M
    and hits_on_targets / total_shots > 0.5  # Grover amplification actually worked
)

print(f"\nQuantum top result decodes to a valid intersecting family: {quantum_valid}, "
      f"size {quantum_size} (classical max {M_CLASSICAL}, A390645 prediction {EXPECTED_M})")

if verified:
    print("PASS")
else:
    print("FAIL")
