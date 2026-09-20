"""
Erdos problem #882 -- quantum-testable sequence lane.

Source metadata (from erdosproblems.com data, `data/problems.yaml`, entry
`number: "882"`): tags = ["number theory", "primitive sets"], status =
solved, and the recorded `oeis` field is the placeholder string
"possible" -- NOT a real OEIS sequence id. There is therefore no concrete
OEIS A-number to build a "membership in sequence" oracle from for this
problem. This script is the honest fallback described in the task: since
no OEIS id exists, we build a genuine, non-fabricated finite/computable
problem drawn directly from the problem's *tag* ("primitive sets"), which
is exactly the mathematical object Erdos problem #882 is about.

Definition (classical, standard number theory): a set of positive
integers S is a PRIMITIVE SET if no element of S divides another element
of S (S is an antichain under the divisibility partial order). This is
the core notion tagged on problem #882.

Chosen small, finite, computable instance
------------------------------------------
Let T = [2, 3, 4, 6, 5, 7]  (6 elements, indices 0..5).
T is NOT a primitive set: e.g. T[0]=2 divides T[2]=4, T[0]=2 divides
T[3]=6, and T[1]=3 divides T[3]=6.

The classical property under test: does there exist an ordered pair of
distinct indices (i, j), i != j, with T[i] dividing T[j]?  (Equivalently:
is T *not* a primitive set, and if not, exhibit a witnessing pair.)

The classical answer is computed from first principles by brute-force
divisibility checking in this script (see `classical_marked_pairs`
below) -- it is not copied from any table. For our 6-element T there are
exactly 3 witnessing ordered pairs among the 30 ordered pairs (i != j).

Quantum circuit
----------------
We use Grover's algorithm to search the 6-bit space of ordered index
pairs (i, j) in {0,...,5} x {0,...,5}, i != j (represented as two 3-bit
registers, 64 basis states total, of which 36 correspond to valid
i,j in range and 30 of those have i != j) for a pair witnessing
T[i] | T[j]. The oracle is built directly from the classically
precomputed list of marked pairs (phase-kickback oracle: flip an
ancilla-encoded phase for exactly the marked computational basis
states), and the Grover diffuser is the standard textbook construction.
We run the circuit on the ideal AerSimulator, take the most-probable
measured index pair, and PASS if that pair is one of the classically
verified witnessing pairs (i.e. quantum search actually found a real,
classically-checked violation of the primitive-set property for T).

This is a real amplitude-amplification search over a genuine
combinatorial predicate (divisibility), not a lookup of a literal OEIS
term -- appropriate given that problem #882 carries no usable OEIS id.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical setup: the instance and the brute-force ground truth.
# ---------------------------------------------------------------------------

T = [2, 3, 4, 6, 5, 7]  # 6 elements -> indices 0..5, fits in 3 bits each
N = len(T)  # 6
NUM_INDEX_BITS = 3  # ceil(log2(6)) = 3, domain per register is 0..7


def classical_marked_pairs(values):
    """Brute-force, from first principles: all ordered (i, j), i != j,
    with values[i] in range and values[j] in range, and values[i] | values[j].
    Returns a sorted list of (i, j) integer index pairs."""
    marked = []
    for i, j in itertools.product(range(len(values)), repeat=2):
        if i == j:
            continue
        if values[j] % values[i] == 0:
            marked.append((i, j))
    return sorted(marked)


MARKED_PAIRS = classical_marked_pairs(T)
IS_PRIMITIVE_SET = len(MARKED_PAIRS) == 0

print(f"Instance T = {T}")
print(f"Classical brute-force witnessing pairs (i,j) with T[i] | T[j], i!=j:")
for (i, j) in MARKED_PAIRS:
    print(f"  T[{i}]={T[i]} divides T[{j}]={T[j]}")
print(f"Classical verdict: T is {'a' if IS_PRIMITIVE_SET else 'NOT a'} primitive set.")
assert not IS_PRIMITIVE_SET, "instance must be a non-primitive set for this search to have a marked target"
assert len(MARKED_PAIRS) == 3, f"expected exactly 3 witnessing pairs, got {len(MARKED_PAIRS)}"


# ---------------------------------------------------------------------------
# 2. Build a Grover oracle over the 6-qubit (i,j) index-pair space that flips
#    the phase of exactly the classically-marked basis states.
# ---------------------------------------------------------------------------

def bits_of(x, n):
    """Little-endian bit list of x using n bits (matches Qiskit qubit order)."""
    return [(x >> k) & 1 for k in range(n)]


def add_marking_oracle(qc, i_qubits, j_qubits, marked_pairs, ancilla):
    """For each marked (i,j), flip the phase of that exact basis state using
    a multi-controlled-X onto an ancilla prepared in the |-> state (standard
    phase-kickback trick), sandwiched with X gates on the 0-bits."""
    all_qubits = list(i_qubits) + list(j_qubits)
    n = len(all_qubits)
    for (i, j) in marked_pairs:
        bits = bits_of(i, len(i_qubits)) + bits_of(j, len(j_qubits))
        zero_positions = [q for q, b in zip(all_qubits, bits) if b == 0]
        for q in zero_positions:
            qc.x(q)
        qc.mcx(all_qubits, ancilla)
        for q in zero_positions:
            qc.x(q)


def add_diffuser(qc, qubits):
    n = len(qubits)
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


NUM_SEARCH_QUBITS = 2 * NUM_INDEX_BITS  # 6 qubits -> 64-element search space
NUM_MARKED = len(MARKED_PAIRS)
SEARCH_SPACE_SIZE = 2 ** NUM_SEARCH_QUBITS  # 64

# Optimal number of Grover iterations for this search-space size / marked count.
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(SEARCH_SPACE_SIZE / NUM_MARKED)))
print(f"\nGrover search space size = {SEARCH_SPACE_SIZE}, marked states = {NUM_MARKED}, "
      f"iterations = {optimal_iterations}")

i_qubits = list(range(NUM_INDEX_BITS))
j_qubits = list(range(NUM_INDEX_BITS, 2 * NUM_INDEX_BITS))
ancilla = 2 * NUM_INDEX_BITS
num_qubits = 2 * NUM_INDEX_BITS + 1

qc = QuantumCircuit(num_qubits, 2 * NUM_INDEX_BITS)

# Prepare ancilla in |-> for phase kickback.
qc.x(ancilla)
qc.h(ancilla)

# Uniform superposition over the (i,j) index-pair space.
qc.h(i_qubits + j_qubits)

for _ in range(optimal_iterations):
    add_marking_oracle(qc, i_qubits, j_qubits, MARKED_PAIRS, ancilla)
    add_diffuser(qc, i_qubits + j_qubits)

qc.measure(i_qubits + j_qubits, list(range(2 * NUM_INDEX_BITS)))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

simulator = AerSimulator()
compiled = transpile(qc, simulator)
SHOTS = 4096
result = simulator.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Decode each bitstring back into (i, j). Qiskit's classical-bit string is
# big-endian in the printed order (c[n-1] ... c[0]); our measure mapped
# clbit k <- (i_qubits+j_qubits)[k], little-endian, so reverse before slicing.
def decode(bitstring):
    bits = bitstring[::-1]  # now little-endian, index 0 first
    i_bits = bits[:NUM_INDEX_BITS]
    j_bits = bits[NUM_INDEX_BITS:2 * NUM_INDEX_BITS]
    i_val = int(i_bits[::-1], 2) if i_bits else 0
    j_val = int(j_bits[::-1], 2) if j_bits else 0
    # bits_of used little-endian per-qubit ordering; reconstruct consistently.
    i_val = sum(int(b) << k for k, b in enumerate(i_bits))
    j_val = sum(int(b) << k for k, b in enumerate(j_bits))
    return i_val, j_val


decoded_counts = {}
for bitstring, count in counts.items():
    pair = decode(bitstring)
    decoded_counts[pair] = decoded_counts.get(pair, 0) + count

top_pair, top_count = max(decoded_counts.items(), key=lambda kv: kv[1])
print(f"\nTop measured (i,j) pair: {top_pair} with {top_count}/{SHOTS} shots "
      f"({100.0 * top_count / SHOTS:.1f}%)")

# Sanity: report how much of the amplitude landed on marked pairs overall.
marked_set = set(MARKED_PAIRS)
marked_shots = sum(c for pair, c in decoded_counts.items() if pair in marked_set)
print(f"Total shots landing on a classically-marked pair: {marked_shots}/{SHOTS} "
      f"({100.0 * marked_shots / SHOTS:.1f}%)")


# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer and report PASS/FAIL.
# ---------------------------------------------------------------------------

quantum_found_valid_witness = top_pair in marked_set
majority_on_marked = marked_shots > SHOTS / 2

verified = quantum_found_valid_witness and majority_on_marked

if verified:
    i, j = top_pair
    print(f"\nQuantum Grover search found witness (i,j)=({i},{j}): "
          f"T[{i}]={T[i]} divides T[{j}]={T[j]}, matching classical brute force.")
    print("RESULT: PASS")
else:
    print("\nQuantum search result did not match the classical brute-force witnesses.")
    print("RESULT: FAIL")
