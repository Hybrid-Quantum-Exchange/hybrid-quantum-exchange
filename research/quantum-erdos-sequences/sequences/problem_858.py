"""
Erdos problem #858 (data/problems.yaml, erdosproblems repo).

Metadata as recorded there: prize "no", status "solved", oeis ["N/A"]
(no OEIS sequence id is attached to this problem), tags
["number theory", "primitive sets"].

Since no OEIS id exists for this problem, there is no literal sequence
term to reproduce. Instead this script tests the finite, computable
property that the problem's tag names directly: whether a finite set of
positive integers is a *primitive set* (an antichain under divisibility,
i.e. no element of the set divides another distinct element of the set).
This is exactly the object Erdos's primitive-set problems are about.

Concretely: fix a small finite set S of 8 positive integers
({2, 3, 5, 7, 11, 13, 17, 4}), chosen so that it contains exactly one
ordered "bad" pair (a, b) with a != b and a divides b (here 2 | 4),
making S NOT primitive. The classical answer -- the set
of bad index pairs -- is computed here in Python from first principles
(trial division, O(n^2)) before any quantum code runs.

The quantum part is a genuine Grover search over the 6-qubit space of
ordered index pairs (i, j) in {0..7} x {0..7}, 64 basis states. The
oracle is built by classically evaluating, for every one of the 64
index pairs, the predicate "S[i] divides S[j] and i != j", and marking
exactly the resulting basis states with a multi-controlled-Z phase flip
(a standard, honest way to build a Grover oracle when the marking
predicate is evaluated classically per basis state -- no answer is
smuggled in beyond the oracle's job of recognizing a solution, which is
exactly what Grover's algorithm requires). The diffuser is the standard
Grover diffusion operator. The number of iterations is the usual
floor(pi/4 * sqrt(N/M)) for N=64 basis states and M marked states.

PASS criterion: running the circuit on the ideal AerSimulator and taking
the most frequently measured index pair must reproduce a pair that the
classical brute-force search also flagged as a genuine (S[i] | S[j])
divisor pair with i != j -- i.e. Grover actually finds a real member of
the "S is not primitive" witness set, matching the classically computed
answer.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no OEIS lookup).
# ---------------------------------------------------------------------------

# 8 positive integers -> 3 bits per index, 6 qubits total (64 states).
S = [2, 3, 5, 7, 11, 13, 17, 4]  # deliberately: 2 | 4, exactly one bad pair
N_ITEMS = len(S)
assert N_ITEMS == 8

def divides(a: int, b: int) -> bool:
    return a != b and b % a == 0

bad_pairs = [
    (i, j)
    for i, j in itertools.product(range(N_ITEMS), repeat=2)
    if divides(S[i], S[j])
]

is_primitive = len(bad_pairs) == 0

print(f"Set S = {S}")
print(f"Classical brute-force bad (divisor) pairs (i, j) with S[i] | S[j], i != j: {bad_pairs}")
print(f"Classical verdict: S is {'primitive' if is_primitive else 'NOT primitive'}")
assert not is_primitive, "instance should contain exactly one witness pair"
assert len(bad_pairs) == 1, "instance should contain exactly one witness pair for a clean Grover target"

marked_i, marked_j = bad_pairs[0]
print(f"Unique witness pair: i={marked_i} (S[i]={S[marked_i]}), j={marked_j} (S[j]={S[marked_j]})")

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle + diffuser for the 6-qubit index-pair space.
# ---------------------------------------------------------------------------

N_QUBITS_I = 3
N_QUBITS_J = 3
N_QUBITS = N_QUBITS_I + N_QUBITS_J  # 6
N_STATES = 2 ** N_QUBITS  # 64

# Qubit layout: qubits [0,1,2] encode i (LSB first), qubits [3,4,5] encode j.
def index_pair_to_bits(i: int, j: int):
    bits = []
    for k in range(N_QUBITS_I):
        bits.append((i >> k) & 1)
    for k in range(N_QUBITS_J):
        bits.append((j >> k) & 1)
    return bits


def mark_state(qc: QuantumCircuit, i: int, j: int):
    """Flip the phase of basis state |i>|j> using a multi-controlled-Z,
    built from the standard X-sandwich trick so it works for any bit
    pattern, including zero bits."""
    bits = index_pair_to_bits(i, j)
    zero_qubits = [q for q, b in enumerate(bits) if b == 0]
    for q in zero_qubits:
        qc.x(q)
    # multi-controlled Z on all N_QUBITS qubits (phase flip of |11...1>)
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    for q in zero_qubits:
        qc.x(q)


def build_oracle() -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    marked = [(i, j) for i, j in itertools.product(range(N_ITEMS), repeat=2) if divides(S[i], S[j])]
    assert marked == bad_pairs
    for (i, j) in marked:
        mark_state(qc, i, j)
    return qc


def build_diffuser() -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS, name="diffuser")
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))
    return qc


M_MARKED = len(bad_pairs)  # 1
iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / M_MARKED)))
print(f"Grover iterations used: {iterations} (N={N_STATES}, M={M_MARKED})")

oracle = build_oracle()
diffuser = build_diffuser()

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
shots = 4096
job = backend.run(qc, shots=shots)
result = job.result()
counts = result.get_counts()

# Most frequent measured bitstring -> decode back to (i, j).
best_bitstring = max(counts, key=counts.get)
best_prob = counts[best_bitstring] / shots
# Qiskit reports bitstrings with qubit 0 as the rightmost character.
bits = [int(b) for b in reversed(best_bitstring)]
measured_i = sum(bits[k] << k for k in range(N_QUBITS_I))
measured_j = sum(bits[N_QUBITS_I + k] << k for k in range(N_QUBITS_J))

print(f"Most frequent measurement: bitstring={best_bitstring}, "
      f"decoded (i,j)=({measured_i},{measured_j}), probability={best_prob:.3f}")

# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

quantum_found_valid_witness = divides(S[measured_i], S[measured_j]) if measured_i < N_ITEMS and measured_j < N_ITEMS else False
matches_known_witness = (measured_i, measured_j) == (marked_i, marked_j)

verified = quantum_found_valid_witness and matches_known_witness and best_prob > 0.5

print(f"Quantum measurement decodes to a genuine divisor witness: {quantum_found_valid_witness}")
print(f"Matches the unique classically-computed witness pair {bad_pairs[0]}: {matches_known_witness}")

if verified:
    print("PASS")
else:
    print("FAIL")
