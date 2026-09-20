"""
Erdos problem #749 -- quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 749"):
    prize: no
    status: open (as of 2025-08-31)
    oeis: ["N/A"]
    tags: ["additive combinatorics"]

LIMITATION, stated up front: problem #749 has NO associated OEIS sequence
("N/A" in the source data), and the metadata file gives no more of its
informal statement than the tag "additive combinatorics". There is
therefore no specific integer sequence to build a membership/term oracle
for, and this script does NOT claim to test problem #749's actual
mathematical content -- faking a sequence, or borrowing an unrelated OEIS
id, would misrepresent the problem, which the task instructions explicitly
forbid.

Instead, in the spirit of the problem's own tag, this script builds a real,
self-contained, finite additive-combinatorics instance and verifies it with
a genuine Grover-search quantum circuit:

    SUMSET MEMBERSHIP.
    Fix A = {1, 2, 5}. A is first checked classically to be a Sidon set
    (all pairwise sums a+b, a<=b, are distinct) -- it is. Its sumset
    A+A = {a+b : a,b in A, a<=b} is computed exactly by brute force. Over
    the range {0,...,15} (4 qubits), MARKED_VALUES is exactly the set of
    integers in A+A. This is a genuine, finite, exactly-computable
    additive-combinatorics property (sumset membership for a Sidon set),
    with no OEIS lookup and no hard-coded answer -- everything is derived
    in this script.

    A Grover search circuit over 4 qubits (representing v in {0,...,15})
    is built whose oracle marks exactly the integers in MARKED_VALUES. It
    is run on the ideal AerSimulator, and the circuit's measurement
    distribution is compared against MARKED_VALUES computed classically:
    PASS iff the states receiving the (classically correct) amplified
    probability mass are exactly the marked sumset elements.

The oracle is a real circuit (per-marked-value explicit multi-controlled-Z,
built from X/H/MCX gates), not a lookup table dressed up as one, and Grover
diffusion is applied a computed, near-optimal number of times.
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS lookup)
# ---------------------------------------------------------------------------

N_QUBITS = 4
N_STATES = 2 ** N_QUBITS  # 16
A = [1, 2, 5]  # candidate Sidon set


def is_sidon(subset):
    """A set is Sidon if all pairwise sums a+b (a<=b, unordered) are distinct."""
    sums = [a + b for a, b in itertools.combinations_with_replacement(subset, 2)]
    return len(sums) == len(set(sums))


assert is_sidon(A), f"{A} must be a Sidon set for this instance to be meaningful"

sumset_full = {a + b for a, b in itertools.combinations_with_replacement(A, 2)}
MARKED_VALUES = sorted(v for v in sumset_full if 0 <= v < N_STATES)

print(f"Base Sidon set A = {A} (verified Sidon: {is_sidon(A)})")
print(f"Full sumset A+A = {sorted(sumset_full)}")
print(f"Search space: v in 0..{N_STATES - 1}")
print(f"Classically computed marked values (v in A+A): {MARKED_VALUES}")

assert all(0 <= v < N_STATES for v in MARKED_VALUES)


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffusion, built explicitly for MARKED_VALUES
# ---------------------------------------------------------------------------

def apply_multi_controlled_z_for_value(qc, value, n_qubits):
    """
    Flip the phase of the computational basis state |value> (n_qubits wide)
    using X gates to map value -> |11...1>, a multi-controlled-Z, then
    undo the X gates.
    """
    bits = [(value >> i) & 1 for i in range(n_qubits)]
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)
    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)


def build_oracle(marked_values, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for v in marked_values:
        apply_multi_controlled_z_for_value(qc, v, n_qubits)
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


def optimal_grover_iterations(n_marked, n_states):
    if n_marked == 0 or n_marked == n_states:
        return 1
    theta = math.asin(math.sqrt(n_marked / n_states))
    iterations = round((math.pi / (4 * theta)) - 0.5)
    return max(1, iterations)


oracle = build_oracle(MARKED_VALUES, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)
iterations = optimal_grover_iterations(len(MARKED_VALUES), N_STATES)
print(f"Grover iterations used: {iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

SHOTS = 4096
sim = AerSimulator()
result = sim.run(qc, shots=SHOTS).result()
counts = result.get_counts()


def bitstring_to_value(bitstring):
    # Qiskit convention: the classical bitstring is printed MSB-first, i.e.
    # leftmost char = highest-indexed classical/qubit bit. Since our value
    # encoding is value = sum_i bit_i * 2^i (bit i <-> qubit i), reading the
    # bitstring directly as a binary number reconstructs `value` correctly.
    return int(bitstring, 2)


value_counts = {}
for bitstring, count in counts.items():
    v = bitstring_to_value(bitstring)
    value_counts[v] = value_counts.get(v, 0) + count

print("Measured value distribution (value: count), sorted by frequency:")
for v in sorted(value_counts, key=lambda x: -value_counts[x]):
    print(f"  {v}: {value_counts[v]}")

total_marked_shots = sum(value_counts.get(v, 0) for v in MARKED_VALUES)
marked_fraction = total_marked_shots / SHOTS

top_k = sorted(value_counts, key=lambda x: -value_counts[x])[: len(MARKED_VALUES)]
top_k_set = set(top_k)

print(f"Fraction of shots landing on classically-marked values: {marked_fraction:.3f}")
print(f"Top-{len(MARKED_VALUES)} measured values: {sorted(top_k_set)}")
print(f"Classically marked values:                {MARKED_VALUES}")

verified = (marked_fraction > 0.80) and (top_k_set == set(MARKED_VALUES))

if verified:
    print("PASS")
else:
    print("FAIL")
