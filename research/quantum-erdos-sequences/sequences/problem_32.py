"""
Erdos problem #32 (erdosproblems.com) — quantum-testable instance.

Source metadata (from data/problems.yaml in the manman4/erdosproblems clone,
entry "number: '32'"):
    tags: ["number theory", "additive basis"]
    oeis: ["N/A"]

HONEST LIMITATION: problem #32's metadata carries no OEIS sequence id
(oeis: ["N/A"]) and no explicit formula/statement text is present in the
data file beyond its tags. There is therefore no OEIS-derived sequence to
build a membership/term-search circuit against. Rather than fabricate an
OEIS value, this script instead builds a genuine, finite, classically
checkable instance of the one concrete mathematical notion the tags do
give us: an "additive basis of order 2" in number theory — a finite set A
of non-negative integers such that every integer in some target range can
be written as a sum of two elements of A (with repetition, order
irrelevant). This is exactly the kind of object Erdos-style additive-basis
problems are about, and it is small and finite enough for a real quantum
circuit to search.

Concrete instance tested here:
    A = [1, 2, 3, 5]                (4 elements -> index register of 2+2=4 qubits)
    target = 6

Classical property under test:
    "Does there exist a pair of indices (i, j) in {0,1,2,3}^2 such that
     A[i] + A[j] == target?"
This is first computed directly in Python (brute force over all 16 index
pairs) to get the ground-truth marked set and the ground-truth answer
(True/False, plus the witnessing pairs).

Quantum method:
    Grover's algorithm over the 4-qubit index register (2 qubits for i,
    2 qubits for j, 16 basis states total). The oracle is built by phase-
    flipping exactly the classically-precomputed marked index pairs
    (a standard, legitimate way to realize a Grover oracle for a small,
    explicitly enumerable predicate — the predicate itself, A[i]+A[j]==target,
    is evaluated classically only to decide which computational basis states
    the oracle circuit marks; the circuit itself performs the marking and the
    amplitude amplification, and its output is what is checked against the
    classical answer). The number of Grover iterations is the standard
    round(pi/4 * sqrt(N/M)) for N=16 basis states and M marked states.

Verification:
    Run the circuit on the ideal AerSimulator, take the most probable
    measured index pair(s), and check that they are among the
    classically-computed marked (witnessing) pairs. PASS if so, else FAIL.
"""

import math
import itertools

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth
# ---------------------------------------------------------------------------

A = [1, 2, 3, 5]
TARGET = 6
N_INDICES = len(A)  # 4 -> 2 qubits per index register
assert N_INDICES == 4, "instance sized for a 2-qubit index register"

# Brute-force all (i, j) in {0,1,2,3}^2 and find the ones with A[i]+A[j]==TARGET.
classical_marked_pairs = [
    (i, j) for i, j in itertools.product(range(N_INDICES), repeat=2)
    if A[i] + A[j] == TARGET
]
classical_answer_exists = len(classical_marked_pairs) > 0

print(f"Set A = {A}, target = {TARGET}")
print(f"Classical brute force: marked (i, j) pairs = {classical_marked_pairs}")
print(f"Classical answer: additive-basis witness exists = {classical_answer_exists}")

assert classical_answer_exists, "chosen instance must have at least one witness"


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the classically-enumerated marked states
# ---------------------------------------------------------------------------
# Register layout (4 qubits total, little-endian in Qiskit's bit ordering):
#   qubits [0,1] encode i (2 bits), qubits [2,3] encode j (2 bits)

N_QUBITS = 4
N_STATES = 2 ** N_QUBITS  # 16


def index_pair_to_bits(i, j):
    """Return the 4-bit string (q3 q2 q1 q0) for basis state |j>|i>."""
    return format(j, "02b") + format(i, "02b")


marked_bitstrings = [index_pair_to_bits(i, j) for (i, j) in classical_marked_pairs]


def apply_oracle(qc: QuantumCircuit, bitstring: str):
    """Phase-flip the single computational basis state given by `bitstring`
    (MSB-first, matching qc.measure ordering q3 q2 q1 q0)."""
    # X-gate on qubits that should be 0 in the target bitstring, so that the
    # target state becomes |1111>, then apply a multi-controlled Z via
    # MCX with an ancilla-free phase trick (H-MCX-H on the last qubit).
    bits = bitstring[::-1]  # convert to q0 q1 q2 q3 order
    zero_qubits = [q for q, b in enumerate(bits) if b == "0"]
    for q in zero_qubits:
        qc.x(q)

    qc.h(N_QUBITS - 1)
    qc.append(MCXGate(N_QUBITS - 1), list(range(N_QUBITS - 1)) + [N_QUBITS - 1])
    qc.h(N_QUBITS - 1)

    for q in zero_qubits:
        qc.x(q)


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


M = len(marked_bitstrings)
iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / M)))
print(f"Grover: N={N_STATES} states, M={M} marked, iterations={iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

diffuser = build_diffuser(N_QUBITS)

for _ in range(iterations):
    for bs in marked_bitstrings:
        apply_oracle(qc, bs)
    qc.compose(diffuser, inplace=True)

qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

sim = AerSimulator()
SHOTS = 4096
result = sim.run(qc, shots=SHOTS).result()
counts = result.get_counts()

# Sort outcomes by frequency, most probable first.
sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
top_bitstring, top_count = sorted_counts[0]
top_prob = top_count / SHOTS

print(f"Top measured bitstring: {top_bitstring} (q3q2q1q0), prob={top_prob:.3f}")
print(f"All marked bitstrings (classical): {marked_bitstrings}")


def bitstring_to_index_pair(bitstring: str):
    bits = bitstring  # q3 q2 q1 q0, matches index_pair_to_bits output directly
    j = int(bitstring[0:2], 2)
    i = int(bitstring[2:4], 2)
    return i, j


measured_i, measured_j = bitstring_to_index_pair(top_bitstring)
measured_is_witness = (measured_i, measured_j) in classical_marked_pairs

# Aggregate probability mass landing on ANY marked state, as a sanity check
# that Grover genuinely amplified the marked subspace (not just luck on the
# single top outcome).
marked_prob_mass = sum(
    c for bs, c in counts.items() if bs in marked_bitstrings
) / SHOTS

print(f"Measured top pair (i, j) = ({measured_i}, {measured_j}); "
      f"A[i]+A[j] = {A[measured_i]}+{A[measured_j]} = {A[measured_i] + A[measured_j]}")
print(f"Total probability mass on marked states: {marked_prob_mass:.3f}")

quantum_matches_classical = measured_is_witness and marked_prob_mass > 0.5

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")
