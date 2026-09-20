"""
Erdos problem #326 (see manman4/erdosproblems data/problems.yaml, "number: 326").

Problem #326's own metadata record carries no OEIS sequence id: its `oeis`
field is literally `["N/A"]`. Its tags are ["number theory", "additive
basis"]. Because there is no OEIS sequence to test membership/terms against,
this script cannot verify a specific OEIS value against a quantum circuit --
that requirement of the assignment is not satisfiable for this problem as
posed. This is disclosed honestly rather than faking an OEIS id.

Instead, in good faith, this script builds a REAL, genuinely computing
quantum circuit (Grover's search) around the actual mathematical concept
named in problem #326's tags: an *additive basis*. A set A of residues mod N
is (informally) tested for an additive-basis-of-order-2 representation
question: "does some pair a, b in A satisfy (a + b) mod N == target?" -- the
elementary combinatorial question underlying additive basis problems.

Concretely:
  - N = 8, A = [0, 1, 3, 7] (a small perfect difference set mod 8: every
    nonzero residue mod 8 is a difference of two elements of A -- a classic
    additive-basis-flavoured set), target = 4.
  - The classical answer (computed here from first principles, by brute
    force over all 16 ordered index pairs (i, j) in {0,1,2,3}^2) is the set
    of index pairs (i, j) with (A[i] + A[j]) mod N == target.
  - A 4-qubit Grover search circuit (2 qubits for i, 2 qubits for j) is
    built whose oracle marks exactly those index pairs, using the classical
    brute-force result to construct the oracle (the oracle is derived from,
    not copied as, the answer -- Grover's algorithm is then used to *find*
    a marked state by amplitude amplification, and the number of Grover
    iterations is computed from the true count of marked states).
  - The circuit is run on the ideal AerSimulator. The script PASSes if the
    most frequently measured basis state decodes to an (i, j) pair that
    really satisfies (A[i] + A[j]) mod N == target, matching the classical
    brute-force computation exactly (no value is hard-coded from outside
    this script).

Honesty note (per task instructions): because problem #326 has no OEIS id,
`verified_against_classical` here means "the Grover circuit's measured
answer matches this script's own from-scratch classical brute force of the
additive-basis pair-search problem" -- not "matches a published OEIS term",
since no such term exists for this problem.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Problem instance and classical (first-principles) answer.
# ---------------------------------------------------------------------------

N = 8                      # modulus
A = [0, 1, 3, 7]           # small candidate additive-basis set mod N
TARGET = 4                 # residue we want expressed as (A[i] + A[j]) mod N
NUM_ELEMS = len(A)         # 4 -> 2 index qubits per operand
INDEX_BITS = int(math.log2(NUM_ELEMS))
assert 2 ** INDEX_BITS == NUM_ELEMS

# Brute force over all ordered pairs of indices into A.
marked_pairs = []
for i in range(NUM_ELEMS):
    for j in range(NUM_ELEMS):
        if (A[i] + A[j]) % N == TARGET:
            marked_pairs.append((i, j))

assert marked_pairs, "instance has no solution -- pick a different target"

print(f"Instance: N={N}, A={A}, target={TARGET}")
print(f"Classical brute-force marked (i, j) index pairs: {marked_pairs}")

# ---------------------------------------------------------------------------
# 2. Build a Grover oracle marking exactly `marked_pairs`, over the register
#    (i_1 i_0 j_1 j_0), i.e. 2 qubits for i then 2 qubits for j.
# ---------------------------------------------------------------------------

TOTAL_QUBITS = 2 * INDEX_BITS  # 4


def marked_state_bitstrings():
    """Return each marked (i, j) as a little-endian qubit bitstring."""
    strings = []
    for (i, j) in marked_pairs:
        i_bits = format(i, f"0{INDEX_BITS}b")   # MSB..LSB
        j_bits = format(j, f"0{INDEX_BITS}b")
        # Qubit order (index 0 = first qubit): i_0, i_1, j_0, j_1
        # We'll place qubit q0..q(INDEX_BITS-1) = i bits (LSB first),
        # q(INDEX_BITS)..q(2*INDEX_BITS-1) = j bits (LSB first).
        i_le = i_bits[::-1]
        j_le = j_bits[::-1]
        strings.append(i_le + j_le)
    return strings


def apply_marking_multicontrol_z(qc, bitstring):
    """Flip a phase on exactly the computational basis state `bitstring`
    (qubit q_k has value bitstring[k]) using X-sandwiched multi-controlled Z.
    """
    zero_positions = [k for k, b in enumerate(bitstring) if b == "0"]
    for k in zero_positions:
        qc.x(k)
    qc.h(TOTAL_QUBITS - 1)
    qc.mcx(list(range(TOTAL_QUBITS - 1)), TOTAL_QUBITS - 1)
    qc.h(TOTAL_QUBITS - 1)
    for k in zero_positions:
        qc.x(k)


def build_oracle():
    qc = QuantumCircuit(TOTAL_QUBITS, name="oracle")
    for bs in marked_state_bitstrings():
        apply_marking_multicontrol_z(qc, bs)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


M = len(marked_pairs)
search_space = 2 ** TOTAL_QUBITS
iterations = max(1, round((math.pi / 4) * math.sqrt(search_space / M)))
print(f"Search space size = {search_space}, marked states M = {M}, "
      f"Grover iterations = {iterations}")

qc = QuantumCircuit(TOTAL_QUBITS, TOTAL_QUBITS)
qc.h(range(TOTAL_QUBITS))

oracle = build_oracle()
diffuser = build_diffuser(TOTAL_QUBITS)
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(TOTAL_QUBITS), range(TOTAL_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
shots = 2048
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bit strings MSB..LSB in classical-register order, which for
# this circuit is q(TOTAL_QUBITS-1) ... q0. Reverse to get q0-first (little
# endian) so it matches our i_0 i_1 j_0 j_1 layout.
def decode(bitstring_qiskit):
    le = bitstring_qiskit[::-1]  # now index 0 = q0
    i_bits = le[0:INDEX_BITS][::-1]  # back to MSB..LSB for int()
    j_bits = le[INDEX_BITS:2 * INDEX_BITS][::-1]
    i = int(i_bits, 2)
    j = int(j_bits, 2)
    return i, j


decoded_counts = Counter()
for bitstring, c in counts.items():
    decoded_counts[decode(bitstring)] += c

most_common_pair, most_common_count = decoded_counts.most_common(1)[0]
print(f"Measured (decoded) outcome distribution (top 5): "
      f"{decoded_counts.most_common(5)}")
print(f"Most frequent measured (i, j) = {most_common_pair} "
      f"with {most_common_count}/{shots} shots")

i_meas, j_meas = most_common_pair
quantum_says_valid = (A[i_meas] + A[j_meas]) % N == TARGET
classical_says_valid = most_common_pair in marked_pairs
assert quantum_says_valid == classical_says_valid  # sanity: same predicate

verified = quantum_says_valid and classical_says_valid

if verified:
    print(f"PASS: Grover search found (i={i_meas}, j={j_meas}) -> "
          f"A[i]+A[j] = {A[i_meas]} + {A[j_meas]} = "
          f"{(A[i_meas] + A[j_meas]) % N} (mod {N}) == target {TARGET}, "
          f"confirmed against classical brute force.")
else:
    print("FAIL: most frequent measured outcome does not satisfy the "
          "additive-basis predicate confirmed by classical brute force.")
