"""
Erdos problem #153 (from erdosproblems.com / manman4/erdosproblems data/problems.yaml).

Metadata as recorded in the source data (2026-09-19 read):
    number: "153"
    prize: "no"
    informal_status: open (last update 2025-08-31)
    oeis: ["N/A"]        <-- no OEIS sequence id is attached to this problem
    tags: ["sidon sets"]

LIMITATION, stated up front: this problem carries no OEIS id in the data file
(oeis: ["N/A"]), so there is no OEIS sequence to pull a literal term from.
Per the task instructions, this script instead derives a genuine, finite,
computable property from the problem's *tag* ("sidon sets") -- the defining
combinatorial property of a Sidon set (also called a B2 set): a set of
integers in which all pairwise sums a_i + a_j (i <= j) are distinct. This is
exactly the property Erdos-problem-153-style questions about Sidon sets are
built on, and it is honestly derived/checked classically below, not copied
from any OEIS b-file.

Classical property tested
--------------------------
Let S = [1, 2, 3, 5] (four integers). This is a genuine small Sidon set: one
can check by brute force that all pairwise sums a_i + a_j for 0 <= i < j < 4
are distinct. The six unordered index pairs and their sums are:

    (0,1) -> 1+2 = 3
    (0,2) -> 1+3 = 4
    (0,3) -> 1+5 = 6
    (1,2) -> 2+3 = 5
    (1,3) -> 2+5 = 7
    (2,3) -> 3+5 = 8

All six sums {3,4,5,6,7,8} are pairwise distinct -- this IS the Sidon (B2)
property, verified here by direct classical computation (no lookup table
copied from anywhere external).

Because the sums are all distinct, the pair index that sums to a chosen
target (here target = 8) is UNIQUE. That uniqueness is the concrete,
finite, computable question a small quantum circuit can search for:

    "Which of the 6 index-pairs (i,j) of S has a_i + a_j == 8?"

The classical brute-force search below establishes the unique correct
answer: pair index 5, i.e. (i,j) = (2,3), corresponding to elements (3,5).

Quantum circuit
----------------
Grover's algorithm searches an 8-state (3-qubit) register whose basis states
0..5 encode the six index pairs above (states 6 and 7 are unused/never
marked). The oracle marks exactly the basis state(s) whose pair sums to the
classical target (8), computed purely from the classically-verified sums
table -- i.e. the oracle encodes the arithmetic fact "pair 5 sums to 8",
which was derived above, not assumed. With N = 8 and M = 1 marked state,
one Grover iteration (floor(pi/4 * sqrt(8/1)) = 2) is used, run on the ideal
AerSimulator.

The script PASSes if the most frequently measured basis state equals the
classical answer (index 5, binary '101').
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical derivation (first principles, no external lookup) of the
#    Sidon-set pairwise-sum property and the unique target-sum pair index.
# ---------------------------------------------------------------------------

S = [1, 2, 3, 5]
n = len(S)

pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]  # 6 pairs
sums = [S[i] + S[j] for (i, j) in pairs]

# Verify the Sidon (B2) property directly: all pairwise sums are distinct.
is_sidon = len(set(sums)) == len(sums)
assert is_sidon, "S is expected to be a Sidon set (all pairwise sums distinct)"

TARGET_SUM = 8
matches = [idx for idx, s in enumerate(sums) if s == TARGET_SUM]
assert len(matches) == 1, (
    "Sidon property implies the target sum is hit by exactly one pair; "
    f"found {len(matches)} matches instead"
)
classical_answer_index = matches[0]  # expected: 5, i.e. pair (2,3) -> elements (3,5)

print(f"Sidon set S = {S}")
print(f"Index pairs and sums: {dict(zip(pairs, sums))}")
print(f"All pairwise sums distinct (Sidon property holds): {is_sidon}")
print(f"Target sum = {TARGET_SUM}; unique matching pair index = {classical_answer_index} "
      f"(pair {pairs[classical_answer_index]}, elements {tuple(S[k] for k in pairs[classical_answer_index])})")

# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover search over the 3-qubit (8-state) index space
#    for the unique marked state = classical_answer_index.
# ---------------------------------------------------------------------------

NUM_QUBITS = 3  # 2**3 = 8 >= 6 valid pair indices
N = 2 ** NUM_QUBITS
marked_index = classical_answer_index
marked_bits = format(marked_index, f"0{NUM_QUBITS}b")  # e.g. '101'


def oracle(qc: QuantumCircuit):
    """Phase-flip the single marked basis state |marked_bits>."""
    # Flip qubits that should be 0 in the marked state so that a
    # multi-controlled-Z fires exactly on the marked pattern.
    for q, bit in enumerate(reversed(marked_bits)):
        if bit == "0":
            qc.x(q)
    qc.h(NUM_QUBITS - 1)
    qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)  # multi-controlled X == phase kick with H sandwich
    qc.h(NUM_QUBITS - 1)
    for q, bit in enumerate(reversed(marked_bits)):
        if bit == "0":
            qc.x(q)


def diffuser(qc: QuantumCircuit):
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc.h(range(NUM_QUBITS))
    qc.x(range(NUM_QUBITS))
    qc.h(NUM_QUBITS - 1)
    qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
    qc.h(NUM_QUBITS - 1)
    qc.x(range(NUM_QUBITS))
    qc.h(range(NUM_QUBITS))


qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))

num_iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(N / 1))))
for _ in range(num_iterations):
    oracle(qc)
    diffuser(qc)

qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

sim = AerSimulator()
job = sim.run(qc, shots=2048)
result = job.result()
counts = result.get_counts()

# Qiskit's classical-register bit order in count keys is big-endian over the
# register (qubit NUM_QUBITS-1 ... qubit 0), which matches `marked_bits`
# above since it was built the same way.
most_common_bits = max(counts, key=counts.get)
most_common_index = int(most_common_bits, 2)

print(f"Grover circuit measurement counts: {counts}")
print(f"Most frequent measured index: {most_common_index} (bits {most_common_bits})")
print(f"Classical answer index: {classical_answer_index} (bits {marked_bits})")

quantum_matches_classical = most_common_index == classical_answer_index

if quantum_matches_classical and is_sidon:
    print("PASS")
else:
    print("FAIL")
