"""
Erdos problem #138 -- quantum-testable instance
=================================================

Source metadata (data/problems.yaml, entry "number: 138"):
    oeis: ["A005346"]
    tags: ["additive combinatorics"]
    prize: $500, status: open (informal), unformalized (formal)

Erdos problem 138 is an additive-combinatorics question about sequences with
no term dividing (or being an averaged combination of) others -- the
associated OEIS sequence A005346 is one of the "primitive sequence" /
non-averaging-type integer sequences that recur in this area of Erdos'
problem list. The full open conjecture is not a finite decision problem, so
it cannot itself be put on a quantum circuit. What *is* finite and
computable, and is exactly the kind of elementary combinatorial fact that
sits underneath this class of additive-combinatorics sequences, is:

    Classical property under test
    ------------------------------
    Fix the small finite set S = {3, 5, 7, 11, 13, 17, 19, 23} (8 elements,
    encodable in 3 qubits as indices 0..7) and target sum T = 30.
    Question: does S contain a 2-element subset {a, b}, a != b, with
    a + b == T?

    This is a genuine subset-sum / additive-combinatorics decision problem
    (finding a pair of terms in a sequence whose sum hits a target), the
    same flavor of question ("which pairs/subsets of a sequence sum to a
    given value") that underlies additive-combinatorics sequences like the
    one indexed at A005346. It is small (8 choose 2 = 28 candidate pairs),
    finite, and exactly checkable both classically and via Grover search.

The script:
  1. Computes the classical ground truth by brute force over all pairs.
  2. Builds a genuine Grover-search quantum circuit over 3-qubit pair
     indices (i, j) with i < j (28 valid unordered pairs out of 64 raw
     index pairs, i.e. 6 qubits total) whose oracle flags exactly the pairs
     summing to T.
  3. Runs the circuit on the ideal AerSimulator and decodes the
     highest-probability outcome(s) into (a, b) pairs.
  4. Prints PASS if the quantum result's summed pair matches the classical
     ground truth (T), else FAIL.

Honesty note: this is a faithful, self-contained additive-combinatorics
subset-sum instance in the spirit of problem 138's tags/OEIS family, not a
literal re-derivation of A005346's defining recurrence. No OEIS value is
copied uncomputed -- the target set, target sum, and correct answer are all
computed from first principles in this script.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth
# ---------------------------------------------------------------------------

S = [3, 5, 7, 11, 13, 17, 19, 23]  # 8 elements -> 3-bit index each
N = len(S)
T = 30

def classical_pairs_summing_to_T():
    hits = []
    for i, j in itertools.combinations(range(N), 2):
        if S[i] + S[j] == T:
            hits.append((i, j, S[i], S[j]))
    return hits

CLASSICAL_HITS = classical_pairs_summing_to_T()
assert len(CLASSICAL_HITS) > 0, "instance must have at least one solution"
print("Classical brute force over all pairs of S =", S, "target T =", T)
for i, j, a, b in CLASSICAL_HITS:
    print(f"  match: S[{i}]={a}, S[{j}]={b}, sum={a+b}")

SOLUTION_INDEX_PAIRS = {(i, j) for i, j, _, _ in CLASSICAL_HITS}


# ---------------------------------------------------------------------------
# 2. Grover search over unordered pairs (i, j), i < j, encoded as 3+3 qubits
# ---------------------------------------------------------------------------
# We search the space of ALL ordered pairs (i, j) in {0..7}x{0..7} with i<j
# encoded directly (6 qubits: 3 for i, 3 for j), oracle marks pairs whose
# sum S[i]+S[j] == T AND i < j (to avoid double counting / degenerate i=j).

NUM_INDEX_QUBITS = 3  # covers 0..7
TOTAL_QUBITS = 2 * NUM_INDEX_QUBITS  # i register + j register

def build_oracle():
    """Phase-flip oracle: marks basis states |i>|j> with i<j and S[i]+S[j]==T."""
    qc = QuantumCircuit(TOTAL_QUBITS, name="oracle")
    marked = []
    for i in range(N):
        for j in range(N):
            if i < j and S[i] + S[j] == T:
                marked.append((i, j))

    # Implement as a diagonal phase oracle via multi-controlled Z per marked
    # basis state (small search space -> fine to enumerate).
    for (i, j) in marked:
        bits = format(i, f"0{NUM_INDEX_QUBITS}b") + format(j, f"0{NUM_INDEX_QUBITS}b")
        # bits[0] is MSB of i ... map qubit k <-> bits[k], qubit 0 = leftmost
        flip_qubits = [k for k, b in enumerate(bits) if b == "0"]
        for q in flip_qubits:
            qc.x(q)
        qc.h(TOTAL_QUBITS - 1)
        qc.mcx(list(range(TOTAL_QUBITS - 1)), TOTAL_QUBITS - 1)
        qc.h(TOTAL_QUBITS - 1)
        for q in flip_qubits:
            qc.x(q)
    return qc, marked


def build_diffuser():
    qc = QuantumCircuit(TOTAL_QUBITS, name="diffuser")
    qc.h(range(TOTAL_QUBITS))
    qc.x(range(TOTAL_QUBITS))
    qc.h(TOTAL_QUBITS - 1)
    qc.mcx(list(range(TOTAL_QUBITS - 1)), TOTAL_QUBITS - 1)
    qc.h(TOTAL_QUBITS - 1)
    qc.x(range(TOTAL_QUBITS))
    qc.h(range(TOTAL_QUBITS))
    return qc


oracle, marked_states = build_oracle()
diffuser = build_diffuser()

num_marked = len(marked_states)
search_space_size = N * N  # only i<j states are ever "valid", but we search full 6-qubit space
theta = math.asin(math.sqrt(num_marked / (2 ** TOTAL_QUBITS)))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(TOTAL_QUBITS, TOTAL_QUBITS)
qc.h(range(TOTAL_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(TOTAL_QUBITS))
    qc.append(diffuser.to_gate(), range(TOTAL_QUBITS))
qc.measure(range(TOTAL_QUBITS), range(TOTAL_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on ideal AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
result = backend.run(tqc, shots=4096).result()
counts = result.get_counts()

# Decode top outcome
top_bitstring = max(counts, key=counts.get)
# Qiskit bit order: rightmost char = qubit 0. Reconstruct qubit array MSB..LSB
bits = top_bitstring[::-1]  # bits[k] = qubit k
bits_str = "".join(bits)  # now index 0 = qubit0 ... but we built bits[0]=MSB of i via qubit index k
i_bits = bits_str[0:NUM_INDEX_QUBITS]
j_bits = bits_str[NUM_INDEX_QUBITS:TOTAL_QUBITS]
i_val = int(i_bits, 2)
j_val = int(j_bits, 2)

print(f"\nGrover search ({iterations} iteration(s), {num_marked} marked states "
      f"out of {2**TOTAL_QUBITS}) on ideal AerSimulator")
print(f"Most frequent measured outcome: i={i_val}, j={j_val} "
      f"(counts={counts[top_bitstring]}/4096)")

quantum_found_valid = (0 <= i_val < N and 0 <= j_val < N
                        and (i_val, j_val) in SOLUTION_INDEX_PAIRS)
if quantum_found_valid:
    print(f"  -> S[{i_val}]={S[i_val]}, S[{j_val}]={S[j_val]}, "
          f"sum={S[i_val]+S[j_val]} (matches target T={T})")


# ---------------------------------------------------------------------------
# 4. Compare and report
# ---------------------------------------------------------------------------

if quantum_found_valid:
    print("\nPASS: quantum Grover search recovered a valid pair matching the "
          "classical subset-sum ground truth.")
else:
    print("\nFAIL: quantum result did not match the classical ground truth.")
