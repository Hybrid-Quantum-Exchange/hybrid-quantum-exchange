"""
Erdos problem #868 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems.com mirror, as cloned at
/home/user/manman4/erdosproblems): problem 868 is tagged
["number theory", "additive basis"], is listed as informally "solved" and
formally verified in Lean, and its `oeis` field is `["N/A"]` -- the problem
has NO associated OEIS sequence id in the source data. There is therefore no
"OEIS-derived small term/membership property" that can honestly be extracted
for this problem the way the task template expects; any script that pretends
otherwise would be fabricating a link that the source data does not contain.

LIMITATION (stated up front, per the task's own fallback instructions): since
there is no OEIS id to anchor a term-membership property to, this script does
NOT test an OEIS sequence. Instead it tests a small, finite, fully classically
checkable instance of the actual mathematical notion the problem is tagged
with -- being an "additive basis of order 2" -- which is a legitimate,
well-defined finite decision/search problem with real content, not a made-up
stand-in. Concretely:

    Let A = {0, 1, 2, 4} (a small Sidon-like set, 4 elements, indices 0..3).
    A is an additive basis of order 2 for the range R = {0, 1, ..., 6} if
    every target t in R can be written as t = A[i] + A[j] for some indices
    i, j in {0,1,2,3} (repetition allowed).

    This script classically enumerates, for target t = 5, every pair
    (i, j) in {0,1,2,3}^2 with A[i] + A[j] == 5 (first principles,
    brute force, computed in this script). It then builds a genuine Grover
    search circuit over the 4-qubit space of (i, j) pairs (2 qubits for i,
    2 qubits for j) whose oracle marks exactly those classically-determined
    solution states (implemented as multi-controlled Z gates on the specific
    marked bitstrings -- not a shortcut, the oracle is derived from the
    classical enumeration above), runs one Grover iteration set tuned to the
    number of marked states, executes on the ideal AerSimulator, and checks
    that the most-probable measured outcome(s) are exactly the classically
    verified solutions.

    PASS means: the quantum search's top output(s), decoded back to (i, j)
    pairs, are exactly the set of pairs with A[i] + A[j] == 5 -- i.e. the
    quantum circuit correctly finds a certificate that A[i]+A[j]=5 is
    solvable within A, which is one instance of the additive-basis question
    the problem is tagged with.

Reported accurately: ran_ok and verified_against_classical reflect whether
this concrete run below succeeded, not any claim about problem 868's OEIS
status (it has none).
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical setup: the finite instance, computed from first principles.
# ---------------------------------------------------------------------------

A = [0, 1, 2, 4]          # small set, tagged "additive basis" in problem 868
TARGET = 5                 # t = A[i] + A[j] ?
N_INDEX_QUBITS = 2          # 2 qubits encode indices 0..3 for i, another 2 for j
TOTAL_QUBITS = 2 * N_INDEX_QUBITS  # 4 qubits total, search space size 16

# Brute-force classical enumeration of every (i, j) with A[i] + A[j] == TARGET.
classical_solutions = []
for i, j in product(range(len(A)), repeat=2):
    if A[i] + A[j] == TARGET:
        classical_solutions.append((i, j))

assert classical_solutions, "instance must have at least one solution to search for"
print(f"Set A = {A}, target = {TARGET}")
print(f"Classical brute-force solutions (i, j) with A[i]+A[j]=={TARGET}: {classical_solutions}")


def bits_for_pair(i, j):
    """4-bit string (Qiskit little-endian: qubit0 is rightmost char) encoding (i, j)."""
    # qubits [0,1] = i (LSB first), qubits [2,3] = j (LSB first)
    i_bits = format(i, f"0{N_INDEX_QUBITS}b")[::-1]
    j_bits = format(j, f"0{N_INDEX_QUBITS}b")[::-1]
    # Qiskit bit-string ordering in results is q_{n-1}...q_0, so build MSB->LSB.
    full = [None] * TOTAL_QUBITS
    for q in range(N_INDEX_QUBITS):
        full[q] = i_bits[q]
    for q in range(N_INDEX_QUBITS):
        full[N_INDEX_QUBITS + q] = j_bits[q]
    return "".join(full[::-1])  # reverse -> qiskit's q_{n-1}..q_0 printable order


marked_bitstrings = sorted({bits_for_pair(i, j) for (i, j) in classical_solutions})
print(f"Marked computational basis states (qiskit bit order): {marked_bitstrings}")


# ---------------------------------------------------------------------------
# 2. Build a genuine Grover oracle for exactly these marked states.
# ---------------------------------------------------------------------------

def apply_oracle(qc, qubits, bitstring):
    """Flip the phase of |bitstring> (qiskit order, q_{n-1}..q_0) via multi-controlled Z."""
    n = len(qubits)
    # bitstring[0] corresponds to qubit n-1 ... bitstring[n-1] corresponds to qubit 0
    zero_positions = [n - 1 - k for k, b in enumerate(bitstring) if b == "0"]
    for pos in zero_positions:
        qc.x(qubits[pos])
    if n == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for pos in zero_positions:
        qc.x(qubits[pos])


def build_oracle(n_qubits, marked):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for bs in marked:
        apply_oracle(qc, list(range(n_qubits)), bs)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


n = TOTAL_QUBITS
N_states = 2 ** n
M = len(marked_bitstrings)

# Optimal number of Grover iterations for M marked states out of N_states.
theta = math.asin(math.sqrt(M / N_states))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"N={N_states} states, M={M} marked, using {iterations} Grover iteration(s)")

oracle = build_oracle(n, marked_bitstrings)
diffuser = build_diffuser(n)

qc = QuantumCircuit(n, n)
qc.h(range(n))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(n))
    qc.append(diffuser.to_gate(), range(n))
qc.measure(range(n), range(n))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Take the top M most frequent outcomes (M = number of classical solutions).
sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
top_outcomes = [bs for bs, _ in sorted_counts[:M]]

print(f"Top {M} measured outcome(s) (bitstring: count): "
      f"{[(bs, counts[bs]) for bs in top_outcomes]}")


# ---------------------------------------------------------------------------
# 4. Decode top outcomes back to (i, j) pairs and verify classically.
# ---------------------------------------------------------------------------

def decode_bitstring(bs):
    # bs is qiskit order q_{n-1}..q_0
    bits = bs[::-1]  # now bits[0] = q0, bits[1] = q1, ...
    i = int(bits[0:N_INDEX_QUBITS][::-1], 2)
    j = int(bits[N_INDEX_QUBITS:2 * N_INDEX_QUBITS][::-1], 2)
    return i, j


decoded_top = sorted(decode_bitstring(bs) for bs in top_outcomes)
expected = sorted(classical_solutions)

# Each decoded top outcome must genuinely satisfy A[i]+A[j]==TARGET (re-checked
# classically here, not assumed), and the *set* of top outcomes must match the
# full classical solution set for this instance.
all_decoded_valid = all(A[i] + A[j] == TARGET for (i, j) in decoded_top)
sets_match = set(decoded_top) == set(expected)

passed = all_decoded_valid and sets_match

print(f"Decoded top outcomes (i, j): {decoded_top}")
print(f"Classical solution set:      {expected}")
print(f"All decoded outcomes satisfy A[i]+A[j]=={TARGET}: {all_decoded_valid}")
print(f"Decoded outcome set matches classical solution set: {sets_match}")

if passed:
    print("PASS")
else:
    print("FAIL")
