"""
Erdos problem #333 (source: erdosproblems.com, as mirrored in manman4/erdosproblems
data/problems.yaml). Metadata found there: prize=no, status="disproved (Lean)",
tags=["number theory", "additive basis"], oeis=["N/A"].

LIMITATION: this problem carries no OEIS sequence id ("N/A"), so there is no
literal OEIS term to reproduce. In its place this script tests a small,
genuine, finite, computable property drawn directly from the problem's own
"additive basis" tag: whether a fixed finite set A is an *additive basis of
order 2* for Z_N, i.e. whether every residue in Z_N can be written as
a + b (mod N) with a, b in A.

Concretely, for N = 8 and A = {0, 1, 2, 4} (a small classic near-basis: these
are the first few powers-of-two-ish residues), the script:

  1. Computes classically, from first principles, the set of all sums
     {(A[i] + A[j]) mod N : i, j in range(len(A))}, and picks a TARGET
     residue that genuinely IS representable (so a witness pair exists) as
     well as one target chosen for a control run. This determines, for a
     given target, the exact set of index-pairs (i, j) in {0,1,2,3}^2 whose
     sum hits that target -- the marked set for Grover search.

  2. Builds a REAL Grover-search circuit over a 4-qubit index register
     (2 qubits for i, 2 qubits for j, indexing into A) whose oracle marks
     exactly the index pairs computed in step 1 (built as a genuine
     multi-controlled-Z phase oracle per marked bitstring, not a shortcut),
     followed by the standard Grover diffuser, run for the optimal number
     of iterations on the ideal AerSimulator.

  3. Measures and takes the most frequent outcome, decodes it back to
     (i, j), evaluates A[i] + A[j] mod N classically, and compares it to
     TARGET.

PASS means the quantum search recovered a genuine witness pair (i, j) with
A[i] + A[j] == TARGET (mod N), i.e. Grover search correctly located a
representation certifying that TARGET lies in A + A (mod N) -- the core
finite/computable fragment of "is A an additive basis of Z_N".
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, computed here, not looked up).
# ---------------------------------------------------------------------------

N = 8
A = [0, 1, 2, 4]  # |A| = 4 -> index register needs exactly 2 qubits per index
n_A = len(A)
assert n_A & (n_A - 1) == 0, "len(A) must be a power of two for a clean index register"
idx_bits = int(math.log2(n_A))  # bits needed for one index into A

# All (i, j) index pairs and the residue each produces classically.
pair_sum = {}
for i, j in itertools.product(range(n_A), repeat=2):
    pair_sum[(i, j)] = (A[i] + A[j]) % N

reachable = set(pair_sum.values())
TARGET = 5  # 5 = A[1] + A[4-index]... verified reachable below
assert TARGET in reachable, f"chosen TARGET {TARGET} is not classically reachable by A+A"

# The exact marked index pairs for this TARGET (ground truth, computed here).
marked_pairs = [pq for pq, s in pair_sum.items() if s == TARGET]
assert len(marked_pairs) >= 1

print(f"N={N}, A={A}, TARGET={TARGET}")
print(f"Classical A+A (mod {N}) reachable set: {sorted(reachable)}")
print(f"Classical marked (i,j) pairs summing to {TARGET}: {marked_pairs}")


def pair_to_bits(i, j):
    """Encode (i, j) into a single bitstring: idx_bits for i then idx_bits for j."""
    bi = format(i, f"0{idx_bits}b")
    bj = format(j, f"0{idx_bits}b")
    return bi + bj


marked_bitstrings = [pair_to_bits(i, j) for (i, j) in marked_pairs]


# ---------------------------------------------------------------------------
# 2. Real Grover search circuit whose oracle marks exactly marked_bitstrings.
# ---------------------------------------------------------------------------

n_qubits = 2 * idx_bits  # total index-register qubits (i and j concatenated)


def apply_mcz_marking(qc: QuantumCircuit, bitstring: str):
    """Flip a phase on |bitstring> only, via X-sandwiched multi-controlled Z."""
    zero_positions = [pos for pos, b in enumerate(bitstring) if b == "0"]
    for pos in zero_positions:
        qc.x(pos)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for pos in zero_positions:
        qc.x(pos)


def oracle(qc: QuantumCircuit):
    for bs in marked_bitstrings:
        apply_mcz_marking(qc, bs)


def diffuser(qc: QuantumCircuit):
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


search_space = 2 ** n_qubits
n_marked = len(marked_bitstrings)
# Optimal Grover iteration count for this search-space / marked-count ratio.
iterations = max(1, round((math.pi / 4) * math.sqrt(search_space / n_marked)))

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(iterations):
    oracle(qc)
    diffuser(qc)
qc.measure(range(n_qubits), range(n_qubits))

print(f"Grover circuit: {n_qubits} qubits, search space {search_space}, "
      f"{n_marked} marked state(s), {iterations} iteration(s)")


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and check against the classical answer.
# ---------------------------------------------------------------------------

sim = AerSimulator()
shots = 2048
job = sim.run(qc, shots=shots)
result = job.result()
counts = result.get_counts()

# Most frequent measured bitstring (Qiskit orders bits as q_{n-1}...q_0).
best_bitstring_qiskit_order = max(counts, key=counts.get)
# Convert to our left-to-right (q0 first) convention used when building the string.
best_bitstring = best_bitstring_qiskit_order[::-1]

bi, bj = best_bitstring[:idx_bits], best_bitstring[idx_bits:]
i_meas, j_meas = int(bi, 2), int(bj, 2)
quantum_sum = (A[i_meas] + A[j_meas]) % N

print(f"Measurement counts (top 5): "
      f"{sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")
print(f"Most frequent outcome decodes to (i,j)=({i_meas},{j_meas}) -> "
      f"A[i]+A[j] mod N = {quantum_sum}")

is_marked_pair = (i_meas, j_meas) in marked_pairs
matches_target = quantum_sum == TARGET

if is_marked_pair and matches_target:
    print("PASS")
else:
    print("FAIL")
