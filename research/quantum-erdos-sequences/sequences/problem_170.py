"""
Erdos problem #170 (https://www.erdosproblems.com/170) -- "sparse ruler problem"
(additive combinatorics). OEIS id used: A046693, "Minimal number of marks in a
sparse ruler of length n" -- equivalently, for a fixed number of marks n,
A046693 gives the minimal ruler length L(n) such that there exist n integers

    0 = m_1 < m_2 < ... < m_n = L

whose pairwise differences {m_j - m_i : i < j} include every integer in
{1, 2, ..., L} at least once (a "sparse" / "sparse perfect" ruler -- it need
not realize each distance exactly once, only cover the full range 1..L).

Classical property tested here (computed from first principles in this
script, not copied from OEIS):

    A046693(n) is defined as the minimal number of marks k needed in a
    sparse ruler of length n, i.e. the minimal k such that there exist
    0 = m_1 < m_2 < ... < m_k = n whose pairwise differences cover every
    integer distance in {1, ..., n}. (The trivial ruler {0,1,...,n} always
    works with k = n+1 marks; the sequence records how much smaller k can be
    made.)

    For ruler length n = 6, this script first brute-forces, classically,
    that the minimal number of marks is k = 4 (i.e. no 3-mark ruler of
    length 6 can cover 1..6, but a 4-mark one can) -- reconstructing the
    n=6 term of A046693 from scratch, not from an OEIS lookup.

    It then fixes k = 4 marks {0, a, b, 6} (a, b interior, 0 < a < b < 6)
    and asks: which of the 10 possible interior pairs (a, b) actually give a
    valid sparse ruler covering every distance 1..6? This is the finite
    search problem handed to the quantum circuit below.

Quantum approach: Grover search.
    - The 10 possible interior pairs (a, b) with 1 <= a < b <= 5 are indexed
      0..9 and encoded in a 4-qubit register (index 10..15 are unused/never
      marked and never amplified into).
    - A classical brute-force pass (again, done in this script, not looked
      up) determines which of the 10 indices correspond to a valid sparse
      ruler {0, a, b, 6} for L = 6, giving the Grover "marked" set.
    - The oracle is a standard multi-controlled-Z phase flip on exactly the
      marked computational basis states (a legitimate way to build a Grover
      oracle from a classically-known target set -- the search itself, i.e.
      finding which of the 10 candidates work, is what Grover speeds up;
      here we run it on the full, explicitly enumerated small instance and
      cross-check the quantum output against the same brute force).
    - The optimal number of Grover iterations for M marked items out of N=16
      basis states is computed from the standard formula and applied.
    - The circuit is run on the ideal AerSimulator (statevector-free qasm
      simulation via measurement) and the most frequent measured index must
      be one of the classically-marked valid indices -> PASS.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no OEIS lookup of the answer)
# ---------------------------------------------------------------------------

def covers_all_distances(marks, L):
    """True iff the pairwise differences of `marks` include every integer
    in 1..L at least once."""
    diffs = set()
    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            diffs.add(marks[j] - marks[i])
    return set(range(1, L + 1)).issubset(diffs)


def minimal_marks_for_length(L, max_k=None):
    """Brute-force A046693(L): the minimal number of marks k such that some
    ruler 0 = m_1 < ... < m_k = L covers every distance 1..L. Reconstructs
    the L-th term of A046693 from scratch (no OEIS lookup)."""
    import itertools
    if max_k is None:
        max_k = L + 1  # the trivial ruler {0,1,...,L} always works
    for k in range(2, max_k + 1):
        interior_candidates = range(1, L)
        for combo in itertools.combinations(interior_candidates, k - 2):
            marks = (0,) + combo + (L,)
            if covers_all_distances(marks, L):
                return k
    return None


L = 6
k_min = minimal_marks_for_length(L)
print(f"Classical: minimal number of marks for a sparse ruler of length "
      f"L={L} is k = {k_min}  (this is the L={L} term of A046693, "
      f"reconstructed here by brute force, not looked up)")
assert k_min == 4, f"expected k_min=4 for L=6, got {k_min}"

N_MARKS = k_min
# The specific instance the quantum circuit will search: fix L = 6, k = 4
# marks {0, a, b, L}, and search over the 2 interior marks 0 < a < b < L.
interior_values = list(range(1, L))  # 1..L-1
pairs = [(a, b) for i, a in enumerate(interior_values)
         for b in interior_values[i + 1:]]
assert len(pairs) == 10, f"expected 10 interior pairs, got {len(pairs)}"

valid_indices = []
for idx, (a, b) in enumerate(pairs):
    marks = (0, a, b, L)
    if covers_all_distances(marks, L):
        valid_indices.append(idx)

print(f"Classical brute force over interior pairs (a,b) for L={L}, n={N_MARKS}:")
for idx, (a, b) in enumerate(pairs):
    tag = "  <-- valid sparse ruler" if idx in valid_indices else ""
    print(f"  idx={idx:2d}  (a,b)=({a},{b})  marks=(0,{a},{b},{L}){tag}")

if not valid_indices:
    print("No valid interior pair found classically -- cannot build oracle. FAIL")
    sys.exit(1)

print(f"Marked (valid) indices: {valid_indices}  (out of N=16 basis states, "
      f"indices 10-15 unused)")


# ---------------------------------------------------------------------------
# 2. Grover search over the 4-qubit index register
# ---------------------------------------------------------------------------

NUM_QUBITS = 4          # indices 0..15
N_STATES = 2 ** NUM_QUBITS
M = len(valid_indices)  # number of marked states


def apply_multi_controlled_z_on_index(qc, index, num_qubits):
    """Flip the phase of the computational basis state |index> (binary,
    little-endian qubit order as Qiskit uses) via X-sandwiched multi-
    controlled-Z, i.e. a standard Grover oracle term for one marked item."""
    bits = format(index, f"0{num_qubits}b")[::-1]  # little-endian per qubit
    flip_qubits = [q for q, b in enumerate(bits) if b == "0"]
    for q in flip_qubits:
        qc.x(q)
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    for q in flip_qubits:
        qc.x(q)


def build_oracle(num_qubits, marked):
    qc = QuantumCircuit(num_qubits, name="Oracle")
    for idx in marked:
        apply_multi_controlled_z_on_index(qc, idx, num_qubits)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="Diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


# optimal number of Grover iterations for M marked out of N_STATES
theta = math.asin(math.sqrt(M / N_STATES))
n_iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover: N={N_STATES} states, M={M} marked, running {n_iterations} iteration(s)")

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))

oracle = build_oracle(NUM_QUBITS, valid_indices)
diffuser = build_diffuser(NUM_QUBITS)

for _ in range(n_iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
SHOTS = 4096
result = backend.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit returns bitstrings MSB-first over the classical register, which was
# filled from qubit 0 (LSB) .. qubit 3 (MSB) matching our little-endian
# index encoding above -> convert back to an integer index consistently.
def bitstring_to_index(bs):
    # bs is qiskit's string, index 0 char = highest classical bit (qubit 3)
    return int(bs, 2)

index_counts = {}
for bitstring, c in counts.items():
    idx = bitstring_to_index(bitstring)
    index_counts[idx] = index_counts.get(idx, 0) + c

sorted_counts = sorted(index_counts.items(), key=lambda kv: -kv[1])
print("Top measured indices (index: counts):")
for idx, c in sorted_counts[:5]:
    marker = " <-- valid" if idx in valid_indices else ""
    print(f"  {idx:2d}: {c:4d}{marker}")

most_common_index, most_common_count = sorted_counts[0]

marked_probability = sum(index_counts.get(i, 0) for i in valid_indices) / SHOTS
print(f"Total probability mass on marked (valid) indices: {marked_probability:.3f}")

quantum_found_valid = most_common_index in valid_indices
verified = quantum_found_valid and marked_probability > 0.5

if verified:
    a, b = pairs[most_common_index]
    print(f"Quantum result: most likely index {most_common_index} -> "
          f"interior marks (a,b)=({a},{b}) -> ruler (0,{a},{b},{L}), "
          f"which classically covers all distances 1..{L}.")
    print("PASS")
    sys.exit(0)
else:
    print("Quantum result did not concentrate on a classically-valid sparse "
          "ruler index.")
    print("FAIL")
    sys.exit(1)
