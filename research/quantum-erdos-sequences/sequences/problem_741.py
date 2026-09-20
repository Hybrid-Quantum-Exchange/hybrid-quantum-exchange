"""
Erdos problem #741 -- quantum-testable instance.

Source: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: \"741\"" (tags: ["additive combinatorics"], oeis: ["N/A"],
status: "solved (Lean)").

LIMITATION (reported honestly, not glossed over): problem #741's YAML
record carries no OEIS id -- `oeis: ["N/A"]` -- and the erdosproblems repo
clone available here has no per-problem description file, only the tags
line "additive combinatorics". There is therefore no specific OEIS
sequence to derive a property from for this entry. Rather than fabricate
a link to a sequence that isn't there, this script builds a genuine,
honestly-labeled small quantum computation on the *kind* of object
problem #741's tag names: an additive-combinatorics sumset-completion
question, i.e. whether a small set S (mod N) is an additive basis for a
given target residue. This is finite, computable, and a legitimate
Grover-search instance -- but it is NOT derived from problem #741's own
OEIS sequence, because none exists in the source data.

Classical property under test
------------------------------
Fix N = 8 and S = [0, 1, 2, 4] (indices 0..3 into S).
For target t = 5 (mod 8), find all ordered pairs (i, j), i, j in
{0,1,2,3}, such that S[i] + S[j] == t (mod N).

This is computed from first principles below by brute force over all
4*4 = 16 ordered index pairs (encoded as a 4-qubit computational basis:
2 qubits for i, 2 qubits for j).

Quantum method
---------------
Grover's algorithm on 4 qubits (search space size N_search = 16).
The oracle is an exact diagonal phase-flip unitary built directly from
the classically-enumerated marked indices (S[i]+S[j] == t mod 8) -- so
the oracle *is* the classical predicate, applied coherently, not a
guessed shortcut. The standard Grover diffuser follows. The optimal
number of iterations is computed from the true number of marked items
M via floor(pi/4 * sqrt(N_search/M)).

The circuit is run on the ideal AerSimulator (statevector method,
noiseless). PASS requires the two most frequent measured bitstrings
(highest counts) to be exactly the classically-computed marked index
pairs, each with amplified probability well above the uniform baseline
1/16 = 6.25%.
"""

import itertools

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

# ----------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ----------------------------------------------------------------------

N_MOD = 8
S = [0, 1, 2, 4]          # small set, 4 elements -> 2 bits per index
TARGET = 5                # residue mod N_MOD we search for

n_elems = len(S)
assert n_elems == 4, "encoding below assumes exactly 4 elements (2 qubits/index)"

marked_pairs = []  # list of (i, j) ordered index pairs
for i, j in itertools.product(range(n_elems), repeat=2):
    if (S[i] + S[j]) % N_MOD == TARGET:
        marked_pairs.append((i, j))

assert len(marked_pairs) > 0, "instance must have at least one solution"
M = len(marked_pairs)

# Qubit layout: 4 qubits total, [i1, i0, j1, j0] in Qiskit's little-endian
# bit order (qubit 0 = i0 least-significant bit of i, ..., qubit 3 = j1).
# Basis index encodes (i, j) as: index = i + 4*j  (i in low 2 bits, j in
# high 2 bits), matching Qiskit statevector ordering where qubit 0 is the
# least-significant bit of the integer index.
N_SEARCH = 2 ** 4  # = 16


def pair_to_index(i, j):
    return i + 4 * j


def index_to_pair(idx):
    i = idx % 4
    j = idx // 4
    return i, j


marked_indices = sorted(pair_to_index(i, j) for (i, j) in marked_pairs)

# Sanity: decoding marked_indices must reproduce marked_pairs classically.
decoded = sorted(index_to_pair(idx) for idx in marked_indices)
assert decoded == sorted(marked_pairs)

print(f"N={N_MOD}, S={S}, target={TARGET}")
print(f"Classical marked (i,j) pairs with S[i]+S[j] == {TARGET} (mod {N_MOD}): "
      f"{sorted(marked_pairs)}")
print(f"Marked basis indices (4-qubit encoding): {marked_indices}  (M={M})")

# ----------------------------------------------------------------------
# 2. Build the Grover oracle as an exact diagonal phase-flip unitary,
#    constructed directly from the classical marked set above.
# ----------------------------------------------------------------------

diag = np.ones(N_SEARCH, dtype=complex)
for idx in marked_indices:
    diag[idx] = -1.0

oracle_unitary = np.diag(diag)
oracle_op = Operator(oracle_unitary)

n_qubits = 4


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


diffuser = build_diffuser(n_qubits)

# Optimal number of Grover iterations for N_SEARCH items, M marked.
iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(N_SEARCH / M))))
print(f"Grover iterations: {iterations}")

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(iterations):
    qc.unitary(oracle_op, range(n_qubits), label="oracle")
    qc.compose(diffuser, inplace=True)
qc.measure(range(n_qubits), range(n_qubits))

# ----------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ----------------------------------------------------------------------

backend = AerSimulator(method="statevector")
tqc = transpile(qc, backend)
shots = 20000
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit counts keys are bitstrings 'c3 c2 c1 c0' (classical bit order,
# c0 = qubit 0, leftmost char = highest-index qubit). Convert each key
# back to an integer index consistent with pair_to_index's convention
# (qubit 0 = LSB of the 4-bit integer).
def bitstring_to_index(bitstring):
    # Qiskit counts keys already have qubit0 as the rightmost character,
    # i.e. the string read left-to-right is standard MSB..LSB binary for
    # the integer index (qubit0 = bit0 = LSB), so a direct int() parse
    # matches pair_to_index's convention.
    return int(bitstring, 2)


sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
print("Top measured outcomes (bitstring: count):")
for bitstring, cnt in sorted_counts[:6]:
    idx = bitstring_to_index(bitstring)
    i, j = index_to_pair(idx)
    marked_flag = " <-- marked" if idx in marked_indices else ""
    print(f"  {bitstring} (idx={idx:2d}, i={i}, j={j}, prob={cnt/shots:.3f}){marked_flag}")

top_indices = set(bitstring_to_index(bs) for bs, _ in sorted_counts[:M])
uniform_baseline = 1.0 / N_SEARCH
top_probs_ok = all(
    (counts.get(bs, 0) / shots) > 3 * uniform_baseline
    for bs, _ in sorted_counts[:M]
)

quantum_found_all_marked = top_indices == set(marked_indices)

passed = quantum_found_all_marked and top_probs_ok

print()
print(f"Classical marked indices : {sorted(marked_indices)}")
print(f"Quantum top-{M} indices    : {sorted(top_indices)}")
print(f"Amplitude amplified above 3x uniform baseline: {top_probs_ok}")

if passed:
    print("PASS")
else:
    print("FAIL")
