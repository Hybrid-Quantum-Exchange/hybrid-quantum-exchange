"""
Erdos problem #424 (erdosproblems.com), OEIS A005244 (Hofstadter's self-generating
sequence): start with the set {2, 3}; repeatedly take the product of any two
DISTINCT previous elements, subtract 1, and adjoin the result to the set if it
is not already present; iterate to closure. A005244 is the sorted union of all
elements ever generated. (Erdos problem #424 asks about the growth rate /
density of this sequence; it is open. This script does not attempt the open
research question -- it tests a small, well-defined, finite, computable
membership-style fact about the sequence itself, suitable for a real quantum
search circuit.)

Classical fact tested (computed from first principles below, not copied from
OEIS): generate the sequence by closure up to 200; take B = the first 6
elements of the sequence, B = [2, 3, 5, 9, 14, 17] (an index register of 3
bits each). Exactly two ordered index pairs (i, j) with i != j satisfy
B[i]*B[j] - 1 == 26 (the 7th term of A005244): (i, j) = (1, 3) and (3, 1),
i.e. {B[1], B[3]} = {3, 9}, since 3*9 - 1 = 26.

Quantum approach: Grover's search over a 6-qubit register (3 qubits for i,
3 qubits for j, values 0..5 valid, 6..7 unused/never marked) whose oracle
marks exactly the ordered index pairs (i, j) classically found above to
satisfy B[i]*B[j] - 1 == 26. The oracle is built by explicitly marking those
basis states with multi-controlled-Z gates (a standard way to realize a
black-box Grover oracle for a small, explicitly known solution set), and the
correctness of the *marked set* is established purely by the classical
computation above -- nothing is asserted about which states are solutions
without deriving it in code. Grover amplifies the two marked basis states;
running the circuit on the ideal AerSimulator and taking the most frequent
measured outcomes should recover exactly {(1,3), (3,1)}.

PASS criterion: the two most-sampled 6-bit outcomes from the Grover circuit,
decoded back to (i, j) pairs, equal exactly the classically computed solution
set {(1, 3), (3, 1)}.
"""

import math
from itertools import combinations, product

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no OEIS values copied blindly)
# ---------------------------------------------------------------------------

def generate_A005244(limit):
    """Closure of {2,3} under: adjoin a*b-1 for distinct a,b already present."""
    s = {2, 3}
    changed = True
    while changed:
        changed = False
        elems = sorted(s)
        for a, b in combinations(elems, 2):
            v = a * b - 1
            if v <= limit and v not in s:
                s.add(v)
                changed = True
    return sorted(s)


SEQ = generate_A005244(200)
# Sanity: matches the known start of A005244.
assert SEQ[:8] == [2, 3, 5, 9, 14, 17, 26, 27], SEQ[:8]

B = SEQ[:6]  # [2, 3, 5, 9, 14, 17] -- index register 0..5, 3 bits
assert B == [2, 3, 5, 9, 14, 17]

TARGET = 26
assert TARGET in SEQ

classical_solutions = set()
for i, j in product(range(len(B)), repeat=2):
    if i != j and B[i] * B[j] - 1 == TARGET:
        classical_solutions.add((i, j))

assert classical_solutions == {(1, 3), (3, 1)}, classical_solutions
print(f"Classical search space size: {len(B) ** 2} ordered pairs "
      f"(indices 0..{len(B)-1} each)")
print(f"Classical solutions for B[i]*B[j]-1 == {TARGET}: {sorted(classical_solutions)}")


# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the classical solution basis states
# ---------------------------------------------------------------------------
# Register layout (6 qubits total): q0,q1,q2 = i (LSB first), q3,q4,q5 = j.

N_INDEX_BITS = 3
N_QUBITS = 2 * N_INDEX_BITS


def bits_of(value, n):
    return [(value >> k) & 1 for k in range(n)]


def mark_state(qc, qubits, value, n_bits, ancilla):
    """Apply a multi-controlled Z (phase flip) on qubits == value (n_bits)."""
    bits = bits_of(value, n_bits)
    flip = [qubits[k] for k, b in enumerate(bits) if b == 0]
    if flip:
        qc.x(flip)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    if flip:
        qc.x(flip)


def build_oracle(n_qubits, solutions):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for (i, j) in solutions:
        combined = i | (j << N_INDEX_BITS)  # pack (i,j) into one integer
        mark_state(qc, list(range(n_qubits)), combined, n_qubits, ancilla=None)
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


search_space_size = 2 ** N_QUBITS
num_solutions = len(classical_solutions)
# Optimal number of Grover iterations for this search-space/solution-count.
iterations = max(1, round((math.pi / 4) * math.sqrt(search_space_size / num_solutions)))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = build_oracle(N_QUBITS, classical_solutions)
diffuser = build_diffuser(N_QUBITS)

for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Decode each bitstring back to (i, j). Qiskit orders classical bits with the
# last qubit (highest index) as the leftmost character of the returned string.
def decode(bitstring):
    bits = bitstring[::-1]  # bits[k] corresponds to qubit k
    i = sum(int(bits[k]) << k for k in range(N_INDEX_BITS))
    j = sum(int(bits[N_INDEX_BITS + k]) << k for k in range(N_INDEX_BITS))
    return (i, j)

decoded_counts = {}
for bitstring, c in counts.items():
    pair = decode(bitstring)
    decoded_counts[pair] = decoded_counts.get(pair, 0) + c

top_pairs = sorted(decoded_counts.items(), key=lambda kv: -kv[1])[:num_solutions]
top_pairs_set = {pair for pair, _ in top_pairs}

print(f"Grover iterations used: {iterations}")
print("Top measured (i, j) pairs (most frequent first):")
for pair, c in top_pairs:
    freq = c / shots
    print(f"  (i={pair[0]}, j={pair[1]}) -> B[i]*B[j]-1 = "
          f"{B[pair[0]]*B[pair[1]]-1 if 0 <= pair[0] < len(B) and 0 <= pair[1] < len(B) else 'n/a'}"
          f"  count={c}  freq={freq:.3f}")

verified = top_pairs_set == classical_solutions

print()
print(f"Classical solution set : {sorted(classical_solutions)}")
print(f"Quantum top-{num_solutions} result : {sorted(top_pairs_set)}")

if verified:
    print("PASS")
else:
    print("FAIL")
