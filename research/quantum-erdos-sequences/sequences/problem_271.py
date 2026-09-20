"""
Erdos problem #271 -- quantum-testable instance.

Source: erdosproblems.com problem 271 ("Stanley sequences"), metadata pulled
from data/problems.yaml: oeis = ["A005487"], tags = ["additive combinatorics",
"arithmetic progressions"], comments = "Stanley sequences".

OEIS A005487 is the (canonical, greedy) Stanley sequence:
    a(0) = 0, a(1) = 1
    a(n) = smallest integer > a(n-1) such that no three terms of
           {a(0), ..., a(n)} form a 3-term arithmetic progression (AP).
Its first terms are 0, 1, 3, 4, 9, 10, 12, 13, 27, ... -- note 2 is skipped
because 0, 1, 2 is a 3-term AP.

Classical property tested (computed from first principles below, not copied
from OEIS): given the sequence prefix S = {0, 1}, and the search space
X = {0, 1, ..., 7} (3 bits), exactly one element x in X completes a 3-term
AP with the two existing terms 0 and 1 -- namely x = 2, because
2*1 - 0 = 2 (the only affine relation a - b = c - a with {a, b} subset of
{0, 1} and c in X, c not already in S, that has a solution in X). This is
exactly the classical fact that forces A005487 to skip 2 and place its next
term at 3.

Quantum computation: a genuine Grover search over the 3-qubit space
{0, ..., 7} whose oracle marks precisely the AP-completing values (i.e.
x such that 0, 1, x is a 3-term AP). With N = 8 and a single marked
state, the Grover-optimal number of iterations is floor(pi/4 * sqrt(8)) = 2.
Running the circuit on the ideal AerSimulator should recover x = 2 as the
overwhelmingly most likely measurement outcome. The script then checks this
quantum result against the classical answer computed independently above,
and also double-checks by explicit brute-force search over X (not just the
single algebraic relation) that x = 2 is indeed the unique AP-completing
element, and that 3 is therefore the correct next Stanley-sequence term.

PASS iff:
  (a) brute-force classical search agrees that x=2 is the unique element of
      {0,...,7} completing a 3-term AP with {0,1}, and 3 is the next
      admissible term of the Stanley sequence; and
  (b) the Grover circuit's most-frequent measured outcome equals 2.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical part: brute-force, from first principles.
# ---------------------------------------------------------------------------

def is_three_term_ap(a, b, c):
    """True iff {a, b, c} (as a set of 3 distinct integers) contains a
    3-term arithmetic progression, i.e. some ordering x, y, z with
    y - x == z - y."""
    vals = sorted((a, b, c))
    x, y, z = vals
    return (y - x) == (z - y)


def completes_ap(prefix, x):
    """True iff adding x to `prefix` creates a 3-term AP among the
    resulting set (checking every triple that includes x)."""
    if x in prefix:
        return False
    pts = list(prefix) + [x]
    for a, b, c in combinations(pts, 3):
        if x in (a, b, c) and is_three_term_ap(a, b, c):
            return True
    return False


def stanley_next_term(prefix, search_space):
    """Smallest x in search_space, x > max(prefix), that does not complete
    a 3-term AP with prefix."""
    candidates = sorted(v for v in search_space if v > max(prefix))
    for x in candidates:
        if not completes_ap(prefix, x):
            return x
    raise ValueError("no admissible next term found in search space")


PREFIX = (0, 1)          # S = {a(0), a(1)} of A005487
SEARCH_SPACE = list(range(8))  # 3-qubit space, N = 8

# Brute-force over the whole search space: which x complete a 3-term AP?
ap_completing = [x for x in SEARCH_SPACE if completes_ap(PREFIX, x)]
assert ap_completing == [2], (
    f"expected the unique AP-completing value to be 2, got {ap_completing}"
)
MARKED = ap_completing[0]  # = 2, this is what the Grover oracle should find

next_term = stanley_next_term(PREFIX, SEARCH_SPACE)
assert next_term == 3, f"expected next Stanley term 3, got {next_term}"

print(f"Classical check: within X={SEARCH_SPACE}, prefix S={PREFIX}, "
      f"the unique AP-completing value is x={MARKED}.")
print(f"Classical check: next Stanley-sequence term (A005487) after "
      f"{PREFIX} restricted to X is {next_term} (matches known term).")


# ---------------------------------------------------------------------------
# 2. Quantum part: Grover search for the marked value over 3 qubits.
# ---------------------------------------------------------------------------

N_QUBITS = 3
N = 2 ** N_QUBITS  # 8
assert MARKED < N


def marked_bits(value, n_qubits):
    """Little-endian bit list (qubit 0 = LSB) of `value`."""
    return [(value >> i) & 1 for i in range(n_qubits)]


def build_oracle(n_qubits, marked_value):
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = marked_bits(marked_value, n_qubits)
    zero_qubits = [i for i, b in enumerate(bits) if b == 0]
    for q in zero_qubits:
        qc.x(q)
    # multi-controlled Z on all qubits (phase flip on |11...1>)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for q in zero_qubits:
        qc.x(q)
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


oracle = build_oracle(N_QUBITS, MARKED)
diffuser = build_diffuser(N_QUBITS)

iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(N / 1))))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Bitstrings from Qiskit are big-endian in the classical register order
# (c[n-1] ... c[0]); since we measured qubit i into clbit i, reverse to get
# little-endian and convert to an integer matching our `marked_bits` scheme.
def bitstring_to_int(bs):
    return int(bs[::-1], 2)

int_counts = {}
for bitstring, cnt in counts.items():
    val = bitstring_to_int(bitstring)
    int_counts[val] = int_counts.get(val, 0) + cnt

most_likely = max(int_counts, key=int_counts.get)
prob_most_likely = int_counts[most_likely] / shots

print(f"Quantum Grover search ({iterations} iteration(s), {shots} shots) "
      f"over {N} candidates found most-likely outcome x={most_likely} "
      f"with probability {prob_most_likely:.3f}.")
print(f"Full outcome distribution: {dict(sorted(int_counts.items()))}")


# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

verified = (most_likely == MARKED) and (prob_most_likely > 0.5)

if verified:
    print("PASS")
else:
    print("FAIL")
