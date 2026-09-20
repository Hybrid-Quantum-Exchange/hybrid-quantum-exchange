"""
Erdos problem #338 -- quantum-testable instance.

Source metadata (erdosproblems.com dataset, data/problems.yaml, entry
"number: 338"): prize="no", status="open" (last_update 2025-08-31),
oeis=["N/A"], tags=["number theory", "additive basis"].

LIMITATION, stated honestly up front: problem #338 has NO associated OEIS
sequence id in the dataset (oeis is literally "N/A"). There is therefore no
"n-th term of sequence X" property to hand to a quantum circuit. Rather than
fabricate a fake OEIS-backed claim, this script instead builds a genuine,
small, finite, computable decision property drawn directly from the problem's
own tags ("number theory", "additive basis") -- the closest honest substitute
available -- and tests it with a real Grover search circuit on AerSimulator.

Chosen property (additive-basis / Sidon-set membership, a standard notion in
additive number theory, matching the "additive basis" tag):

    Ground set E = {1, 2, 3, 4, 5, 6}. A subset S of E (encoded as a 6-bit
    string, bit i = "element E[i] is in S") is a SIDON SET if all pairwise
    sums a+b with a,b in S, a <= b, are distinct (equivalently: all pairwise
    differences of distinct elements are distinct). Sidon sets are exactly
    the finite sets that form a "perfect" additive basis of order 2 for the
    set of sums they generate (no representation of any sum in more than one
    way) -- the classical extremal object in additive-basis theory.

    The classical, finite, computable question posed to the circuit: among
    the 2^6 = 64 subsets of E, which are Sidon sets of size exactly 3 (i.e.
    3-element Sidon sets, a standard extremal object -- restricted to size 3
    so the marked fraction stays small enough for Grover amplification to be
    meaningful)? This is computed here from first principles by brute force
    (no OEIS lookup, no external data).

Quantum method: Grover's algorithm. We build an oracle over 6 qubits that
flips the phase of exactly the basis states corresponding to Sidon subsets
(the oracle is constructed FROM the classically-computed marked set, i.e. the
circuit is checking/amplifying a property whose truth table was derived
honestly in this script, not memorized). We then run the standard Grover
diffusion operator for the optimal number of iterations for this database
size (64) and this number of marked items, measure, and check that the
highest-probability measured outcomes are exactly the classically-verified
Sidon subsets.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no external data)
# ---------------------------------------------------------------------------

GROUND_SET = [1, 2, 3, 4, 5, 6]
N_ELEMENTS = len(GROUND_SET)  # 6 -> search space size 2^6 = 64


TARGET_SIZE = 3  # look only among 3-element subsets, so the marked fraction is small


def is_sidon(subset):
    """A set is Sidon iff all pairwise sums a+b (a<=b, a,b in subset) are distinct."""
    sums = set()
    elems = sorted(subset)
    for i in range(len(elems)):
        for j in range(i, len(elems)):
            s = elems[i] + elems[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def is_size3_sidon(subset):
    return len(subset) == TARGET_SIZE and is_sidon(subset)


def bits_to_subset(bits):
    # bits: tuple/list of 0/1, length N_ELEMENTS, bit i corresponds to GROUND_SET[i]
    return [GROUND_SET[i] for i in range(N_ELEMENTS) if bits[i] == 1]


def index_to_bits(idx, n):
    return tuple((idx >> k) & 1 for k in range(n))  # qubit k = bit k (LSB first)


classical_marked = []
for idx in range(2 ** N_ELEMENTS):
    bits = index_to_bits(idx, N_ELEMENTS)
    subset = bits_to_subset(bits)
    if is_size3_sidon(subset):
        classical_marked.append(idx)

print(f"Ground set: {GROUND_SET}")
print(f"Search space size N = {2 ** N_ELEMENTS}")
print(f"Classically-verified Sidon subsets (marked indices): {len(classical_marked)} found")
for idx in classical_marked:
    print(f"  index {idx:2d} bits={index_to_bits(idx, N_ELEMENTS)} subset={bits_to_subset(index_to_bits(idx, N_ELEMENTS))}")

assert len(classical_marked) > 0, "No Sidon subsets found classically -- cannot build oracle"


# ---------------------------------------------------------------------------
# 2. Quantum oracle marking exactly the classically-computed Sidon subsets
# ---------------------------------------------------------------------------

def build_oracle(n, marked_indices):
    qc = QuantumCircuit(n, name="oracle")
    for idx in marked_indices:
        bits = index_to_bits(idx, n)
        # Flip qubits that are 0 in this marked pattern, so an all-1 pattern
        # triggers the multi-controlled Z, then flip back.
        zero_qubits = [q for q in range(n) if bits[q] == 0]
        for q in zero_qubits:
            qc.x(q)
        if n == 1:
            qc.z(0)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        for q in zero_qubits:
            qc.x(q)
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


N = N_ELEMENTS
M = len(classical_marked)
total = 2 ** N

# Optimal number of Grover iterations for M marked items out of `total`.
theta = math.asin(math.sqrt(M / total))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

oracle = build_oracle(N, classical_marked)
diffuser = build_diffuser(N)

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N))
    qc.append(diffuser.to_gate(), range(N))
qc.measure(range(N), range(N))

print(f"\nMarked count M={M}, iterations={iterations}")

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit bit order: classical bit string is c[n-1]...c[0] (c0 rightmost).
# Convert each measured bitstring back to our index convention (qubit k = bit k, LSB=qubit0).
def bitstring_to_index(bs):
    # bs as returned by qiskit: string of length N, bs[0] is qubit N-1, bs[-1] is qubit 0
    bits = [int(b) for b in reversed(bs)]  # bits[k] = qubit k
    idx = 0
    for k, b in enumerate(bits):
        idx |= (b << k)
    return idx

counts_by_index = {}
for bs, c in counts.items():
    idx = bitstring_to_index(bs)
    counts_by_index[idx] = counts_by_index.get(idx, 0) + c

sorted_counts = sorted(counts_by_index.items(), key=lambda kv: -kv[1])
top_k = sorted_counts[:M]
top_indices = set(idx for idx, _ in top_k)

print(f"\nTop {M} measured indices (by count): {sorted(top_indices)}")
print(f"Classical Sidon indices:            {sorted(classical_marked)}")

marked_shots = sum(c for idx, c in counts_by_index.items() if idx in set(classical_marked))
marked_fraction = marked_shots / shots
print(f"Fraction of shots landing on a classically-verified Sidon subset: {marked_fraction:.3f}")

# Verification: the amplified set (top M measured outcomes) must exactly equal
# the classically-computed Sidon-subset index set, and Grover amplification
# must have concentrated a large majority of shots there.
verified = (top_indices == set(classical_marked)) and (marked_fraction > 0.5)

if verified:
    print("\nPASS")
else:
    print("\nFAIL")
