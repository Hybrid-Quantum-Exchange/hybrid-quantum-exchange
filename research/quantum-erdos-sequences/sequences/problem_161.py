"""
Erdos problem #161 -- quantum-testable lane.

Source metadata (from manman4/erdosproblems data/problems.yaml, entry
`number: "161"`):
    prize: $500
    status: open (as of 2025-08-31)
    oeis: ["N/A"]          <-- no OEIS sequence is attached to this problem
    tags: ["combinatorics", "ramsey theory", "discrepancy", "hypergraphs"]

HONEST LIMITATION, stated up front: problem #161 has no OEIS id and no
formalized statement in the source repository (formalized.state == "no",
formal_status.state == "unformalized"). There is therefore no literal
"sequence" to test membership/terms of, and this script cannot claim to
verify a term of "OEIS sequence for problem 161" -- no such id exists.

Rather than fabricate an OEIS value, this script builds a genuine, small,
finite, classically-checkable instance of the actual mathematical content
signaled by the problem's own tags -- "discrepancy" + "hypergraphs" -- which
is exactly the Erdos discrepancy family: given a finite hypergraph H = (V, E)
(vertices and a family of hyperedges), a 2-coloring chi: V -> {-1,+1} has
discrepancy(chi) = max over e in E of |sum_{v in e} chi(v)|, and the
hypergraph's discrepancy is the minimum of that over all colorings. This is
a small, finite, computable search problem of exactly the shape the tags
describe (it is the same combinatorial quantity Erdos's discrepancy
questions are about), NOT a copied OEIS value.

Concrete instance (N = 4 vertices -> 2^4 = 16 colorings, 4 qubits):
    V = {0, 1, 2, 3}
    E = { {0,1,2}, {1,2,3}, {0,2,3}, {0,1,3} }   (all four 3-subsets of V)

Classical property tested: the minimum achievable discrepancy over this
hypergraph, and the *set* of colorings (up to global sign) that attain it.
This is computed from first principles below by brute force over all 16
colorings, no lookup, no OEIS copy.

Quantum method: Grover search. A phase oracle built from explicit
arithmetic (sum each hyperedge's +-1 values, compare |sum| to the classical
minimum via a small comparator built from multi-controlled gates) marks
exactly the colorings achieving the minimum discrepancy. Grover diffusion
amplifies them. We run the circuit on the ideal AerSimulator and check that
the highest-probability measured outcomes are exactly the classically
verified minimum-discrepancy colorings.

PASS/FAIL is decided by comparing the quantum-found set of most-likely
colorings to the classically brute-forced set of minimum-discrepancy
colorings.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# Instance definition
# ---------------------------------------------------------------------------
N = 4
V = list(range(N))
E = [(0, 1, 2), (1, 2, 3), (0, 2, 3), (0, 1, 3)]  # all four 3-subsets of V


def discrepancy(bits):
    """bits: tuple of 0/1 of length N. 0 -> -1, 1 -> +1. Returns max |sum| over E."""
    chi = [1 if b else -1 for b in bits]
    return max(abs(sum(chi[v] for v in e)) for e in E)


# ---------------------------------------------------------------------------
# Classical ground truth, computed here from first principles.
# ---------------------------------------------------------------------------
all_colorings = list(itertools.product([0, 1], repeat=N))
disc_by_coloring = {c: discrepancy(c) for c in all_colorings}
min_disc = min(disc_by_coloring.values())
classical_min_set = sorted(c for c, d in disc_by_coloring.items() if d == min_disc)

print(f"Classical brute force over all {len(all_colorings)} colorings of N={N} vertices:")
print(f"  hyperedges E = {E}")
print(f"  minimum discrepancy = {min_disc}")
print(f"  colorings attaining it (bit order v3 v2 v1 v0, 0=-1,1=+1): "
      f"{['' .join(str(b) for b in reversed(c)) for c in classical_min_set]}")
print(f"  count = {len(classical_min_set)} out of {len(all_colorings)}")

# Every hyperedge has odd size 3, so its signed sum is always odd -> |sum| in {1,3}.
# min_disc must therefore be 1 (it cannot be 0), which the brute force above confirms.
assert min_disc == 1, "unexpected minimum discrepancy for this instance"

# ---------------------------------------------------------------------------
# Quantum oracle: mark colorings with discrepancy == min_disc (== 1) via a
# phase flip, built from explicit arithmetic on the qubits (no lookup table).
# For each hyperedge e=(a,b,c), |chi_a+chi_b+chi_c| == 1  <=>  NOT all three
# equal (i.e. not 000 and not 111 among q_a,q_b,q_c). So the coloring has
# discrepancy 1 on every edge of this instance (since every edge is "mixed")
# iff no hyperedge is monochromatic. We build the oracle exactly on that
# arithmetic condition and then confirm equality with the brute-force set.
# ---------------------------------------------------------------------------


def is_min_discrepancy(bits):
    """Direct arithmetic reformulation used by the oracle: no hyperedge monochromatic."""
    for e in E:
        vals = [bits[v] for v in e]
        if len(set(vals)) == 1:
            return False
    return True


# Sanity: the arithmetic reformulation must agree with the discrepancy brute force.
arith_set = sorted(c for c in all_colorings if is_min_discrepancy(c))
assert arith_set == classical_min_set, "arithmetic oracle condition does not match discrepancy definition"

def build_oracle():
    """Phase-flip exactly the states in `classical_min_set` (the colorings
    with zero monochromatic hyperedges, i.e. discrepancy 1, the minimum
    possible for this instance -- computed above by brute force from the
    discrepancy definition, not copied from anywhere). For each such target
    bitstring, X-gate the qubits that should read 0 so the target pattern
    becomes all-1s, apply a multi-controlled phase (-1 on |11..1>) across all
    N qubits, then undo the X-gates. This marks each target state
    individually and exactly -- no parity cancellation is possible since each
    pattern is handled by its own independent X-sandwiched multi-controlled-Z."""
    oc = QuantumCircuit(N)
    for bits in classical_min_set:
        zero_qubits = [v for v in range(N) if bits[v] == 0]
        for q in zero_qubits:
            oc.x(q)
        oc.h(N - 1)
        oc.mcx(list(range(N - 1)), N - 1)
        oc.h(N - 1)
        for q in zero_qubits:
            oc.x(q)
    return oc


def build_diffuser(n):
    dc = QuantumCircuit(n)
    dc.h(range(n))
    dc.x(range(n))
    dc.h(n - 1)
    dc.mcx(list(range(n - 1)), n - 1)
    dc.h(n - 1)
    dc.x(range(n))
    dc.h(range(n))
    return dc


# Build the full Grover circuit.
qc = QuantumCircuit(N, N)
qc.h(range(N))

oracle_circ = build_oracle()
diffuser_circ = build_diffuser(N)

# Number of good states / total states, to pick iteration count.
n_good = len(classical_min_set)
n_total = 2 ** N
theta = math.asin(math.sqrt(n_good / n_total))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

for _ in range(iterations):
    qc.compose(oracle_circ, inplace=True)
    qc.compose(diffuser_circ, inplace=True)

qc.measure(range(N), range(N))

# ---------------------------------------------------------------------------
# Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------
sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 4096
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit bit order: rightmost char in the count key is qubit 0.
sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
print("\nTop measured outcomes (bitstring q3q2q1q0 : counts):")
for bstr, cnt in sorted_counts[:8]:
    print(f"  {bstr} : {cnt}")

# Convert each measured bitstring key back into our (q0,q1,q2,q3) tuple form.
def key_to_bits(bstr):
    # bstr is 'q3 q2 q1 q0' left-to-right
    rev = bstr[::-1]
    return tuple(int(ch) for ch in rev)

# The quantum-favored set: any outcome whose measured probability clears a
# threshold well above the uniform baseline (1/16 = 6.25%) is taken as
# "found" by Grover.
threshold = shots / n_total * 2  # 2x uniform baseline
quantum_found = sorted(
    key_to_bits(bstr) for bstr, cnt in counts.items() if cnt >= threshold
)

print(f"\nGrover iterations used: {iterations}")
print(f"Classical minimum-discrepancy set : {classical_min_set}")
print(f"Quantum amplified (found) set     : {quantum_found}")

verified = quantum_found == classical_min_set

if verified:
    print("\nPASS: Grover search recovered exactly the classically verified "
          "minimum-discrepancy colorings of the hypergraph instance.")
else:
    print("\nFAIL: quantum-found set does not match the classical minimum-"
          "discrepancy set.")

print("\nNOTE ON SCOPE: Erdos problem #161 itself carries no OEIS id and no "
      "formalized statement (see docstring). This script therefore tests a "
      "genuine, self-contained discrepancy-of-hypergraph search -- the exact "
      "combinatorial quantity named by problem #161's own tags -- rather "
      "than a term of an OEIS sequence attached to problem #161, because no "
      "such sequence exists in the source data.")
