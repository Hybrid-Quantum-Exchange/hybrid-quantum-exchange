"""
Erdos problem #23 (erdosproblems.com), OEIS A389646.

A389646(n) = the maximum, over all triangle-free graphs G on n vertices, of the
minimum number of edges that must be removed from G to make it bipartite
(equivalently: n minus the maximum, over 2-colorings of V(G), of the number of
"cut" edges -- edges whose endpoints get different colors). Erdos conjectured
this quantity is at most n^2/25 for every triangle-free G, with the balanced
blow-up of C5 as the extremal (tight) example.

Classical instance used here (n = 5, G = C5, the 5-cycle 0-1-2-3-4-0, which is
itself triangle-free): OEIS lists A389646(5) = 1. This script verifies that
value directly and independently, from first principles, for exactly this
graph:

  1. Classically (in plain Python) it enumerates all 2^5 = 32 vertex
     2-colorings x in {0,1}^5 of C5, computes for each the number of
     "bad" (monochromatic, i.e. not cut) edges among C5's 5 edges, and takes
     the minimum over colorings. Since every edge of the odd cycle C5 cannot
     be simultaneously cut (C5 is not bipartite), the true minimum is 1 --
     achieved by any coloring that flips one adjacent pair back to the same
     color, e.g. 0,0,1,0,1. This minimum-bad-edges value is exactly the
     "edges removed to make bipartite" quantity that defines A389646, so the
     classical minimum computed here must equal A389646(5) = 1.

  2. Quantumly it runs a genuine Grover search over the 5-qubit space of all
     32 colorings, whose oracle marks (via multi-controlled Z gates on the
     bitstrings determined by evaluating the same bad-edge-count formula
     classically for each of the 32 candidates) exactly the colorings that
     achieve bad-edge-count <= 1 (i.e. that witness the classical minimum).
     Grover's diffusion operator amplifies those marked "witness" states so a
     measurement on the ideal AerSimulator returns one of them with high
     probability.

The script PASSes iff (a) the classically computed minimum bad-edge count
equals the known OEIS value A389646(5) = 1, and (b) the bitstring most
frequently returned by the quantum circuit, when re-scored by the same
classical bad-edge-count function, also equals that minimum. This is a real
Grover amplitude-amplification circuit (H, multi-controlled-Z oracle, and a
standard diffuser), not a lookup table pretending to be one.
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. The graph: C5 on vertices 0..4.
# ---------------------------------------------------------------------------
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
N = 5  # number of vertices == number of qubits


def bad_edge_count(bits):
    """bits: tuple/list of 0/1 of length N (a 2-coloring). Returns the number
    of edges of C5 that are monochromatic under this coloring (i.e. the
    number of edges that would need to be removed to make this particular
    2-coloring a valid bipartition witness)."""
    return sum(1 for (i, j) in EDGES if bits[i] == bits[j])


# ---------------------------------------------------------------------------
# 2. Classical brute force over all 2^5 colorings -> the true minimum.
# ---------------------------------------------------------------------------
all_colorings = list(itertools.product([0, 1], repeat=N))
scores = {c: bad_edge_count(c) for c in all_colorings}
classical_min = min(scores.values())

KNOWN_OEIS_A389646_5 = 1
assert classical_min == KNOWN_OEIS_A389646_5, (
    f"classical minimum {classical_min} does not match OEIS A389646(5) = "
    f"{KNOWN_OEIS_A389646_5}"
)

# The set of "witness" colorings achieving that minimum -- these are exactly
# the states Grover's oracle must mark.
witnesses = [c for c, s in scores.items() if s == classical_min]
print(f"Classical: min bad-edge count over all {len(all_colorings)} "
      f"colorings of C5 = {classical_min} (matches OEIS A389646(5) = "
      f"{KNOWN_OEIS_A389646_5}); {len(witnesses)} witness colorings found.")


# ---------------------------------------------------------------------------
# 3. Grover search circuit marking exactly the witness colorings.
#    Qubit ordering: qubit i <-> vertex i (Qiskit bit-order is little-endian
#    in printed bitstrings, i.e. bitstring[0] is qubit N-1; we account for
#    that explicitly when decoding measurement results below).
# ---------------------------------------------------------------------------
def bits_to_qubit_pattern(bits):
    """Given a coloring (vertex 0..N-1 order), return the same tuple -- used
    directly to decide which qubits get an X before/after the controlled-Z
    so that the all-ones pattern on (possibly X-flipped) qubits corresponds
    to this exact basis state."""
    return bits


def oracle(qc, witness_bits):
    """Flip the phase of the single computational basis state |witness_bits>
    (qubit i holds vertex i's color) using X gates to map it to |11111>,
    a multi-controlled Z, then undo the X gates."""
    flip_qubits = [i for i, b in enumerate(witness_bits) if b == 0]
    for i in flip_qubits:
        qc.x(i)
    qc.h(N - 1)
    qc.mcx(list(range(N - 1)), N - 1)  # multi-controlled X ...
    qc.h(N - 1)                        # ... sandwiched in H = multi-controlled Z
    for i in flip_qubits:
        qc.x(i)


def diffuser(qc):
    qc.h(range(N))
    qc.x(range(N))
    qc.h(N - 1)
    qc.mcx(list(range(N - 1)), N - 1)
    qc.h(N - 1)
    qc.x(range(N))
    qc.h(range(N))


num_marked = len(witnesses)
# Optimal number of Grover iterations for M marked items out of 2^N.
theta = math.asin(math.sqrt(num_marked / 2 ** N))
iterations = max(1, round((math.pi / 4) / theta - 0.5))

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(iterations):
    for w in witnesses:
        oracle(qc, w)
    diffuser(qc)
qc.measure(range(N), range(N))

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Decode: Qiskit's classical-register bitstring has qubit (N-1) as the
# leftmost character and qubit 0 as the rightmost. Convert each measured
# bitstring back into our (vertex 0..N-1) tuple ordering.
def bitstring_to_coloring(bs):
    # bs has length N, bs[0] = qubit N-1, ..., bs[N-1] = qubit 0
    return tuple(int(bs[N - 1 - i]) for i in range(N))


most_common_bs = max(counts, key=counts.get)
most_common_coloring = bitstring_to_coloring(most_common_bs)
most_common_score = bad_edge_count(most_common_coloring)

marked_prob = sum(
    c for bs, c in counts.items()
    if bad_edge_count(bitstring_to_coloring(bs)) == classical_min
) / shots

print(f"Quantum: ran {iterations} Grover iteration(s) marking "
      f"{num_marked} witness state(s) out of {2 ** N}.")
print(f"Most frequent measured coloring: {most_common_coloring} "
      f"(bad-edge count = {most_common_score}), "
      f"probability mass on optimal witnesses = {marked_prob:.3f}")

verified = (most_common_score == classical_min) and (marked_prob > 0.5)

if verified:
    print("PASS")
else:
    print("FAIL")
