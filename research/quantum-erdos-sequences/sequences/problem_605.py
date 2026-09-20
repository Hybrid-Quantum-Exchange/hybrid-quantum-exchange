"""
Erdos problem #605 -- quantum-testable instance
=================================================

Source metadata (erdosproblems.com data, from the read-only clone at
/home/user/manman4/erdosproblems/data/problems.yaml, entry "number: 605"):

    prize: no
    status: proved (as of 2025-08-31)
    oeis: ["possible"]
    tags: ["geometry", "distances"]

Honesty note on the OEIS id
----------------------------
Problem #605's `oeis` field in the source data is the literal placeholder
string "possible" -- there is no real OEIS sequence id attached to this
problem in the source file. So this script does NOT use an OEIS sequence.
Instead it uses the problem's *tags* ("geometry", "distances"), which
identify it as an instance of the Erdos distinct-distances family: given a
finite point set, how few distinct pairwise distances can it determine.
That is a genuine, small, finite, computable combinatorial-geometry
property, well suited to a real Grover search circuit, and it is the most
faithful thing this script can honestly build given the source metadata.

Classical property being tested
--------------------------------
Fix the 3x3 integer grid of 9 points {(x, y) : x, y in {0, 1, 2}}.
Consider all C(9, 4) = 126 ways to choose an unordered 4-point subset.
For each subset, compute the number of *distinct* squared Euclidean
pairwise distances among its 6 pairs (using squared distances keeps
everything integer / exact -- no floating point).

    f(subset) = |{ dist2(p, q) : {p, q} subset of the 4 points }|

The classical property under test is:

    "Which 4-point subsets of the 3x3 grid minimize f(subset), and what
     is that minimum value m*?"

This is computed here from first principles by brute force over all 126
subsets (see `classical_solve()` below) -- nothing is copied from OEIS or
any external table. The brute-force computation finds m* = 2 (e.g. any
axis-aligned unit square: 4 unit-length sides + 2 equal diagonals -> only
2 distinct squared distances), achieved by exactly 6 of the 126 subsets.

Quantum circuit
----------------
The 126 subsets are indexed 0..125 and packed into 7 qubits (2^7 = 128,
with indices 126 and 127 left as unused/never-marked "padding" states).
A Grover search is built whose oracle phase-flips exactly the basis
states corresponding to the minimizing subsets (the marked set found by
`classical_solve()`), and whose diffuser is the standard 7-qubit Grover
diffusion operator. The number of Grover iterations is chosen from the
standard formula floor((pi/4) * sqrt(N / M)) using N = 128, M = number of
marked states.

The circuit is run on the ideal Qiskit Aer simulator. The test passes if
Grover search amplifies the marked (minimizing) subsets so that sampling
the final state returns, with high probability, an index that is in the
classically-computed marked set.
"""

from itertools import combinations

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Classical part: brute-force solve the distinct-distances minimization over
# all 4-point subsets of the 3x3 integer grid. Nothing here is looked up from
# OEIS or any external source -- it is computed directly from the geometry.
# ---------------------------------------------------------------------------

def classical_solve():
    pts = [(x, y) for x in range(3) for y in range(3)]  # 9 points
    subset_list = list(combinations(range(9), 4))  # 126 subsets, index 0..125
    assert len(subset_list) == 126

    def dist2(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    counts = []
    for s in subset_list:
        ds = set()
        for a, b in combinations(s, 2):
            ds.add(dist2(pts[a], pts[b]))
        counts.append(len(ds))

    m_star = min(counts)
    marked = [i for i, c in enumerate(counts) if c == m_star]
    return subset_list, counts, m_star, marked


SUBSETS, COUNTS, M_STAR, MARKED = classical_solve()
N_QUBITS = 7  # 2**7 = 128 >= 126
N_STATES = 2 ** N_QUBITS

print(f"Classical brute force over all {len(SUBSETS)} 4-point subsets of the 3x3 grid:")
print(f"  minimum number of distinct pairwise squared distances m* = {M_STAR}")
print(f"  achieved by {len(MARKED)} subsets, indices = {MARKED}")


# ---------------------------------------------------------------------------
# Quantum part: Grover search over the 7-qubit index register for the marked
# (distance-minimizing) subset indices.
# ---------------------------------------------------------------------------

def bits_of(index, n):
    """Little-endian bit list (qubit 0 = least significant bit)."""
    return [(index >> b) & 1 for b in range(n)]


def apply_marking_oracle(qc, index, n):
    """Phase-flip the single computational basis state |index> on an n-qubit
    register, using the standard X-sandwiched multi-controlled-Z trick."""
    bits = bits_of(index, n)
    flip_qubits = [q for q, b in enumerate(bits) if b == 0]
    for q in flip_qubits:
        qc.x(q)
    # multi-controlled Z on all n qubits: H on target, MCX, H on target
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    for q in flip_qubits:
        qc.x(q)


def build_oracle(marked_indices, n):
    qc = QuantumCircuit(n, name="oracle")
    for idx in marked_indices:
        apply_marking_oracle(qc, idx, n)
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


def grover_iterations(N, M):
    import math
    if M <= 0:
        return 0
    theta = math.asin(math.sqrt(M / N))
    k = math.floor((math.pi / 4) / theta - 0.5)
    return max(1, k)


def build_grover_circuit(marked_indices, n):
    oracle = build_oracle(marked_indices, n)
    diffuser = build_diffuser(n)
    iters = grover_iterations(2 ** n, len(marked_indices))

    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iters):
        qc.append(oracle.to_instruction(), range(n))
        qc.append(diffuser.to_instruction(), range(n))
    qc.measure(range(n), range(n))
    return qc.decompose(), iters


def main():
    qc, iters = build_grover_circuit(MARKED, N_QUBITS)
    print(f"Grover circuit: {N_QUBITS} qubits, {N_STATES} basis states, "
          f"{len(MARKED)} marked states, {iters} Grover iteration(s).")

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical-register string is "c[n-1]...c[0]" (leftmost char is
    # the highest-index bit), which is exactly the MSB-first encoding of our
    # little-endian `bits_of` index (qubit b = bit b of the index), so a
    # plain binary parse recovers the index directly.
    def bitstring_to_index(bs):
        return int(bs, 2)

    total_marked_hits = 0
    best_bs, best_count = None, -1
    for bs, c in counts.items():
        if c > best_count:
            best_bs, best_count = bs, c
        idx = bitstring_to_index(bs)
        if idx in MARKED:
            total_marked_hits += c

    most_likely_index = bitstring_to_index(best_bs)
    marked_fraction = total_marked_hits / shots

    print(f"Most frequently sampled index: {most_likely_index} "
          f"(count {best_count}/{shots})")
    print(f"Fraction of shots landing on a marked (minimizing) index: "
          f"{marked_fraction:.3f}")

    verified = (most_likely_index in MARKED) and (marked_fraction > 0.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
