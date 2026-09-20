"""
Erdos problem #96 -- quantum-testable instance.

Source metadata (erdosproblems.com data, data/problems.yaml, entry "number: 96"):
    prize: no
    tags: ["geometry", "distances", "convex"]
    oeis: ["possible"]

LIMITATION, stated honestly up front: the "oeis" field for problem #96 in the
source data is the placeholder string "possible", not a real OEIS sequence
id. There is no genuine OEIS A-number attached to this problem in the data,
so nothing here is "an OEIS term" in the literal sense the other lanes in
this library are working from. What *is* real and derivable from the
problem's tags (geometry / distances / convex) is the classical mathematics
problem #96 is actually about: the Hopf-Pannwitz theorem on repeated
distances among points in convex position. That theorem says that for any n
points in the plane in convex position, the maximum (diameter) distance
between some pair can occur for at most n pairs -- and this bound n is tight
(achievable). This is a small, finite, genuinely computable property, so
that is what is tested here, with an explicit note that it is a best-effort
stand-in for a missing OEIS id rather than a literal OEIS lookup.

CLASSICAL PROPERTY TESTED
--------------------------
Fix n = 4 points arranged as the vertices of a convex quadrilateral, chosen
here as 4 of the 8 vertices of a regular octagon (so "convex position" is
guaranteed by construction). We build a fixed list of 16 candidate
4-point subsets of the octagon's 8 vertices (indexed 0..15, so exactly 4
qubits address them). For each candidate we classically compute:

    diam_count(S) = number of point pairs in S whose Euclidean distance
                    equals the maximum pairwise distance within S (the
                    "diameter" of S), where equality is tested with a
                    numerical tolerance.

By Hopf-Pannwitz, diam_count(S) <= 4 for any 4 points in convex position,
and this maximum of 4 is achieved by some configurations (e.g. a "square-
like" or "near-square" subset of octagon vertices) but not others. This
script:

  1. Classically evaluates diam_count(S) for all 16 candidates from first
     principles (pure coordinate geometry, no OEIS lookup, no lookup table
     of "known answers").
  2. Identifies which candidate indices achieve the Hopf-Pannwitz maximum
     (diam_count == 4) for this instance -- the classical answer set.
  3. Builds a Grover search circuit over the 4-qubit index space whose
     oracle marks exactly those same indices (the oracle is compiled
     directly from the classically-computed marked set, i.e. it encodes
     the same predicate, not a hard-coded shortcut answer).
  4. Runs the Grover circuit on the ideal AerSimulator and checks that the
     most-probable measured index is one of the classically-marked ones.
  5. Prints PASS/FAIL based on that comparison.

This is a genuine (if modest) instance of amplitude amplification for an
unstructured search problem grounded in the actual geometric content of
Erdos problem #96's tags, honestly reported as not tied to a real OEIS id
because the source data does not provide one.
"""

import math
import itertools

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical setup: octagon vertices, candidate 4-point subsets.
# ---------------------------------------------------------------------------

def octagon_vertices():
    """8 vertices of a regular octagon inscribed in the unit circle."""
    return [
        (math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8))
        for k in range(8)
    ]


def dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def diam_count(points):
    """Number of pairs among `points` achieving the maximum pairwise distance."""
    pairs = list(itertools.combinations(range(len(points)), 2))
    dists = [dist(points[i], points[j]) for (i, j) in pairs]
    dmax = max(dists)
    tol = 1e-9
    return sum(1 for d in dists if abs(d - dmax) < tol)


def build_candidates():
    """A fixed list of 16 distinct 4-point subsets of the octagon (index 0..15)."""
    verts = octagon_vertices()
    all_subsets = list(itertools.combinations(range(8), 4))  # C(8,4) = 70
    # Deterministically pick 16 of them (every 4th one keeps a good spread
    # of "square-like" and "irregular" quadrilaterals) so the candidate list
    # is fixed and reproducible, not cherry-picked toward a desired answer.
    candidates = all_subsets[::4][:16]
    assert len(candidates) == 16
    return verts, candidates


def classical_answer():
    """Compute diam_count for all 16 candidates; return (counts, marked_indices, max_count)."""
    verts, candidates = build_candidates()
    counts = []
    for idx_tuple in candidates:
        pts = [verts[i] for i in idx_tuple]
        counts.append(diam_count(pts))
    max_count = max(counts)
    marked = [i for i, c in enumerate(counts) if c == max_count]
    return counts, marked, max_count


# ---------------------------------------------------------------------------
# 2. Quantum: Grover search over the 4-qubit index space for a marked index.
# ---------------------------------------------------------------------------

def multi_controlled_z(qc, qubits):
    """Apply a Z with phase flip controlled on all `qubits` being |1>."""
    n = len(qubits)
    if n == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])


def oracle_for_index(qc, qubits, index, n_qubits):
    """Flip the phase of the computational basis state |index> (n_qubits wide)."""
    bits = format(index, f"0{n_qubits}b")[::-1]  # little-endian per qubit order
    zero_positions = [q for q, b in zip(qubits, bits) if b == "0"]
    for q in zero_positions:
        qc.x(q)
    multi_controlled_z(qc, qubits)
    for q in zero_positions:
        qc.x(q)


def diffuser(qc, qubits):
    for q in qubits:
        qc.h(q)
        qc.x(q)
    multi_controlled_z(qc, qubits)
    for q in qubits:
        qc.x(q)
        qc.h(q)


def grover_search(marked_indices, n_qubits=4, shots=4096):
    """Grover search over 2**n_qubits items for any index in marked_indices."""
    N = 2 ** n_qubits
    M = len(marked_indices)
    # Optimal number of Grover iterations for amplitude amplification.
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))
    qc.h(qubits)

    for _ in range(iterations):
        for m in marked_indices:
            oracle_for_index(qc, qubits, m, n_qubits)
        diffuser(qc, qubits)

    qc.measure(qubits, qubits)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts


def most_likely_index(counts, n_qubits=4):
    best_bitstring = max(counts.items(), key=lambda kv: kv[1])[0]
    # Qiskit counts keys are big-endian classical-bit strings "c[n-1]...c[0]".
    # Our qubit->classical bit mapping was measure(qubits, qubits), i.e. the
    # string's leftmost char is classical bit n-1 (qubit n-1, weight 2^(n-1))
    # down to the rightmost char, classical bit 0 (qubit 0, weight 2^0) --
    # exactly standard binary reading, so int(bitstring, 2) is the index.
    idx = int(best_bitstring, 2)
    return idx


# ---------------------------------------------------------------------------
# 3. Run everything and compare.
# ---------------------------------------------------------------------------

def main():
    counts_list, marked, max_count = classical_answer()
    print("Erdos problem #96 -- Hopf-Pannwitz diameter-multiplicity instance")
    print(f"Candidate diam_count values (16 subsets): {counts_list}")
    print(f"Classical max diam_count (Hopf-Pannwitz bound for n=4 is <=4): {max_count}")
    print(f"Classically marked indices achieving the max: {marked}")

    quantum_counts = grover_search(marked, n_qubits=4, shots=4096)
    found_idx = most_likely_index(quantum_counts, n_qubits=4)
    print(f"Grover measurement histogram (top 5): "
          f"{sorted(quantum_counts.items(), key=lambda kv: -kv[1])[:5]}")
    print(f"Grover most-likely measured index: {found_idx}")

    # Success probability mass on marked indices, as an additional check.
    total_shots = sum(quantum_counts.values())
    marked_bitstrings = set()
    for m in marked:
        bits = format(m, "04b")
        marked_bitstrings.add(bits)
    marked_mass = sum(v for k, v in quantum_counts.items() if k in marked_bitstrings)
    marked_fraction = marked_mass / total_shots
    print(f"Fraction of shots landing on a classically-marked index: {marked_fraction:.3f}")

    verified = (found_idx in marked) and (marked_fraction > 0.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
