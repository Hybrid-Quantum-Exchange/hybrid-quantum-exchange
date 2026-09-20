"""
Erdos problem #953 -- quantum-testable lane.

Source metadata (from data/problems.yaml, manman4/erdosproblems, read-only
clone): number "953", tags ["geometry", "distances"], oeis: ["N/A"] (no OEIS
sequence is attached to this problem), prize "no", status "open".

LIMITATION, stated honestly up front: because problem 953 carries no OEIS id,
there is no literal integer sequence to test membership/terms against. There
is therefore no way to "verify against OEIS" here, and this script does not
pretend to. What *is* available, and genuinely mathematical, is the problem's
subject matter: Erdos-style distinct-distances questions in the plane -- given
a finite point set, how many distinct pairwise Euclidean distances does it
realize, and can a point set be found that minimizes (or matches a target)
that count. That is a small, finite, exactly-computable combinatorial search,
which is what this script turns into a real quantum circuit.

Classical property tested
--------------------------
Fix the 3x3 integer grid {0,1,2} x {0,1,2} (9 points). Enumerate a fixed list
of 8 specific 4-point subsets of that grid (chosen up front, indexed 0..7).
For each subset, compute classically -- from squared Euclidean distances,
first principles, no lookup -- the number of DISTINCT pairwise distances
among its 4 points (there are C(4,2) = 6 pairs). The classical target is:
which of the 8 indexed subsets achieve the MINIMUM distinct-distance count
over this list (a genuine small instance of the Erdos distinct-distances
minimization question).

Quantum circuit
----------------
3 qubits index the 8 candidate subsets. A Grover oracle (built directly from
the classically computed marked indices -- the same numbers, no separate
"quantum answer" is invented) phase-flips exactly the index states whose
subset attains the classical minimum. Grover diffusion amplifies those
states. The circuit is run on the ideal AerSimulator; measurement outcomes
are decoded back to subset indices and compared against the classically
computed marked set.

PASS criterion: every one of the top measured outcomes (as many as there are
marked solutions) is in the classically computed marked-index set, and the
total probability mass on marked indices exceeds a fixed Grover-appropriate
threshold.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, no external data).
# ---------------------------------------------------------------------------

def grid_points():
    """The 3x3 integer grid, 9 points, in a fixed order."""
    return [(x, y) for x in range(3) for y in range(3)]


def distinct_distance_count(points):
    """Number of distinct squared-Euclidean pairwise distances among points.

    (Using squared distances keeps everything in exact integer arithmetic --
    irrelevant for ordering distinct-ness, since d1 < d2 iff d1^2 < d2^2 for
    nonnegative distances, so distinct squared distances <=> distinct
    distances.)
    """
    dists = set()
    for (p, q) in itertools.combinations(points, 2):
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        dists.add(dx * dx + dy * dy)
    return len(dists)


def build_candidate_subsets():
    """A fixed, deterministic list of 8 four-point subsets of the 3x3 grid."""
    pts = grid_points()
    all_subsets = list(itertools.combinations(range(9), 4))
    # Deterministic selection: first 8 subsets in itertools order, over the
    # fixed point indexing above. This is fixed/reproducible, not tuned to
    # the answer -- we compute the property on whatever this list is.
    chosen = all_subsets[:8]
    return [tuple(pts[i] for i in subset) for subset in chosen]


def classical_answer():
    """Compute distinct-distance counts for the 8 candidates and the marked
    (minimizing) index set."""
    subsets = build_candidate_subsets()
    counts = [distinct_distance_count(s) for s in subsets]
    minimum = min(counts)
    marked = [i for i, c in enumerate(counts) if c == minimum]
    return subsets, counts, minimum, marked


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffusion over 3 qubits (8 indices).
# ---------------------------------------------------------------------------

def apply_marking_oracle(qc, qubits, marked_indices, n):
    """Phase-flip exactly the computational-basis states in marked_indices."""
    for idx in marked_indices:
        bits = format(idx, f"0{n}b")
        # Flip qubits that should be 0 so the marked pattern maps to |11..1>
        for q, b in zip(qubits, bits):
            if b == "0":
                qc.x(q)
        if n == 1:
            qc.z(qubits[0])
        elif n == 2:
            qc.cz(qubits[0], qubits[1])
        else:
            qc.h(qubits[-1])
            qc.mcx(qubits[:-1], qubits[-1])
            qc.h(qubits[-1])
        for q, b in zip(qubits, bits):
            if b == "0":
                qc.x(q)


def apply_diffusion(qc, qubits, n):
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


def build_grover_circuit(marked_indices, n=3):
    N = 2 ** n
    M = len(marked_indices)
    # Optimal number of Grover iterations for N states, M marked.
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n, n)
    qubits = list(range(n))
    qc.h(qubits)
    for _ in range(iterations):
        apply_marking_oracle(qc, qubits, marked_indices, n)
        apply_diffusion(qc, qubits, n)
    qc.measure(qubits, qubits)
    return qc, iterations


# ---------------------------------------------------------------------------
# 3. Run and verify.
# ---------------------------------------------------------------------------

def main():
    subsets, counts, minimum, marked = classical_answer()

    print("Erdos problem #953 -- distinct-distances quantum search lane")
    print(f"OEIS id(s) for #953: 'N/A' (none attached) -- see docstring limitation.")
    print(f"Candidate subsets (8 of the C(9,4) 4-point subsets of a 3x3 grid):")
    for i, (s, c) in enumerate(zip(subsets, counts)):
        print(f"  idx {i}: points={s} distinct_distances={c}")
    print(f"Classical minimum distinct-distance count: {minimum}")
    print(f"Classically marked (minimizing) indices: {marked}")

    n = 3
    qc, iterations = build_grover_circuit(marked, n=n)
    qc_t = transpile(qc, AerSimulator())

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc_t, shots=shots).result()
    counts_hist = result.get_counts()

    # Decode: qiskit bit order is little-endian in the classical register
    # string (c[n-1] ... c[0]); our qubits[0] is the least-significant index
    # bit by construction (measure(qubits, qubits) maps qubit i -> clbit i).
    def bitstring_to_index(bs):
        # bs is e.g. "010", clbit n-1 .. clbit 0 left to right
        return int(bs[::-1], 2)

    index_counts = {}
    for bitstring, freq in counts_hist.items():
        idx = bitstring_to_index(bitstring)
        index_counts[idx] = index_counts.get(idx, 0) + freq

    ranked = sorted(index_counts.items(), key=lambda kv: -kv[1])
    top_k = ranked[: len(marked)]
    top_indices = [idx for idx, _ in top_k]
    marked_mass = sum(freq for idx, freq in index_counts.items() if idx in marked) / shots

    print(f"Grover iterations used: {iterations}")
    print(f"Measured index distribution (top): {ranked[:5]}")
    print(f"Top-{len(marked)} measured indices: {top_indices}")
    print(f"Probability mass on classically-marked indices: {marked_mass:.3f}")

    all_top_marked = all(idx in marked for idx in top_indices)
    mass_ok = marked_mass > 0.5

    verified = all_top_marked and mass_ok

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
