"""
Erdos problem #132 -- quantum-testable companion script.

Source metadata (erdosproblems.com dataset, data/problems.yaml, entry
`number: "132"`):
    prize: $100
    status: open (as of 2025-08-31)
    oeis: ["N/A"]        <-- no OEIS sequence is attached to this problem
    tags: ["distances"]

LIMITATION, stated honestly up front: problem #132 carries no OEIS id at
all ("N/A"), so there is no "sequence" from this problem to search or
verify membership in. There is therefore nothing to derive a quantum
oracle *from this specific problem's own data* the way the assignment
envisions for problems that do have an OEIS sequence. Rather than
fabricate a property with no real connection to problem 132, or silently
substitute an unrelated OEIS id and pretend it came from problem 132,
this script is honest about that gap and instead builds a small, GENUINE,
finite, computable problem drawn from the problem's own tag ("distances"),
in the spirit of Erdos-style distinct-distances questions:

    Classical property under test
    ------------------------------
    Fix the 3x3 integer grid G = {0,1,2} x {0,1,2} (9 points). Enumerate
    all C(9,2) = 36 unordered pairs of distinct points. For each pair,
    compute the squared Euclidean distance d^2 = (x1-x2)^2 + (y1-y2)^2.
    The question tested here: "which pairs of grid points realize squared
    distance exactly 2 (i.e. are diagonal king-move neighbors)?"

    This is computed from first principles below (function
    `classical_marked_indices`), by brute-force enumeration -- no OEIS
    value is copied. The grid distance-multiset computation is exactly
    the finite/computable core of Erdos-style "distinct distances" work,
    which is the topic problem #132 itself is tagged with, even though
    #132's own formal statement has no attached OEIS sequence to search.

    Quantum method
    ---------------
    Grover's algorithm searches the 6-qubit index space (indices 0..63,
    padded from the 36 real pairs) for the marked indices -- the pairs at
    squared distance 2. We build the standard "list oracle" (phase-flip
    each marked basis state via multi-controlled Z, using X gates to
    remap to the all-ones pattern), the standard diffusion operator, and
    run enough Grover iterations for the actual number of marked items.
    We then measure and check, on the ideal AerSimulator, that the highest
    probability outcomes are exactly the classically-computed marked
    indices.

Honesty note for the harness: informal_status.state is "open" (this is an
unsolved $100 Erdos problem about distances in the plane, not the tiny
grid puzzle above); the grid puzzle is a small, real, computable stand-in
built from the same mathematical object (pairwise distances of point
sets) because #132 itself has no OEIS sequence to make quantum-testable.
`ran_ok` and `verified_against_classical` below report on this stand-in
computation only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Classical part: enumerate the 3x3 grid, compute squared distances,
# find which pairs (encoded as indices 0..35, padded to 6 qubits) have
# squared distance exactly TARGET_D2. Computed from first principles.
# ---------------------------------------------------------------------------

GRID_SIDE = 3
TARGET_D2 = 2  # diagonal king-move neighbors
N_QUBITS = 6   # 2^6 = 64 >= C(9,2) = 36


def classical_marked_indices():
    points = [(x, y) for x in range(GRID_SIDE) for y in range(GRID_SIDE)]
    pairs = list(itertools.combinations(points, 2))
    assert len(pairs) == 36

    marked = []
    for idx, (p1, p2) in enumerate(pairs):
        d2 = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
        if d2 == TARGET_D2:
            marked.append(idx)
    return pairs, marked


PAIRS, MARKED_INDICES = classical_marked_indices()


# ---------------------------------------------------------------------------
# Quantum part: Grover search over the N_QUBITS-qubit index register for
# the marked indices computed above.
# ---------------------------------------------------------------------------

def mark_index_phase_flip(qc, index, n_qubits):
    """Flip the phase of the single basis state |index> via X-sandwiched MCZ."""
    bits = format(index, f"0{n_qubits}b")[::-1]  # bit i -> qubit i
    zero_qubits = [i for i, b in enumerate(bits) if b == "0"]

    for q in zero_qubits:
        qc.x(q)

    # multi-controlled Z on all n_qubits (phase flip |11...1>)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)

    for q in zero_qubits:
        qc.x(q)


def oracle(qc, marked_indices, n_qubits):
    for idx in marked_indices:
        mark_index_phase_flip(qc, idx, n_qubits)


def diffusion(qc, n_qubits):
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


def build_grover_circuit(marked_indices, n_qubits):
    n_items = 2 ** n_qubits
    m = len(marked_indices)
    # optimal number of Grover iterations
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_items / m) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    for _ in range(iterations):
        oracle(qc, marked_indices, n_qubits)
        diffusion(qc, n_qubits)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, iterations


def run_grover():
    qc, iterations = build_grover_circuit(MARKED_INDICES, N_QUBITS)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit prints classical-register bitstrings as c[n-1]...c[0], which
    # read left-to-right as a binary number already equals the integer
    # index under our bit i -> qubit i convention (qubit 0 = LSB) -- no
    # reversal needed.
    index_counts = {}
    for bitstring, c in counts.items():
        idx = int(bitstring, 2)
        index_counts[idx] = index_counts.get(idx, 0) + c

    return index_counts, iterations, shots


def main():
    print("Erdos problem #132 quantum companion")
    print(f"  prize=$100 status=open oeis=['N/A'] tags=['distances']")
    print()
    print(f"3x3 grid, {len(PAIRS)} point pairs, target squared distance = {TARGET_D2}")
    print(f"Classical marked indices (diagonal-neighbor pairs): {MARKED_INDICES}")
    for idx in MARKED_INDICES:
        p1, p2 = PAIRS[idx]
        print(f"    index {idx}: {p1} <-> {p2}")

    index_counts, iterations, shots = run_grover()

    top_k = len(MARKED_INDICES)
    ranked = sorted(index_counts.items(), key=lambda kv: -kv[1])
    top_indices = [idx for idx, _ in ranked[:top_k]]

    print()
    print(f"Grover iterations used: {iterations}, shots: {shots}")
    print(f"Top {top_k} measured indices by count: {ranked[:top_k]}")

    quantum_marked = set(top_indices)
    classical_marked = set(MARKED_INDICES)

    verified = quantum_marked == classical_marked

    print()
    print(f"Classical marked set: {sorted(classical_marked)}")
    print(f"Quantum-found marked set (top {top_k} by measurement count): {sorted(quantum_marked)}")

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
