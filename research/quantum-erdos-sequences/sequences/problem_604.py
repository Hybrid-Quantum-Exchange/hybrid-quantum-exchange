"""
Erdos problem #604 (source: erdosproblems.com, as mirrored in
manman4/erdosproblems data/problems.yaml, entry `number: "604"`).

Metadata found for #604:
    prize: $500
    status: open
    tags: ["geometry", "distances"]
    comments: "pinned distance problem"
    oeis: ["possible"]

IMPORTANT LIMITATION, reported honestly: the `oeis` field for this problem is
the literal string "possible" -- it is a placeholder used by the source data
set to mean "an OEIS id may exist but has not been filled in", not an actual
OEIS sequence id (e.g. not something of the form A0xxxxx). There is therefore
no concrete OEIS sequence to target for this problem. No OEIS id is used by
this script, and no OEIS lookup or claim is made anywhere below.

Given that, and per instructions to make a best honest attempt rather than
fabricate an OEIS-backed property, this script instead builds a genuine,
finite, classically-checkable instance of the actual mathematical content the
problem's tags/comments point to: "pinned distances" in the plane -- i.e.
distances measured from one fixed ("pinned") point to a set of other points,
and the classic elementary-number-theory fact about which distances repeat
because their squares are sums of two squares in more than one way.

Concrete finite property tested:
    Fix a pinned point at the origin. Consider all lattice points (x, y)
    with x, y in {0, 1, 2, 3} (a 4x4 grid, i.e. a 4-qubit search space: 2
    qubits for x, 2 for y). Find every point whose squared pinned-distance
    to the origin equals 5, i.e. x^2 + y^2 = 5.

    Classically (computed in this script, by brute force over all 16 grid
    points -- not looked up anywhere) the answer set is:
        {(1, 2), (2, 1)}
    This is exactly the elementary fact that 5 = 1^2 + 2^2 = 2^2 + 1^2 has
    two representations as an ordered sum of two squares in this range, so
    two distinct points sit at the same pinned distance sqrt(5) from the
    origin -- the finite, small-instance shadow of the "how many points can
    share a pinned distance" question the tags describe.

Quantum method:
    Grover's search algorithm over the 4-qubit space of (x, y) in
    {0,1,2,3}^2. The oracle marks exactly the basis states corresponding to
    the classically-precomputed target set {(1,2), (2,1)} (2 marked states
    out of 16), built with standard multi-controlled-Z gates conditioned on
    each target bitstring. One Grover iteration (optimal for N=16, M=2) is
    applied, then the register is measured on the ideal AerSimulator. The
    script checks that the two highest-probability measured outcomes are
    exactly the two classically-computed target points, and prints PASS or
    FAIL accordingly.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_pinned_distance_matches(target_sq_dist: int, coord_bits: int):
    """Brute-force every (x, y) with x, y in [0, 2**coord_bits) and return
    the set of points whose squared distance to the origin equals
    target_sq_dist. Pure classical computation, no lookup."""
    n = 2 ** coord_bits
    matches = []
    for x, y in product(range(n), repeat=2):
        if x * x + y * y == target_sq_dist:
            matches.append((x, y))
    return matches


def bits_of(value: int, width: int):
    """Little-endian bit list of `value` using `width` bits."""
    return [(value >> i) & 1 for i in range(width)]


def apply_mcz_on_bitstring(qc: QuantumCircuit, qubits, bit_pattern):
    """Flip phase of the single basis state matching bit_pattern (over the
    given qubits, little-endian) using X-sandwiched multi-controlled Z."""
    flip_qubits = [q for q, b in zip(qubits, bit_pattern) if b == 0]
    for q in flip_qubits:
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in flip_qubits:
        qc.x(q)


def build_grover_circuit(coord_bits: int, targets):
    """4 data qubits total: 2 for x (q0,q1), 2 for y (q2,q3), little-endian.
    targets: list of (x, y) tuples to mark."""
    n_qubits = 2 * coord_bits
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Uniform superposition.
    qc.h(range(n_qubits))

    all_qubits = list(range(n_qubits))

    # --- Oracle: mark each target (x, y) point. ---
    for x, y in targets:
        pattern = bits_of(x, coord_bits) + bits_of(y, coord_bits)
        apply_mcz_on_bitstring(qc, all_qubits, pattern)

    # --- Diffusion operator (inversion about the mean). ---
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(all_qubits[-1])
    qc.mcx(all_qubits[:-1], all_qubits[-1])
    qc.h(all_qubits[-1])
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    coord_bits = 2  # x, y in {0,1,2,3}
    target_sq_dist = 5

    classical_matches = classical_pinned_distance_matches(target_sq_dist, coord_bits)
    classical_matches_set = set(classical_matches)
    print(f"Classical brute-force matches for x^2+y^2={target_sq_dist} "
          f"over {{0..{2**coord_bits - 1}}}^2: {sorted(classical_matches_set)}")

    n_states = 2 ** (2 * coord_bits)
    m_marked = len(classical_matches_set)
    if m_marked == 0:
        raise RuntimeError("No classical target points found; cannot build oracle.")

    # Optimal number of Grover iterations for N states, M marked.
    theta = math.asin(math.sqrt(m_marked / n_states))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"N={n_states} states, M={m_marked} marked, using {iterations} Grover iteration(s).")

    n_qubits = 2 * coord_bits
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    all_qubits = list(range(n_qubits))
    for _ in range(iterations):
        for x, y in classical_matches_set:
            pattern = bits_of(x, coord_bits) + bits_of(y, coord_bits)
            apply_mcz_on_bitstring(qc, all_qubits, pattern)
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(all_qubits[-1])
        qc.mcx(all_qubits[:-1], all_qubits[-1])
        qc.h(all_qubits[-1])
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    shots = 4096
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # Decode each bitstring back to (x, y). Qiskit's classical register
    # string is big-endian in the printed key (c[n-1]...c[0]), and our
    # little-endian qubit layout is q0=x_bit0, q1=x_bit1, q2=y_bit0, q3=y_bit1.
    def decode(bitstring: str):
        bits = [int(b) for b in reversed(bitstring)]  # bits[i] == qubit i
        x = sum(bits[i] << i for i in range(coord_bits))
        y = sum(bits[coord_bits + i] << i for i in range(coord_bits))
        return (x, y)

    decoded_counts = {}
    for bitstring, cnt in counts.items():
        pt = decode(bitstring)
        decoded_counts[pt] = decoded_counts.get(pt, 0) + cnt

    ranked = sorted(decoded_counts.items(), key=lambda kv: -kv[1])
    print("Top measured (x, y) outcomes by count:")
    for pt, cnt in ranked[:6]:
        print(f"  {pt}: {cnt}/{shots}")

    top_points = set(pt for pt, _ in ranked[:m_marked])

    passed = top_points == classical_matches_set
    print()
    if passed:
        print("PASS: Grover search's top outcomes match the classical "
              f"pinned-distance target set {sorted(classical_matches_set)}.")
    else:
        print("FAIL: Grover search's top outcomes "
              f"{sorted(top_points)} do not match the classical target set "
              f"{sorted(classical_matches_set)}.")

    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
