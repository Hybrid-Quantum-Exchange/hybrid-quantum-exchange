"""
Erdos problem #193 -- quantum-testable instance.

Source: erdosproblems.com problem 193 (data/problems.yaml entry `number: "193"`,
tags: ["geometry"], informal_status: disproved (Lean), OEIS id used: A231255).

A231255(n) = the smallest integer t such that *every* length-t North-East
lattice walk from the origin (steps only (0,1) "up" or (1,0) "right")
necessarily contains n collinear points among the t+1 points it visits.

OEIS data (verified against the OEIS entry for A231255):
    a(3) = 4,  with the known witness explanation:
    "a(3) = 4 because two consecutive identical steps from (0,0) generate
     3 collinear points, so the first three steps must alternate
     (0,1),(1,0),(0,1) or its complement (1,0),(0,1),(1,0). Then no matter
     what is chosen for the next step, three collinear points are generated."

The classical, finite, computable property tested here:

    For t = 3 (search space of all 2^3 = 8 NE walks of length 3, i.e. all
    3-bit strings, one qubit per step), find the set S of walks that AVOID
    3 collinear points among their 4 visited points (origin + 3 steps).
    Because a(3) = 4 (not 3), S must be non-empty -- t=3 is exactly the
    last length at which avoidance is still possible, and OEIS says the
    only two avoiders are the alternating walks 1,0,1 and 0,1,0 (using the
    encoding 0="up", 1="right"; OEIS's own %C line writes these as "121"
    and "212" in 1/2 notation).

This script:
  1. Computes S classically from first principles (enumerate all 8 walks,
     test all C(4,3)=4 point-triples of each for collinearity via the
     cross-product test), with NO OEIS values copied in -- only used as a
     sanity cross-check after the fact.
  2. Builds a genuine Grover search circuit over the 3-qubit space (one
     qubit per step) whose oracle marks exactly the walks classically
     found not to contain 3 collinear points, then amplifies and measures.
  3. Runs it on the ideal AerSimulator and checks that the walks Grover
     returns with highest probability are exactly the classically
     computed set S. Prints PASS/FAIL.

No OEIS numeric values are hard-coded into the classical computation --
a(3)=4 and the "121/212" walks are only used afterwards, as a printed
cross-check, not as an input to the search or the oracle.
"""

import itertools
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

T = 3  # walk length (number of steps / qubits) for this instance


# ---------------------------------------------------------------------------
# Step 1: classical computation, from first principles, no OEIS values used.
# ---------------------------------------------------------------------------

def walk_points(bits):
    """bits: tuple of 0/1, 0 = step (0,1) 'up', 1 = step (1,0) 'right'.
    Returns the list of T+1 lattice points visited, starting at (0,0)."""
    x, y = 0, 0
    pts = [(0, 0)]
    for b in bits:
        if b == 0:
            y += 1
        else:
            x += 1
        pts.append((x, y))
    return pts


def has_three_collinear(pts):
    """True if some 3 of the given points are collinear (cross product
    of the two difference vectors is zero)."""
    for (p1, p2, p3) in itertools.combinations(pts, 3):
        (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
        cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        if cross == 0:
            return True
    return False


all_bitstrings = list(itertools.product([0, 1], repeat=T))
avoiding = []  # walks of length T whose T+1 points contain NO 3 collinear points
for bits in all_bitstrings:
    pts = walk_points(bits)
    if not has_three_collinear(pts):
        avoiding.append(bits)

print(f"Classical enumeration: {len(all_bitstrings)} walks of length {T}.")
print(f"Walks avoiding 3 collinear points (classically found): {avoiding}")

# Cross-check against the known OEIS fact a(3) = 4 (t=3 must still allow
# avoidance, since a(3) is the first length that FORCES 3 collinear points).
# This is only a sanity check, not an input to the search above.
assert len(avoiding) > 0, "expected some avoiders to exist for t=3 (a(3)=4 means t=3 does not yet force collinearity)"
expected = {(0, 1, 0), (1, 0, 1)}  # OEIS %C: "121" and its complement, in 0/1 encoding above
assert set(avoiding) == expected, f"classical result {avoiding} does not match OEIS-documented avoiders {expected}"
print(f"Matches OEIS-documented avoiders (encoded as 0/1 bit strings): {sorted(expected)}")

marked_states = avoiding  # the set Grover search must find
N = 2 ** T
M = len(marked_states)
print(f"Search space size N={N}, marked (target) states M={M}")


# ---------------------------------------------------------------------------
# Step 2: build a genuine Grover search circuit marking exactly `marked_states`.
# ---------------------------------------------------------------------------

def oracle_circuit(n_qubits, targets):
    """Phase-flip oracle marking each bitstring in `targets` (tuples, bit i
    = value of qubit i, matching Qiskit's little-endian qubit-to-bit-string
    convention used throughout this script)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bits in targets:
        # flip qubits that should be 0 so a multi-controlled Z fires only
        # when the register equals `bits`
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def diffuser_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


# optimal number of Grover iterations for N states, M marked
num_iterations = max(1, round((np.pi / 4) * np.sqrt(N / M) - 0.5))

qc = QuantumCircuit(T, T)
qc.h(range(T))
oracle = oracle_circuit(T, marked_states)
diffuser = diffuser_circuit(T)
for _ in range(num_iterations):
    qc.append(oracle.to_instruction(), range(T))
    qc.append(diffuser.to_instruction(), range(T))
qc.measure(range(T), range(T))

print(f"Grover iterations used: {num_iterations}")


# ---------------------------------------------------------------------------
# Step 3: run on the ideal AerSimulator and compare to the classical result.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's bit strings are big-endian in the printed key (qubit T-1 first).
# Convert each measured key back to our little-endian bits tuple (bit i =
# qubit i) so it can be compared to `marked_states` directly.
def key_to_bits(key):
    rev = key[::-1]
    return tuple(int(c) for c in rev)

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_m = sorted_counts[:M]
top_m_bits = {key_to_bits(k) for k, _ in top_m}

total_marked_shots = sum(c for k, c in counts.items() if key_to_bits(k) in set(marked_states))
marked_fraction = total_marked_shots / shots

print(f"Measurement counts: {counts}")
print(f"Top-{M} measured bit strings (converted): {sorted(top_m_bits)}")
print(f"Fraction of shots landing on classically-marked states: {marked_fraction:.3f}")

verified = (top_m_bits == set(marked_states)) and (marked_fraction > 0.8)

if verified:
    print("PASS")
else:
    print("FAIL")
