"""
Erdos Problem #36 -- quantum-testable instance
================================================

Problem #36 (erdosproblems.com / manman4/erdosproblems data/problems.yaml,
entry "number: '36'", comment "minimum overlap problem", tags
["number theory", "additive combinatorics"]) is Erdos's MINIMUM OVERLAP
PROBLEM. The associated OEIS id recorded in the source data is A393584
(flagged there as "possible" -- i.e. the maintainers themselves mark the
OEIS match as tentative, not confirmed). Because that id's exact defining
formula is not independently verifiable from this offline environment, this
script does NOT lean on any literal value copied from OEIS. Instead it
implements, from first principles, the classical combinatorial quantity that
problem #36 is actually about, computes the correct answer for a small
instance with brute-force classical search (so the "expected" value is
derived here, not looked up), and then uses a real Grover search circuit to
rediscover that same answer quantumly.

The minimum overlap problem
----------------------------
Partition {1, 2, ..., 2n} into two disjoint sets A and B with |A| = |B| = n.
For an integer k, define the "overlap at shift k" as

    f_{A,B}(k) = #{ (a, b) : a in A, b in B, a - b = k }.

(Since A and B are disjoint, f(0) is always 0, so it never determines the
max.) Erdos asked for

    M(n) = min over all such partitions (A, B) of  max_k f_{A,B}(k).

This is a genuine, finite, exactly-computable combinatorial search problem:
for fixed small n there are C(2n, n) partitions, and M(n) can be found by
brute force.

Small instance used here: n = 3, universe {1, ..., 6}. The search space is
all 6-bit strings x in {0,1}^6 (bit i selects whether element i+1 is in A);
this is exactly N = 2^6 = 64 basis states, small enough for a real Grover
search circuit on 6 qubits.

What the script does
---------------------
1. Classically brute-forces, over all C(6,3) = 20 valid partitions (bit
   strings of popcount 3), the value M(3) = min_A max_k f_{A,B}(k), and also
   records the full set of *optimal* bit strings (those achieving the
   overlap minimum) -- this is the ground truth, derived here, not copied
   from anywhere.
2. Builds a Grover search circuit over 6 qubits whose oracle marks exactly
   the optimal bit strings found in step 1 (a multi-controlled-Z per marked
   basis state, via X-gate sandwiching -- the standard "marked list" Grover
   oracle construction), with the standard diffuser, run for the
   theoretically optimal number of Grover iterations.
3. Runs the circuit on the ideal AerSimulator, takes the most frequently
   measured bitstring(s), and checks that every high-probability outcome
   really is one of the classically-optimal partitions (popcount 3, overlap
   equal to M(3)).
4. Prints PASS if the quantum search result matches the classical answer,
   FAIL otherwise.

No values are hard-coded from OEIS; every number that matters (M(3), the set
of optimal partitions, the Grover iteration count) is computed in this file.
"""

import itertools
import math
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_ELEMENTS = 6      # universe {1, ..., 6}
HALF = N_ELEMENTS // 2  # |A| = |B| = 3
N_QUBITS = N_ELEMENTS    # one qubit per element, 2^6 = 64 basis states


def overlap_profile(bits):
    """bits: tuple of 0/1 of length N_ELEMENTS, bit i => element i+1 in A.

    Returns max_k f(k) for the induced partition (A, B), or None if the
    bitstring is not a valid |A| = |B| = N_ELEMENTS/2 partition.
    """
    if sum(bits) != HALF:
        return None
    A = [i + 1 for i, b in enumerate(bits) if b == 1]
    B = [i + 1 for i, b in enumerate(bits) if b == 0]
    counts = {}
    for a in A:
        for b in B:
            k = a - b
            counts[k] = counts.get(k, 0) + 1
    return max(counts.values()) if counts else 0


def classical_ground_truth():
    """Brute-force M(3) and the set of optimal bit strings, exactly."""
    best = None
    optimal_bits = []
    all_valid = []
    for bits in itertools.product([0, 1], repeat=N_QUBITS):
        prof = overlap_profile(bits)
        if prof is None:
            continue
        all_valid.append(bits)
        if best is None or prof < best:
            best = prof
            optimal_bits = [bits]
        elif prof == best:
            optimal_bits.append(bits)
    assert len(all_valid) == math.comb(N_ELEMENTS, HALF)
    return best, optimal_bits


def bits_to_int(bits):
    """bits[0] is qubit 0 (least significant in Qiskit's little-endian
    convention used for circuit construction); build the integer with that
    same convention so it matches circuit basis-state indexing."""
    val = 0
    for i, b in enumerate(bits):
        if b:
            val |= (1 << i)
    return val


def build_oracle(marked_bitstrings, n_qubits):
    """Standard 'marked list' Grover oracle: for each marked computational
    basis state, sandwich an X on every 0-bit around a multi-controlled Z,
    flipping the phase of exactly that basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bits in marked_bitstrings:
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


def run_grover(marked_bitstrings, n_qubits, shots=4096):
    n_total = 2 ** n_qubits
    n_marked = len(marked_bitstrings)
    theta = math.asin(math.sqrt(n_marked / n_total))
    iterations = max(1, round((math.pi / 4 / theta) - 0.5))

    oracle = build_oracle(marked_bitstrings, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    classical_M, optimal_bits = classical_ground_truth()
    marked_ints = sorted(bits_to_int(b) for b in optimal_bits)

    print(f"Erdos problem #36 -- minimum overlap problem, instance n={HALF}")
    print(f"Classical brute force: M({HALF}) = {classical_M}")
    print(f"Number of optimal partitions (bitstrings): {len(optimal_bits)}")
    print(f"Optimal bitstrings (as ints, qubit0=LSB): {marked_ints}")

    counts, iterations = run_grover(optimal_bits, N_QUBITS)
    shots = sum(counts.values())
    print(f"Grover iterations used: {iterations}")

    # Qiskit counts keys are big-endian strings 'q_{n-1}...q_0'; convert to
    # our little-endian integer convention (qubit 0 = LSB) to compare.
    def key_to_int(key):
        # key given MSB..LSB (qubit n-1 .. qubit 0)
        rev = key[::-1]
        return int(rev, 2)

    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_prob_mass = sum(c for _, c in sorted_counts[: len(marked_ints)]) / shots
    top_outcomes = [key_to_int(k) for k, _ in sorted_counts[: len(marked_ints)]]

    print(f"Top {len(marked_ints)} measured outcomes (by count): {top_outcomes}")
    print(f"Probability mass on top-{len(marked_ints)} outcomes: {top_prob_mass:.3f}")

    all_top_are_marked = all(v in marked_ints for v in top_outcomes)
    # Also verify each top outcome independently, classically, achieves M.
    verified_classically = True
    for v in top_outcomes:
        bits = tuple((v >> i) & 1 for i in range(N_QUBITS))
        prof = overlap_profile(bits)
        if prof is None or prof != classical_M:
            verified_classically = False

    success = all_top_are_marked and verified_classically and top_prob_mass > 0.5

    print()
    if success:
        print("PASS: Grover search recovered a classically-optimal minimum-"
              "overlap partition (matches brute-force M(3)).")
    else:
        print("FAIL: Grover search result did not match the classical "
              "minimum-overlap answer.")
        sys.exit(1)


if __name__ == "__main__":
    main()
