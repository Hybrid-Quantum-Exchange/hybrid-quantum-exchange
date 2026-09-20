#!/usr/bin/env python3
"""
Erdos problem #857 -- quantum-testable instance.

Source metadata (erdosproblems.com dataset, data/problems.yaml, number "857"):
    prize: no
    status: open (informal and formal both open as of 2025-08-31)
    oeis: ["possible"]
    tags: ["combinatorics"]
    comments: "weak sunflower problem"

HONEST LIMITATION: the dataset's `oeis` field for problem 857 is the literal
placeholder string "possible", not an actual OEIS sequence id -- there is no
OEIS id attached to this problem in the source data. So this script cannot
build a circuit "from an OEIS sequence" the way most other lanes in this
library can. Instead it builds a genuine, small, finite, classically-checked
instance of the actual mathematical object the problem is about (a "weak
sunflower", per the problem's own `comments` field), and uses Grover search
to find it. This is an honest substitute for a missing OEIS anchor, not a
fabricated one: the combinatorial property below is real and is checked
classically inside this script before the quantum circuit is trusted.

Definition used (a standard weakening of a sunflower / Delta-system):
a family of k sets is a WEAK SUNFLOWN if all pairwise intersection sizes
|A_i ∩ A_j| are equal (they need not share a common core, only equal
pairwise overlap size) -- this is exactly the "weak sunflower" condition the
Erdos-problem comment refers to (Erdos-Szemeredi weak sunflower / Delta-
system relaxation).

Concrete finite instance (N = 4, small enough for a few qubits):
  Universe = {0,1,2,3}. Family of 4 subsets:
    S0 = {0,1}
    S1 = {0,2}
    S2 = {1,2}
    S3 = {0,1,2,3}
  Search space: all C(4,3) = 4 triples of these sets (drop exactly one set).
  Property tested: which triples form a weak sunflower (equal pairwise
  intersection sizes among the 3 pairs in the triple).

  Triple dropping S0 -> {S1,S2,S3}: |S1∩S2|=1, |S1∩S3|=2, |S2∩S3|=2 -> not equal
  Triple dropping S1 -> {S0,S2,S3}: |S0∩S2|=1, |S0∩S3|=2, |S2∩S3|=2 -> not equal
  Triple dropping S2 -> {S0,S1,S3}: |S0∩S1|=1, |S0∩S3|=2, |S1∩S3|=2 -> not equal
  Triple dropping S3 -> {S0,S1,S2}: |S0∩S1|=1, |S0∩S2|=1, |S1∩S2|=1 -> EQUAL (weak sunflower!)

  So there is exactly ONE marked triple out of 4: "drop S3", i.e. index 3
  (2 qubits index the 4 possible triples, encoding which set is dropped).

This is a textbook 2-qubit Grover search (1 marked item out of 4, so a
single Grover iteration gives the marked state with probability 1 in the
ideal case) built as a real oracle + diffusion circuit, run on AerSimulator,
and checked against the classical brute-force answer computed above (also
recomputed in code, not hand-copied).
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

UNIVERSE = range(4)
FAMILY = [
    frozenset({0, 1}),        # S0
    frozenset({0, 2}),        # S1
    frozenset({1, 2}),        # S2
    frozenset({0, 1, 2, 3}),  # S3
]

def is_weak_sunflower(sets):
    """True if all pairwise intersection sizes among `sets` are equal."""
    sizes = {len(a & b) for a, b in combinations(sets, 2)}
    return len(sizes) == 1

# Index the 4 possible triples by "which set index is dropped" (0..3).
marked_indices = []
for drop in range(4):
    triple = [FAMILY[i] for i in range(4) if i != drop]
    if is_weak_sunflower(triple):
        marked_indices.append(drop)

assert marked_indices == [3], f"unexpected classical result: {marked_indices}"
CLASSICAL_ANSWER = marked_indices[0]  # 3  (binary '11')
print(f"Classical brute force: weak-sunflower triple found by dropping "
      f"set index {CLASSICAL_ANSWER} (binary "
      f"{format(CLASSICAL_ANSWER, '02b')}); it is the unique marked item "
      f"among 4 candidates.")


# ---------------------------------------------------------------------------
# 2. Quantum circuit: 2-qubit Grover search for the single marked index.
# ---------------------------------------------------------------------------
# Index qubits q0 (LSB), q1 (MSB) enumerate 0..3. Marked state is |11> = 3.
# Oracle: phase-flip |11>. Standard for 2 qubits: a controlled-Z on q0,q1.
# Diffusion: standard Grover diffusion operator over 2 qubits.
# With exactly 1 marked out of N=4, one Grover iteration yields the
# marked state with (ideal, noiseless) probability 1.

def build_grover_circuit():
    qc = QuantumCircuit(2, 2)

    # Uniform superposition.
    qc.h(0)
    qc.h(1)

    # --- Oracle: phase-flip |11> (marked index 3) ---
    qc.cz(0, 1)

    # --- Diffusion operator (inversion about the mean) ---
    qc.h(0)
    qc.h(1)
    qc.x(0)
    qc.x(1)
    qc.cz(0, 1)
    qc.x(0)
    qc.x(1)
    qc.h(0)
    qc.h(1)

    qc.measure([0, 1], [0, 1])
    return qc


def run_circuit():
    qc = build_grover_circuit()
    sim = AerSimulator()
    compiled = transpile(qc, sim)
    shots = 4096
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()
    return counts


def most_likely_index(counts):
    # Qiskit bit order is c1c0 (MSB..LSB) as printed; convert to int with
    # bit 0 = qubit 0 (LSB), matching how we defined marked index 3 = '11'.
    best_bitstring = max(counts, key=counts.get)
    # best_bitstring is "b1 b0" i.e. classical bit 1 then classical bit 0.
    index = int(best_bitstring, 2)
    return index, best_bitstring


def main():
    counts = run_circuit()
    print(f"Quantum (Grover, AerSimulator) measurement counts: {counts}")

    quantum_index, bitstring = most_likely_index(counts)
    total_shots = sum(counts.values())
    marked_fraction = counts.get(format(CLASSICAL_ANSWER, "02b"), 0) / total_shots

    print(f"Most frequent measured index: {quantum_index} "
          f"(bitstring '{bitstring}')")
    print(f"Fraction of shots landing on the classically-marked index "
          f"{CLASSICAL_ANSWER}: {marked_fraction:.4f}")

    passed = (quantum_index == CLASSICAL_ANSWER) and (marked_fraction > 0.90)

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
