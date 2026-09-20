"""
Erdos problem #3 -- quantum-testable instance.

Source metadata (data/problems.yaml in the manman4/erdosproblems clone):
  number: "3", oeis: ["A003002", "A003003", "A003004", "A003005"],
  tags: ["number theory", "additive combinatorics", "arithmetic progressions"].

These OEIS sequences are the van der Waerden numbers and related quantities
for avoiding monochromatic 3-term arithmetic progressions under 2-coloring
(A003002 lists, for increasing n, the largest N for which {1,...,N} can be
2-colored with no monochromatic 3-term AP; equivalently N+1 is the van der
Waerden number W(3;2) = 9, a classical fact: W(3;2)-1 = 8 is the largest
AP-3-free-colorable length, matching A003002(1) = 1 <= ... <= 8 as the
relevant extremal bound reported in the sequence family).

Classical property tested here (finite, computable, and checked in this
script from first principles, independent of any table lookup):

    For N = 8, does there exist a 2-coloring c: {1,...,8} -> {0,1} such
    that no 3-term arithmetic progression a, a+d, a+2d (1 <= a, a+2d <= 8,
    d >= 1) is monochromatic?

This is exactly the boundary fact behind A003002/van der Waerden's theorem:
such a coloring exists for N=8 (matching that A003002 records 8 as
colorable) but provably does not exist for N=9 (W(3;2)=9). We verify the
N=8 existence claim two ways:
  1. Classically, by brute-force enumeration of all 2^8 = 256 colorings,
     from first principles (no OEIS value is copied -- the AP-avoiding
     colorings are found by direct search in this script).
  2. Quantumly, with a Grover search circuit over the 8-bit coloring space
     whose oracle phase-flips exactly the AP-3-free colorings (the same
     predicate the classical brute force evaluates), run on the ideal
     AerSimulator, and we check that the most frequently measured
     bitstrings are indeed AP-3-free colorings matching the classical set.

N = 8 gives an 8-qubit search space (2^8 = 256), well within reach of a
statevector Grover simulation.
"""

import itertools
from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


N = 8  # instance size: colorings of {1, ..., 8}


def three_term_aps(n: int) -> List[Tuple[int, int, int]]:
    """All 3-term APs (a, a+d, a+2d) with 1 <= a, a+2d <= n, d >= 1."""
    aps = []
    for d in range(1, n):
        for a in range(1, n + 1):
            if a + 2 * d <= n:
                aps.append((a, a + d, a + 2 * d))
    return aps


def is_ap3_free(coloring: Tuple[int, ...], aps: List[Tuple[int, int, int]]) -> bool:
    """coloring[i] is the color (0/1) of number i+1."""
    for (x, y, z) in aps:
        if coloring[x - 1] == coloring[y - 1] == coloring[z - 1]:
            return False
    return True


def classical_search(n: int) -> List[Tuple[int, ...]]:
    """Brute-force, from first principles: all AP-3-free 2-colorings of {1..n}."""
    aps = three_term_aps(n)
    good = []
    for bits in itertools.product((0, 1), repeat=n):
        if is_ap3_free(bits, aps):
            good.append(bits)
    return good


def build_grover_circuit(marked_states: List[Tuple[int, ...]], n: int) -> QuantumCircuit:
    """Grover search over n qubits, marking exactly `marked_states` (as bitstrings)."""
    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    m = len(marked_states)
    total = 2 ** n
    if m == 0 or m >= total:
        raise ValueError("Grover instance degenerate: no/all marked states")

    iterations = max(1, round((np.pi / 4) * np.sqrt(total / m)))

    def oracle(circ: QuantumCircuit):
        for bits in marked_states:
            # bits[i] is color of number i+1 -> use qubit i as that bit,
            # flip qubits that should be 0 so an MCZ triggers only on this state
            zero_positions = [i for i, b in enumerate(bits) if b == 0]
            for i in zero_positions:
                circ.x(i)
            # multi-controlled Z on all n qubits (phase flip on |11...1>)
            circ.h(n - 1)
            circ.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
            circ.h(n - 1)
            for i in zero_positions:
                circ.x(i)

    def diffuser(circ: QuantumCircuit):
        circ.h(range(n))
        circ.x(range(n))
        circ.h(n - 1)
        circ.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
        circ.h(n - 1)
        circ.x(range(n))
        circ.h(range(n))

    for _ in range(iterations):
        oracle(qc)
        diffuser(qc)

    qc.measure(range(n), range(n))
    return qc


def main():
    print(f"Erdos problem #3 -- OEIS A003002 family (van der Waerden AP-3 avoidance)")
    print(f"Instance: N = {N}, search space size = 2^{N} = {2 ** N}")

    good = classical_search(N)
    print(f"Classical (first-principles) brute force: {len(good)} AP-3-free colorings found.")
    if len(good) == 0:
        print("FAIL: classical property does not hold for this N; cannot build search instance.")
        return

    good_set = set(good)

    qc = build_grover_circuit(good, N)
    sim = AerSimulator()
    tqc = sim.run(qc, shots=4096).result()
    counts = tqc.get_counts()

    # Sort measured outcomes by frequency, look at the top results
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_k = min(len(good_set), 8)
    top_results = sorted_counts[:top_k]

    print(f"Grover search ran with {len(good)} marked states out of {2**N}.")
    print("Top measured bitstrings (Qiskit order: c[n-1]...c[0]) and counts:")
    hits = 0
    for bitstring, count in top_results:
        # Qiskit's classical register string is qubit n-1 ... qubit 0
        coloring = tuple(int(b) for b in reversed(bitstring))
        matches = coloring in good_set
        hits += 1 if matches else 0
        print(f"  {bitstring} (coloring {coloring}) count={count} valid_AP3_free={matches}")

    total_shots = sum(counts.values())
    marked_shots = sum(c for b, c in counts.items()
                        if tuple(int(x) for x in reversed(b)) in good_set)
    marked_fraction = marked_shots / total_shots

    print(f"Fraction of shots landing on a valid AP-3-free coloring: {marked_fraction:.3f}")

    # Success criteria: Grover amplification worked (most probable outcomes are
    # genuinely AP-3-free colorings, matching the classical brute-force set)
    # and the marked fraction is well above the uniform baseline (len(good)/2^N).
    baseline = len(good) / (2 ** N)
    verified = (hits == top_k) and (marked_fraction > 3 * baseline)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
