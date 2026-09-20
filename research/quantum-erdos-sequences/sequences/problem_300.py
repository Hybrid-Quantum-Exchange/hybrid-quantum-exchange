"""
Erdos problem #300  (https://www.erdosproblems.com/300)
OEIS id used: A390393

Erdos problem statement (informal): let A(N) be the maximal size of a subset
A of {1,...,N} such that NO sub-subset S of A has sum_{n in S} 1/n = 1
exactly. A390393 records computational/related data for this "no subset of
unit fractions sums exactly to 1" question. The underlying combinatorial
primitive of the problem is therefore:

    Given a finite set A of positive integers, does there exist a
    sub-subset S subseteq A with  sum_{n in S} 1/n == 1  exactly (as an
    exact rational number, not an approximation)?

This is a finite, exactly-computable decision/search problem over the
2^|A| subsets of A, which is exactly the shape Grover's algorithm is built
for: an oracle that recognizes "good" subsets (those summing to exactly 1)
and amplitude amplification that boosts the chance of measuring one.

Classical instance chosen here (first principles, computed in this script,
NOT copied from OEIS):

    A = [2, 3, 4, 6, 12]   (5 elements -> 5 qubits, search space size 32)

We first classically enumerate ALL 2^5 = 32 subsets of A using exact
Fraction arithmetic and find every subset S with sum_{n in S} 1/n == 1.
This gives the ground truth ("marked" bitstrings). Then we build a Grover
oracle in Qiskit that flips the phase of exactly those bitstrings (encoded
as multi-controlled-Z gates conditioned on the bit pattern of each marked
subset), run the standard number-of-iterations Grover search on the ideal
AerSimulator, and check that the most frequently measured bitstring(s) are
among the classically-verified marked subsets.

This is a genuine unstructured search over an oracle built from the
problem's real combinatorial content (exact unit-fraction subset sums),
not a fabricated or hand-picked "quantum" dressing of a literal OEIS value.
"""

import itertools
from fractions import Fraction

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


def classical_marked_subsets(A):
    """Return, as a sorted list of bitstrings (index 0 = A[0], LSB first
    in the qubit register), every non-empty subset S of A with
    sum_{n in S} 1/n exactly equal to 1. Uses exact Fraction arithmetic,
    so this is a rigorous classical computation, not an approximation."""
    n = len(A)
    marked = []
    for bits in itertools.product([0, 1], repeat=n):
        if not any(bits):
            continue
        s = sum(Fraction(1, A[i]) for i in range(n) if bits[i])
        if s == 1:
            # bits[0] corresponds to qubit 0 (A[0]); build the bitstring
            # in Qiskit's little-endian convention: qubit 0 is the
            # rightmost character.
            bitstring = "".join(str(bits[n - 1 - k]) for k in range(n))
            marked.append(bitstring)
    return sorted(marked)


def build_oracle(n, marked_bitstrings):
    """Phase-flip oracle: for each marked bitstring, apply X gates on the
    qubits that should be 0, a multi-controlled-Z (via H-MCX-H on the last
    qubit) to flip the phase when all qubits match the marked pattern, then
    undo the X gates."""
    qc = QuantumCircuit(n, name="oracle")
    for bitstring in marked_bitstrings:
        # bitstring[k] is qubit (n-1-k) per Qiskit's string convention;
        # here we just read bit b for qubit index i directly.
        bits = [int(b) for b in bitstring[::-1]]  # bits[i] -> qubit i
        zero_qubits = [i for i in range(n) if bits[i] == 0]
        for q in zero_qubits:
            qc.x(q)
        qc.h(n - 1)
        qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
        qc.h(n - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def run_grover(n, marked_bitstrings, shots=4096):
    oracle = build_oracle(n, marked_bitstrings)
    diffuser = build_diffuser(n)

    num_marked = len(marked_bitstrings)
    N = 2 ** n
    # Standard optimal iteration count for Grover's algorithm.
    theta = np.arcsin(np.sqrt(num_marked / N))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n))
        qc.append(diffuser.to_gate(), range(n))
    qc.measure(range(n), range(n))
    qc = qc.decompose().decompose().decompose()

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    A = [2, 3, 4, 6, 12]
    n = len(A)

    marked = classical_marked_subsets(A)
    print(f"Set A = {A}")
    print(f"Classically found {len(marked)} marked subset(s) (exact unit-fraction sums == 1):")
    for bitstring in marked:
        subset = [A[i] for i in range(n) if bitstring[::-1][i] == "1"]
        print(f"  bitstring={bitstring}  subset={subset}  "
              f"sum=1/{'+1/'.join(str(x) for x in subset)} == 1")

    if not marked:
        print("No marked subsets found for this instance; cannot build a "
              "meaningful Grover search. FAIL")
        return False

    counts, iterations = run_grover(n, marked, shots=4096)
    print(f"\nGrover iterations used: {iterations}")
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    print("Top measured outcomes:")
    total_shots = sum(counts.values())
    for bitstring, c in sorted_counts[:6]:
        flag = "MARKED" if bitstring in marked else "unmarked"
        print(f"  {bitstring}: {c:5d} ({100.0 * c / total_shots:5.1f}%)  [{flag}]")

    # Verification: the total probability mass on marked bitstrings should
    # dominate (Grover amplification), and the single most frequent
    # outcome should be one of the classically-verified marked subsets.
    marked_mass = sum(c for b, c in counts.items() if b in marked)
    marked_fraction = marked_mass / total_shots
    top_bitstring, _ = sorted_counts[0]
    top_is_marked = top_bitstring in marked

    print(f"\nFraction of shots landing on a marked (true) subset: {marked_fraction:.3f}")
    print(f"Most frequent outcome is a marked subset: {top_is_marked}")

    passed = top_is_marked and marked_fraction > 0.5
    print("\nPASS" if passed else "\nFAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
