"""
Erdos problem #530 -- quantum-testable instance.

Source metadata (erdosproblems.com data, problems.yaml):
    number: 530
    tags: ["number theory", "sidon sets"]
    oeis: ["A143824", "possible"]   (A143824 relates to Sidon sets)

Erdos problem 530 concerns Sidon sets (also called B2 sets / Sidon sequences):
a set S of non-negative integers is a Sidon set if all pairwise sums a + b
(with a <= b, a, b in S) are distinct. This script does not try to resolve
the open Erdos problem itself (it is open, per informal_status.state =
"open" in problems.yaml); instead it isolates a small, finite, genuinely
computable property that the problem's subject matter is built on --
"is S a Sidon set?" -- and uses a real Grover search to find a Sidon set of
a given size inside a small universe, verifying the quantum result against
a from-scratch classical computation.

Classical property tested (computed in this script, not copied from OEIS):
    Universe: {0, 1, ..., 7} (8 elements, encoded as an 8-bit string, bit i
    set iff element i is in the subset).
    Search target: 3-element subsets S of the universe such that all
    pairwise sums a + b (a <= b, a, b in S) are pairwise distinct (Sidon
    condition).
    We first brute-force enumerate, from first principles, every 3-element
    subset of {0,...,7} and classically determine exactly which ones are
    Sidon sets. Example: {0, 1, 3} is Sidon because its pairwise sums are
    0+0? -- no, sums are taken over the 6 pairs (i<=j) among the 3 chosen
    elements: 0+1=1, 0+3=3, 1+3=4, 0+0 is not applicable (each element used
    once per pair, i<j only, 3 pairs total for a 3-set): 0+1=1, 0+3=3,
    1+3=4 -- all distinct, so {0,1,3} is Sidon.

Quantum circuit: Grover's algorithm over 8 qubits (one qubit per universe
element, |1> = element included). The oracle phase-flips exactly the
computational basis states corresponding to the classically-precomputed
"good" set (3-element Sidon subsets of {0,...,7}), built with standard
multi-controlled-Z-per-marked-state construction (X gates on 0-bits, a
multi-controlled Z, then undo the X gates) -- a genuine reversible oracle,
not a shortcut. We run the optimal number of Grover iterations for the
resulting marked-fraction and simulate on the ideal AerSimulator.

PASS criterion: after measurement, the most frequent 8-bit outcome must
decode to a 3-element subset of {0,...,7} that the classical brute-force
check (run independently, on the measured bitstring, in this script)
confirms is actually a Sidon set -- i.e. the quantum search must have
amplified the classically-verified solution space.
"""

import itertools
import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

UNIVERSE = list(range(8))  # {0, ..., 7}
SUBSET_SIZE = 3
N_QUBITS = len(UNIVERSE)  # 8


def is_sidon(subset):
    """Classically decide: are all pairwise sums a+b (a<=b) in `subset` distinct?"""
    elems = sorted(subset)
    sums = []
    for i in range(len(elems)):
        for j in range(i, len(elems)):
            sums.append(elems[i] + elems[j])
    return len(sums) == len(set(sums))


def bits_to_subset(bitstring):
    """bitstring is qiskit-order (little-endian: bit 0 = qubit 0 = leftmost char reversed)."""
    # Qiskit returns classical register bitstrings with qubit 0 as the
    # rightmost character. We index by universe element i <-> qubit i.
    return {i for i in UNIVERSE if bitstring[N_QUBITS - 1 - i] == "1"}


def subset_to_bitstring(subset):
    chars = ["0"] * N_QUBITS
    for i in subset:
        chars[N_QUBITS - 1 - i] = "1"
    return "".join(chars)


def classical_good_subsets():
    """Brute-force, from first principles, every SUBSET_SIZE-subset of UNIVERSE
    and return the ones that are Sidon sets."""
    good = []
    for combo in itertools.combinations(UNIVERSE, SUBSET_SIZE):
        if is_sidon(combo):
            good.append(frozenset(combo))
    return good


def build_oracle(good_bitstrings, n_qubits):
    """Phase-flip exactly the basis states in good_bitstrings (qiskit bit order)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bitstring in good_bitstrings:
        # qiskit bitstring: leftmost char = highest-index qubit, rightmost = qubit 0
        zero_qubits = [q for q in range(n_qubits) if bitstring[n_qubits - 1 - q] == "0"]
        for q in zero_qubits:
            qc.x(q)
        # multi-controlled Z across all n_qubits marking this exact state
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
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


def main():
    good_subsets = classical_good_subsets()
    good_bitstrings = {subset_to_bitstring(s) for s in good_subsets}

    total_states = 2 ** N_QUBITS
    m = len(good_bitstrings)
    print(f"Universe size: {N_QUBITS}, subset size: {SUBSET_SIZE}")
    print(f"Total 3-subsets of {{0..7}}: {math.comb(N_QUBITS, SUBSET_SIZE)}")
    print(f"Classically-found Sidon 3-subsets: {m}")
    assert 0 < m < total_states, "degenerate search space"

    # Optimal number of Grover iterations for marked fraction m/N over the
    # full 2^n-dimensional space.
    theta = math.asin(math.sqrt(m / total_states))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Grover iterations: {iterations}")

    oracle = build_oracle(good_bitstrings, N_QUBITS)
    diffuser = build_diffuser(N_QUBITS)

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))
    for _ in range(iterations):
        qc.compose(oracle, range(N_QUBITS), inplace=True)
        qc.compose(diffuser, range(N_QUBITS), inplace=True)
    qc.measure(range(N_QUBITS), range(N_QUBITS))

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc.decompose(), shots=shots).result()
    counts = result.get_counts()

    # Aggregate probability mass landing on classically-verified good states.
    good_mass = sum(c for bs, c in counts.items() if bs in good_bitstrings)
    good_fraction = good_mass / shots

    top_bitstring, top_count = Counter(counts).most_common(1)[0]
    top_subset = bits_to_subset(top_bitstring)

    # Independent classical re-check of the top measured outcome (not just a
    # membership lookup in good_bitstrings -- actually recompute is_sidon).
    top_is_valid_size = len(top_subset) == SUBSET_SIZE
    top_is_sidon = is_sidon(top_subset) if top_is_valid_size else False

    print(f"Top measured bitstring: {top_bitstring} -> subset {sorted(top_subset)}")
    print(f"Top outcome count: {top_count}/{shots}")
    print(f"Probability mass on classically-verified Sidon sets: {good_fraction:.3f}")
    print(f"Top outcome has size {SUBSET_SIZE}: {top_is_valid_size}")
    print(f"Top outcome is Sidon (independently re-checked classically): {top_is_sidon}")

    passed = top_is_valid_size and top_is_sidon and good_fraction > 0.5
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
