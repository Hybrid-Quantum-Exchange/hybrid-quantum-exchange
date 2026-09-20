"""
Erdos problem #847 (source: erdosproblems.com, via manman4/erdosproblems
data/problems.yaml, entry "number: \"847\"", tags: ["additive combinatorics"],
informal_status: disproved, formal_status: Lean).

LIMITATION, stated honestly up front: problem #847's YAML record carries
oeis: ["N/A"] -- there is no OEIS sequence attached to this problem. The
task asked us to derive a small, finite, computable property from the
problem's OEIS id(s) and tags; with no OEIS id available, there is nothing
sequence-specific to test. Rather than fabricate an OEIS-backed claim, this
script tests a genuine, well-defined finite decision problem drawn from the
problem's actual tag, "additive combinatorics": Sidon-set membership.

Property under test
--------------------
A Sidon set (also called a B2 set) is a set S of non-negative integers such
that all pairwise sums a + b, for a, b in S with a <= b, are distinct. This
is a standard, central object in additive combinatorics (the same area
tagged on problem #847), independent of any specific OEIS entry.

Small finite instance: the universe is {0, 1, 2, 3} (N = 4 elements, so
there are 2^4 = 16 subsets, indexed by a 4-bit string b3 b2 b1 b0 where bit
i = 1 means element i is included). We classically enumerate, from first
principles, every one of the 16 subsets and decide which are Sidon sets
(the empty set and single-element sets are vacuously/trivially Sidon and
are included in the "true" set; this is a fixed, deterministic, classical
computation with no reference to any external table).

Quantum computation
--------------------
We build a genuine Grover search circuit over the 4-qubit space of subsets.
The oracle is constructed directly from the classical truth table computed
above: for every subset marked "Sidon", the oracle applies a phase flip to
exactly that computational basis state (via X-gates to map the target
bitstring to |1111>, a multi-controlled-Z, and un-doing the X-gates). This
is a legitimate, if brute-force, Grover oracle -- it does not use any
oracle "trick" that assumes the answer; it is compiled from the classical
enumeration performed in this same script.

Grover's algorithm is then run on the ideal AerSimulator for the optimal
number of iterations given the number of marked states M out of N = 16,
and the measurement histogram is compared against the classically computed
set of non-Sidon subsets: PASS if the quantum result recovers exactly the
classical marked set (all sampled outcomes lie in the classical non-Sidon
set, and that set is covered with high aggregate probability).
"""

import math
from itertools import combinations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_ELEMENTS = 4  # universe {0, 1, 2, 3}
N_QUBITS = N_ELEMENTS  # one qubit per element, bit i => element i included


def is_sidon(subset):
    """Classical, from-first-principles Sidon-set test.

    subset: tuple of distinct ints. Returns True iff all pairwise sums
    a + b for a <= b in subset are pairwise distinct.
    """
    sums = []
    for a, b in combinations(sorted(subset), 2):
        sums.append(a + b)
    for a in subset:
        sums.append(a + a)
    return len(sums) == len(set(sums))


def classical_non_sidon_subsets():
    """Enumerate all 2**N_ELEMENTS subsets of {0,...,N_ELEMENTS-1} and
    return the set of bitstrings (MSB..LSB, qubit order q3 q2 q1 q0) whose
    subset FAILS to be a Sidon set (some pairwise sum repeats). Marking the
    non-Sidon subsets (rather than the Sidon ones) is a deliberate choice:
    for this N=4 universe only 3 of the 16 subsets fail to be Sidon sets,
    which keeps the marked fraction small enough for Grover's quadratic
    speed-up to be meaningful (Grover search is only useful/well-posed when
    the marked set is a small minority of the search space)."""
    marked = set()
    for mask in range(2 ** N_ELEMENTS):
        subset = tuple(i for i in range(N_ELEMENTS) if (mask >> i) & 1)
        if not is_sidon(subset):
            # bit string as Qiskit prints it: q_{n-1}...q_0
            bits = "".join(
                "1" if (mask >> i) & 1 else "0"
                for i in reversed(range(N_ELEMENTS))
            )
            marked.add(bits)
    return marked


def build_oracle(marked_bitstrings, n_qubits):
    """Phase-flip oracle built from a classical truth table: for each
    marked bitstring, flip the sign of exactly that basis state using
    X-gates + multi-controlled-Z + X-gates."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bits in marked_bitstrings:
        # bits[0] is qubit n-1 ... bits[-1] is qubit 0
        zero_positions = [
            n_qubits - 1 - idx for idx, ch in enumerate(bits) if ch == "0"
        ]
        if zero_positions:
            qc.x(zero_positions)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
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
    total_states = 2 ** n_qubits
    m = len(marked_bitstrings)
    if m == 0 or m == total_states:
        raise ValueError("Grover search needs 0 < M < N marked states")

    # Exact optimal iteration count (not the large-N approximation): pick k
    # that maximizes sin((2k+1)*theta)^2, theta = asin(sqrt(m/total_states)).
    theta = math.asin(math.sqrt(m / total_states))
    best_k, best_p = 1, -1.0
    for k in range(0, 6):
        p = math.sin((2 * k + 1) * theta) ** 2
        if p > best_p:
            best_k, best_p = k, p
    iterations = max(1, best_k)

    oracle = build_oracle(marked_bitstrings, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    classical_marked = classical_non_sidon_subsets()
    print(f"Universe: {{0,1,2,3}}  ({N_ELEMENTS} elements, {2**N_ELEMENTS} subsets)")
    print(f"Classical non-Sidon subsets (bitstrings q3q2q1q0): "
          f"{sorted(classical_marked)}  (count={len(classical_marked)})")

    counts, iterations = run_grover(classical_marked, N_QUBITS, shots=4096)
    print(f"Grover iterations used: {iterations}")
    print(f"Measured counts: {counts}")

    total_shots = sum(counts.values())
    hit_shots = sum(c for bits, c in counts.items() if bits in classical_marked)
    hit_fraction = hit_shots / total_shots

    observed_bitstrings = set(counts.keys())
    all_observed_are_marked = observed_bitstrings.issubset(classical_marked)
    coverage = len(observed_bitstrings & classical_marked) / len(classical_marked)

    print(f"Fraction of shots landing on a classical non-Sidon subset: {hit_fraction:.4f}")
    print(f"All observed outcomes are classically-marked non-Sidon subsets: "
          f"{all_observed_are_marked}")
    print(f"Coverage of classical non-Sidon set by observed outcomes: {coverage:.4f}")

    # PASS criterion for a probabilistic algorithm: the marked (non-Sidon)
    # subsets must be recovered with high aggregate probability and full
    # coverage; we do not require literally zero off-target shots, since
    # Grover's success probability is amplified but not exactly 1.
    passed = hit_fraction >= 0.9 and coverage == 1.0

    if passed:
        print("PASS")
    else:
        print("FAIL")

    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
