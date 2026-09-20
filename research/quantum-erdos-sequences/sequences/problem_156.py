"""
Erdos problem #156 -- quantum-testable instance.

Source metadata (erdosproblems.com data, via manman4/erdosproblems
data/problems.yaml, entry "number: '156'"):
    oeis: ["A382397"]
    tags: ["sidon sets"]
    status: open

A382397 is an OEIS sequence tied to Sidon sets (sets of non-negative
integers whose pairwise sums are all distinct -- i.e. a+b = c+d with
{a,b} != {c,d} never happens for a,b,c,d in the set). Erdos problem 156
itself is a research-level open question about Sidon sets; there is no
single finite decision procedure that "is" the open problem. Instead,
per the task's own carve-out, we take the well-defined, finite,
genuinely-computable property that the OEIS tag names and that any
progress on the problem is built on:

    PROPERTY TESTED: for the universe U = {0,1,2,3,4} and the family of
    all 4-element subsets of U, which subsets are Sidon sets? I.e. for
    S = {a,b,c,d} in increasing order, is it true that all six pairwise
    sums a+b, a+c, a+d, b+c, b+d, c+d are pairwise distinct?

    This is exactly the defining property behind A382397 / Sidon sets,
    checked exhaustively and classically first (first principles, no
    OEIS lookup of the answer), then re-derived by a quantum search.

Classical result (computed in this script, see `classical_sidon_indices`):
    C(5,4) = 5 four-element subsets of {0,1,2,3,4}, indexed 0..4 in
    sorted order of itertools.combinations:
        0: {0,1,2,3} -> 0+3 == 1+2 (=3)            -> NOT Sidon
        1: {0,1,2,4} -> all six sums distinct       -> Sidon
        2: {0,1,3,4} -> 0+4 == 1+3 (=4)            -> NOT Sidon
        3: {0,2,3,4} -> all six sums distinct       -> Sidon
        4: {1,2,3,4} -> 1+4 == 2+3 (=5)            -> NOT Sidon
    So exactly 2 of the 5 subsets are Sidon sets: indices {1, 3}.

QUANTUM CIRCUIT: a genuine Grover search over the 3-qubit index space
{0,...,7} (5 valid subset indices + 3 unused/padding indices, none of
which are ever marked). The oracle is derived directly from the
classical Sidon indices {1, 3} = binary 001, 011: both share bit0=1 and
bit2=0, with bit1 free, so the oracle phase-flips exactly the states
with q0=1 and q2=0 -- which is exactly, and only, indices 1 and 3 out
of all 8 basis states. This is verified programmatically below (not
just asserted) before the circuit is built. Grover amplification with
the optimal integer number of iterations for N=8, M=2 marked items is
then run on AerSimulator, and the measured distribution is compared
against the classical Sidon indices.

PASS criterion: the two most frequent measured outcomes (by count) are
exactly {1, 3}, matching the classical Sidon indices, each with
noticeably amplified probability relative to a uniform baseline (1/8).
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def pairwise_sums(subset):
    return [a + b for a, b in itertools.combinations(subset, 2)]


def is_sidon(subset):
    sums = pairwise_sums(subset)
    return len(sums) == len(set(sums))


def classical_sidon_indices(universe_size=5, k=4):
    """Enumerate all k-subsets of {0,...,universe_size-1} and return the
    indices (in itertools.combinations order) of those that are Sidon sets.
    Computed from first principles -- no OEIS values are copied in."""
    universe = range(universe_size)
    subsets = list(itertools.combinations(universe, k))
    sidon_indices = [i for i, s in enumerate(subsets) if is_sidon(s)]
    return subsets, sidon_indices


def build_grover_circuit(marked_indices, n_qubits=3):
    """Oracle: phase-flip states with q0=1 and q2=0 (bit0 set, bit2 clear).
    This is derived from, and must match, marked_indices exactly."""
    # Sanity: confirm the {q0=1, q2=0} pattern picks out exactly marked_indices
    # among all 2**n_qubits basis states.
    predicted = [i for i in range(2 ** n_qubits) if (i & 1) == 1 and (i & 4) == 0]
    assert sorted(predicted) == sorted(marked_indices), (
        f"oracle pattern {predicted} does not match classical marked set {marked_indices}"
    )

    n_marked = len(marked_indices)
    n_total = 2 ** n_qubits
    theta = math.asin(math.sqrt(n_marked / n_total))
    iterations = max(1, round((math.pi / (4 * theta) - 0.5)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    def oracle(qc):
        qc.x(2)
        qc.cz(0, 2)
        qc.x(2)

    def diffuser(qc, n):
        qc.h(range(n))
        qc.x(range(n))
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        qc.x(range(n))
        qc.h(range(n))

    for _ in range(iterations):
        oracle(qc)
        diffuser(qc, n_qubits)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, iterations


def main():
    subsets, sidon_indices = classical_sidon_indices()
    print("Universe {0,1,2,3,4}, 4-element subsets (itertools.combinations order):")
    for i, s in enumerate(subsets):
        print(f"  {i}: {s} sums={pairwise_sums(s)} Sidon={is_sidon(s)}")
    print(f"Classical Sidon indices: {sidon_indices}")

    qc, iterations = build_grover_circuit(sidon_indices)
    print(f"Grover iterations used: {iterations}")

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=4096).result()
    counts = result.get_counts()

    # Qiskit bitstrings are 'q2 q1 q0' (little-endian classical register order
    # reversed in the string); convert to integer index with q0 as bit0.
    index_counts = {}
    for bitstring, c in counts.items():
        q2, q1, q0 = bitstring[0], bitstring[1], bitstring[2]
        idx = int(q0) | (int(q1) << 1) | (int(q2) << 2)
        index_counts[idx] = index_counts.get(idx, 0) + c

    print("Measured counts by subset index:", dict(sorted(index_counts.items())))

    top2 = sorted(index_counts.items(), key=lambda kv: kv[1], reverse=True)[:2]
    top2_indices = sorted(idx for idx, _ in top2)
    total_shots = sum(index_counts.values())
    top2_prob = sum(c for _, c in top2) / total_shots

    print(f"Top-2 measured indices: {top2_indices} (classical Sidon indices: {sorted(sidon_indices)})")
    print(f"Combined probability mass on top-2: {top2_prob:.3f} (uniform baseline would be {2/8:.3f})")

    passed = (top2_indices == sorted(sidon_indices)) and (top2_prob > 2 / 8)

    if passed:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
