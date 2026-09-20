"""
Erdos problem #14 (erdosproblems.com), OEIS id used: A143824.

erdosproblems.com problem 14 is about Sidon sets (tags: "number theory",
"sidon sets", "additive combinatorics"): a Sidon set is a set of positive
integers such that all pairwise sums of two *distinct* elements are
distinct. The problem asks about the growth of B(n), the size of the
largest Sidon set contained in {1, ..., n}.

Classical property tested here (computed from first principles in this
script, not copied from OEIS):

    For n = 6, what is B(6), the size of the largest Sidon subset of
    {1, 2, 3, 4, 5, 6}, and which subsets attain it?

We brute-force this classically over all 2^6 = 64 subsets: a subset S is
Sidon iff for every pair of distinct elements a < b in S, the sums a+b are
pairwise distinct across all such pairs. This gives B(6) = 4, attained by
exactly 8 of the 64 subsets (verified in `classical_search` below).

Quantum part: we build a genuine Grover search circuit over the 6-qubit
space of subset-indicator bitstrings. The oracle is a phase oracle that
flips the sign of exactly the computational basis states corresponding to
the (classically precomputed) maximum-size Sidon subsets -- this is the
standard "oracle marks a known solution set" construction used to
demonstrate the quadratic speedup of Grover's algorithm; the point being
verified quantumly is that Grover amplification concentrates measurement
probability onto that marked set, i.e. that the circuit (state prep +
oracle + diffuser, iterated the Grover-optimal number of times) actually
does what the algorithm claims for this instance.

We run the circuit on the ideal AerSimulator and check that the most
frequent measured bitstrings are exactly members of the classically
computed maximum Sidon set collection, and that this matches with high
probability mass. PASS/FAIL is printed based on that comparison.
"""

import math
from itertools import combinations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N = 6  # ground set {1, ..., 6}
ELEMS = list(range(1, N + 1))


def is_sidon(subset):
    """True iff all pairwise sums of two distinct elements of `subset` are distinct."""
    sums = set()
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def classical_search():
    """Brute force over all 2^N subsets of {1..N}; return (B(N), list of marked masks)."""
    best = 0
    masks = []
    for mask in range(1 << N):
        subset = [ELEMS[i] for i in range(N) if mask & (1 << i)]
        if is_sidon(subset):
            if len(subset) > best:
                best = len(subset)
                masks = [mask]
            elif len(subset) == best:
                masks.append(mask)
    return best, masks


def build_oracle(n_qubits, marked_masks):
    """Phase oracle: flip the sign of each marked computational basis state.

    For each marked bitmask, X-gate the 0-bits so the target pattern maps to
    |11...1>, apply a multi-controlled Z (phase flip) on all n qubits, then
    undo the X-gates. This is a standard construction for a Grover oracle
    over a known, explicit marked set.
    """
    qc = QuantumCircuit(n_qubits, name="oracle")
    for mask in marked_masks:
        zero_bits = [q for q in range(n_qubits) if not (mask & (1 << q))]
        for q in zero_bits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_bits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def optimal_iterations(n_qubits, n_marked):
    N_total = 2 ** n_qubits
    theta = math.asin(math.sqrt(n_marked / N_total))
    r = round((math.pi / (4 * theta)) - 0.5)
    return max(1, r)


def main():
    best_size, marked_masks = classical_search()
    print(f"Classical: B({N}) = {best_size}, "
          f"{len(marked_masks)} maximum Sidon subsets out of {1 << N} total subsets.")
    print(f"Marked masks: {sorted(marked_masks)}")

    n_qubits = N
    iterations = optimal_iterations(n_qubits, len(marked_masks))
    print(f"Grover iterations used: {iterations}")

    oracle = build_oracle(n_qubits, marked_masks)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit order: rightmost char of the bitstring is qubit 0.
    def bitstring_to_mask(bs):
        return int(bs[::-1], 2)

    marked_set = set(marked_masks)
    marked_hits = sum(c for bs, c in counts.items() if bitstring_to_mask(bs) in marked_set)
    marked_probability = marked_hits / shots

    # Determine the most frequent measured outcomes (top len(marked_masks) by count).
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_masks = [bitstring_to_mask(bs) for bs, _ in sorted_counts[: len(marked_masks)]]
    top_masks_all_marked = all(m in marked_set for m in top_masks)

    print(f"Probability mass on marked (maximum Sidon set) states: {marked_probability:.4f}")
    print(f"Top {len(marked_masks)} measured outcomes are all marked: {top_masks_all_marked}")

    # Uniform-random baseline for comparison: len(marked)/2^n.
    baseline = len(marked_masks) / (1 << n_qubits)
    amplified = marked_probability > 3 * baseline

    verified = top_masks_all_marked and amplified
    print(f"Classical answer: B({N}) = {best_size}, marked count = {len(marked_masks)}")
    print(f"Quantum (Grover) result amplified onto classical answer's marked set: {verified}")

    print("PASS" if verified else "FAIL")


if __name__ == "__main__":
    main()
