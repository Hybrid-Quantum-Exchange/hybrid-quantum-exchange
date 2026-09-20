"""
Erdos problem #658 (from erdosproblems.com / manman4/erdosproblems data/problems.yaml,
entry "number: \"658\"", tags: ["additive combinatorics"], oeis: ["possible"]).

LIMITATION, stated up front: problem #658's YAML record does not carry a real
OEIS sequence id -- its `oeis` field is the literal placeholder string
"possible", not an identifier like "A123456". So there is no concrete OEIS
sequence to target for this entry, and this script cannot honestly claim to
test "the" sequence for problem 658. Rather than fabricate an OEIS id or copy
a value with no traceable source, this script instead tests a small, genuine,
computable property drawn from the problem's own tag, "additive
combinatorics": Sidon-set-ness of a subset of a finite set of integers.

A Sidon set (a B_2 set) is a set of integers such that all pairwise sums of
two (not necessarily distinct... here: distinct) elements are different.
Formally, for S = {s_1, ..., s_k}, S is Sidon iff all sums s_i + s_j with
i <= j are pairwise distinct (equivalently, all pairwise differences are
distinct). This is a textbook finite/computable additive-combinatorics
property, in the spirit of problem 658's tag.

Concrete finite instance used here:
  Ground set: {0, 1, 2, 3, 4, 5}  (6 elements -> 2^6 = 64 subsets, indexed
  0..63 by which elements are included, bit i of the index <-> element i
  present).
  Property tested: "index i encodes a subset of size exactly 3 that is a
  Sidon subset of {0,...,5}". (An unrestricted "is Sidon" predicate over all
  64 subsets was tried first and rejected: it marks a *majority* -- 36 of 64
  -- of subsets, since small subsets of consecutive integers are Sidon far
  more often than not, leaving nothing interesting for Grover's algorithm to
  amplify. Fixing the subset size to 3 keeps the marked fraction modest.)

The classical answer (which subset-indices of size 3 are Sidon) is computed
directly in this script by brute force over all 64 subsets, from first
principles (no OEIS value is copied). This gives a marked set M of "good"
indices.

Quantum method: Grover's algorithm on 6 qubits (search space size N=64).
  - The oracle is built by classically evaluating the Sidon predicate for all
    16 basis states and applying a phase flip to exactly the marked ones,
    via a diagonal unitary (Qiskit's UnitaryGate on the phase pattern). This
    is a legitimate Grover oracle: it flips the sign of every marked
    computational basis state and nothing else.
  - The diffusion operator is the standard Grover diffuser about the uniform
    superposition.
  - The optimal number of Grover iterations for N=16 and M=|marked| is
    computed from the standard formula r ~ (pi/4) * sqrt(N/M).
  - The circuit is run on the ideal AerSimulator (statevector-exact, shots
    sampled from it) and the most frequently measured index is compared
    against the classically-computed marked set.

PASS/FAIL: the script prints PASS iff the most-sampled measurement outcome
of the Grover circuit is a member of the classically verified Sidon-subset
set M (i.e. Grover actually amplified a true positive), and prints details
of the classical vs. quantum answer either way.
"""

import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator


def is_sidon(subset):
    """Classically decide whether `subset` (a tuple/list of distinct ints)
    is a Sidon set: all pairwise sums s_i + s_j (i <= j) are distinct."""
    sums = []
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            sums.append(subset[i] + subset[j])
    return len(sums) == len(set(sums))


def classical_marked_indices(n_elements=6, target_size=3):
    """Brute-force, from first principles, every subset of {0,...,n-1} and
    return the set of subset-indices (bit i of index <-> element i present)
    whose subset (a) has exactly `target_size` elements and (b) is a Sidon
    set. Restricting to a fixed size keeps the marked fraction of the 2^n
    search space in a range Grover's algorithm can usefully amplify (an
    unrestricted "is Sidon" predicate turns out to mark a *majority* of
    small subsets of consecutive integers, which was checked directly and
    rejected as a poor instance for demonstrating amplitude amplification)."""
    ground = list(range(n_elements))
    marked = set()
    for idx in range(2 ** n_elements):
        subset = [ground[b] for b in range(n_elements) if (idx >> b) & 1]
        if len(subset) == target_size and is_sidon(subset):
            marked.add(idx)
    return marked


def build_oracle(n_qubits, marked_indices):
    """Diagonal phase-flip oracle: -1 on marked computational basis states,
    +1 elsewhere. Built as an explicit unitary from the classically computed
    marked set, then wrapped as a Qiskit gate."""
    dim = 2 ** n_qubits
    diag = np.ones(dim, dtype=complex)
    for idx in marked_indices:
        diag[idx] = -1.0
    return UnitaryGate(np.diag(diag), label="Oracle")


def build_diffuser(n_qubits):
    """Standard Grover diffusion operator: phase flip about the uniform
    superposition, i.e. 2|s><s| - I where |s> is the all-plus state."""
    dim = 2 ** n_qubits
    diag = -np.ones(dim, dtype=complex)
    diag[0] = 1.0  # after H^n ... H^n sandwich, index 0 keeps its sign
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    # multi-controlled Z on all-ones (i.e. phase flip |11...1>)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    n_elements = 6
    target_size = 3
    n_qubits = n_elements
    N = 2 ** n_qubits

    marked = classical_marked_indices(n_elements, target_size)
    M = len(marked)
    assert 0 < M < N, "degenerate marked set; instance choice is bad"

    print(f"Ground set: {{0,...,{n_elements - 1}}}  (N = {N} subsets, {n_qubits} qubits)")
    print(f"Target: subsets of size {target_size} that are Sidon sets")
    print(f"Classically computed marked indices: {sorted(marked)}")
    print(f"|M| = {M}")

    # Optimal number of Grover iterations
    r = max(1, round((np.pi / 4) * np.sqrt(N / M)))
    print(f"Grover iterations: r = {r}")

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    qreg = QuantumRegister(n_qubits, "q")
    qc = QuantumCircuit(qreg)
    qc.h(range(n_qubits))
    for _ in range(r):
        qc.append(oracle, qreg[:])
        qc.compose(diffuser, qreg[:], inplace=True)
    qc.measure_all()

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit orders classical bits with qubit 0 as the rightmost bit; build
    # the integer index consistently with how idx was constructed above
    # (bit i of idx <-> element i, i.e. qubit i).
    def bitstring_to_index(bitstring):
        bits = bitstring.replace(" ", "")
        # bitstring is c[n-1]...c[1]c[0]
        return int(bits[::-1], 2)

    index_counts = {}
    for bitstring, c in counts.items():
        idx = bitstring_to_index(bitstring)
        index_counts[idx] = index_counts.get(idx, 0) + c

    top_index, top_count = max(index_counts.items(), key=lambda kv: kv[1])
    top_subset = [b for b in range(n_elements) if (top_index >> b) & 1]
    total_marked_prob = sum(index_counts.get(i, 0) for i in marked) / shots

    print(f"Most frequent measured index: {top_index} "
          f"(subset {top_subset}), count {top_count}/{shots}")
    print(f"Total probability mass on marked (Sidon) indices: "
          f"{total_marked_prob:.3f}")

    classical_is_sidon = top_index in marked
    verified = classical_is_sidon and (top_index in marked) == is_sidon(top_subset)

    if classical_is_sidon and top_count / shots > 1.0 / N * 2:
        print("PASS")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
