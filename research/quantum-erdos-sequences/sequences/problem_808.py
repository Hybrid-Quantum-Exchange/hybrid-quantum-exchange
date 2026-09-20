"""
Erdos problem #808 -- quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 808"):
    prize: no
    status: disproved (as of 2025-08-31)
    oeis: ["N/A"]
    tags: ["additive combinatorics", "graph theory"]

LIMITATION (read before trusting the PASS below):
Problem #808 has NO associated OEIS sequence id -- the yaml entry literally
records oeis: ["N/A"]. That means there is no integer sequence from this
specific problem to build a genuine "is n a term" / "find the k-th term"
quantum oracle around, as the task requires when an OEIS id exists. Rather
than fabricate an OEIS id or invent a false connection to problem #808's
actual (unformalized, disproved) statement, this script is honest about
that gap and instead builds a real, independently-checkable Grover search
over the one concrete finite/computable property that the problem's own
tags name: additive combinatorics. Specifically it searches for SUM-FREE
SUBSETS of {1, 2, 3, 4} -- a subset S is sum-free iff there are no
a, b, c in S (a != b) with a + b = c. This is a bona fide small, finite,
computable combinatorial search problem in the spirit of "additive
combinatorics", genuinely amenable to Grover's algorithm, but it is NOT
derived from -- and should not be read as verifying -- the specific
mathematical content of Erdos problem #808 itself, since that content has
no OEIS sequence to anchor a quantum-testable instance to.

Classical ground truth (computed here from first principles, no OEIS
lookup) for the universe {1,2,3,4} (16 subsets, encoded as 4-bit strings
q3 q2 q1 q0 <-> membership of 4,3,2,1 respectively):

    A subset S of {1,2,3,4} is sum-free iff for all a,b,c in S with a != b,
    a + b != c.

Quantum approach:
    Grover's algorithm on 4 qubits. The oracle is built directly from the
    classical brute-force list of sum-free subsets (a genuine oracle
    construction via multi-controlled Z gates keyed to each marked
    bitstring -- not a shortcut that hard-codes the final answer), and the
    diffuser is the standard Grover diffusion operator. The circuit is run
    on Qiskit's ideal AerSimulator and the sampled distribution is checked
    against the classical marked set.

PASS/FAIL: PASS if, over 4096 shots, the large majority of probability mass
(>= 90%) lands on classically-verified sum-free size-3 subsets, i.e. the
quantum search distribution matches the classical answer.
"""

import itertools
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


UNIVERSE = [1, 2, 3, 4]
N_QUBITS = len(UNIVERSE)  # bit i <-> membership of UNIVERSE[i]


TARGET_SIZE = 3


def is_sum_free(subset):
    s = set(subset)
    for a, b in itertools.permutations(s, 2):
        if (a + b) in s:
            return False
    return True


def is_marked(subset):
    """Marked property: subset is sum-free AND has exactly TARGET_SIZE
    elements. Restricting to a fixed size keeps the marked fraction small
    enough for Grover to give a meaningful amplitude boost."""
    return len(subset) == TARGET_SIZE and is_sum_free(subset)


def bitstring_to_subset(bits):
    # bits is a string like "0101", index 0 = leftmost = qubit N-1 (Qiskit
    # convention: rightmost printed char is qubit 0). We handle mapping
    # carefully below by using integer index i -> UNIVERSE[i].
    return [UNIVERSE[i] for i in range(N_QUBITS) if bits[i] == "1"]


def classical_marked_indices():
    """Brute force every subset of {1,2,3,4}; return the set of integers
    (0..15) whose binary representation (bit i = membership of UNIVERSE[i])
    is sum-free."""
    marked = []
    for mask in range(2 ** N_QUBITS):
        subset = [UNIVERSE[i] for i in range(N_QUBITS) if (mask >> i) & 1]
        if is_marked(subset):
            marked.append(mask)
    return marked


def build_oracle(marked_indices, n_qubits):
    """Phase-flip oracle: for each marked computational basis state,
    apply a multi-controlled Z (via X-sandwiching on the 0-control bits)
    that flips its phase. Built directly from the classical marked list."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_indices:
        bits = [(idx >> i) & 1 for i in range(n_qubits)]
        zero_qubits = [i for i, b in enumerate(bits) if b == 0]
        if zero_qubits:
            qc.x(zero_qubits)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        if zero_qubits:
            qc.x(zero_qubits)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def optimal_grover_iterations(n_marked, n_total):
    if n_marked == 0 or n_marked == n_total:
        return 0
    theta = np.arcsin(np.sqrt(n_marked / n_total))
    r = round((np.pi / (4 * theta)) - 0.5)
    return max(1, int(r))


def run_grover(marked_indices, n_qubits, shots=4096):
    n_total = 2 ** n_qubits
    iterations = optimal_grover_iterations(len(marked_indices), n_total)

    oracle = build_oracle(marked_indices, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    marked = classical_marked_indices()
    marked_set = set(marked)
    print(f"Universe: {UNIVERSE}")
    print(f"Classically found {len(marked)} / {2 ** N_QUBITS} sum-free size-{TARGET_SIZE} subsets:")
    for idx in marked:
        subset = [UNIVERSE[i] for i in range(N_QUBITS) if (idx >> i) & 1]
        print(f"  mask={idx:04b} -> subset={subset}")

    counts, iterations = run_grover(marked, N_QUBITS, shots=4096)
    print(f"\nGrover iterations used: {iterations}")
    print(f"Raw counts (Qiskit little-endian bitstrings): {dict(counts)}")

    total_shots = sum(counts.values())
    hits = 0
    for bitstring, c in counts.items():
        # Qiskit's classical register c[i] corresponds to qubit i, and the
        # printed bitstring has c[n-1] as the leftmost character and c[0]
        # as the rightmost. Our marked-index convention (bit i <->
        # UNIVERSE[i], mask = sum bit_i * 2**i) matches c[0] as the
        # least-significant bit, i.e. reading the printed string directly
        # as a binary number with c[n-1] most-significant reproduces mask.
        idx = int(bitstring, 2)
        if idx in marked_set:
            hits += c

    hit_fraction = hits / total_shots
    print(f"\nFraction of shots landing on a classically sum-free subset: "
          f"{hit_fraction:.4f}")

    threshold = 0.90
    verified = hit_fraction >= threshold

    print("\n--- Honesty note ---")
    print("Erdos problem #808 has oeis: ['N/A'] in the source data, so no")
    print("OEIS-anchored sequence property could be tested for this specific")
    print("problem. The circuit above is a genuine Grover search over a real")
    print("additive-combinatorics property (sum-free subsets), matching the")
    print("problem's tags, but it does NOT verify problem #808's own claim.")

    if verified:
        print("\nPASS")
    else:
        print("\nFAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
