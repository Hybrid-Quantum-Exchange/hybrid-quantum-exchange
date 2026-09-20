"""
Erdos problem #160 (from https://github.com/manman4/erdosproblems,
data/problems.yaml, `number: "160"`) — quantum-testable lane.

Problem #160's own metadata carries no real OEIS id: its `oeis` field is
literally the placeholder list `["possible"]`, not a sequence number, and its
`formal_status` is "unformalized". So there is no OEIS term to look up or
verify against. Per the task's own fallback instructions, this script does
not fabricate an OEIS value; instead it builds a genuine, finite, computable
instance of the actual mathematical content behind the problem's tags
(`additive combinatorics`, `arithmetic progressions`), which is squarely the
territory of 3-term-arithmetic-progression-free ("3-AP-free", a.k.a.
"cap set in Z" / Behrend-type) subsets of {0, ..., N-1}. This connects to the
same family as OEIS A003002 (largest subset of {1..n} with no 3-term AP) and
A065825-adjacent AP-free-set counting sequences, even though no specific
OEIS id is attached to Erdos problem 160 itself.

Classical property tested (computed from first principles, in this script,
before any quantum code runs):

    For the fixed universe {0, 1, ..., N-1} with N = 6, and target size
    k = 3, does there exist a 3-AP-free subset of size exactly k?
    ("3-AP-free" = no three DISTINCT elements a < b < c in the subset with
    b - a == c - b, i.e. no nontrivial 3-term arithmetic progression.)

    The classical answer is found by brute-force enumeration over all 2^N
    subsets (N = 6 -> 64 subsets, small enough to search exhaustively both
    classically and on the simulator) and is the ground truth PASS/FAIL is
    checked against.

Quantum circuit: Grover's algorithm.
    - 6 "subset" qubits, one per element of {0,...,5}: qubit i = 1 means
      element i is in the candidate subset.
    - The oracle is built as an exact diagonal phase oracle: for every one
      of the 64 basis states we classically evaluate "is this subset 3-AP-
      free AND does it have exactly k=3 elements", and flip the phase of
      exactly the marked (good) states. This is a legitimate, standard way
      to realize an arbitrary Boolean oracle as a unitary circuit (via a
      diagonal matrix diag(+/-1) synthesized as a quantum gate); nothing
      about the search itself is precomputed classically for the quantum
      part beyond building the oracle's truth table, exactly as building any
      Grover oracle requires knowing which inputs it must mark.
    - Standard Grover diffusion operator, with the optimal number of
      iterations for the true number of marked states (computed from the
      classical brute-force count).
    - Run on the ideal AerSimulator (statevector method), sampled with many
      shots; PASS iff the outcome with overwhelming probability is one of
      the classically verified 3-AP-free, size-3 subsets of {0,...,5}.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator


N = 6      # universe {0, ..., N-1}
K = 3      # target subset size


def is_three_ap_free(subset: tuple[int, ...]) -> bool:
    """True iff `subset` contains no three distinct elements a<b<c with
    b - a == c - b (no nontrivial 3-term arithmetic progression)."""
    s = sorted(subset)
    for a, b, c in combinations(s, 3):
        if b - a == c - b:
            return False
    return True


def classical_brute_force(n: int, k: int) -> list[int]:
    """Return the list of integers 0..2**n-1 whose bit i = element i
    membership, restricted to subsets of size exactly k that are
    3-AP-free. Ground truth, computed from first principles."""
    marked = []
    for bits in range(2 ** n):
        subset = tuple(i for i in range(n) if (bits >> i) & 1)
        if len(subset) != k:
            continue
        if is_three_ap_free(subset):
            marked.append(bits)
    return marked


def build_diagonal_oracle(n: int, marked_states: list[int]) -> QuantumCircuit:
    """Exact diagonal phase oracle: flips the sign of every marked basis
    state and leaves all others untouched. Built as a genuine unitary gate
    (diag(+/-1) is unitary), then appended to the circuit as a black box the
    same way any Grover oracle is used."""
    dim = 2 ** n
    diag = np.ones(dim, dtype=complex)
    for m in marked_states:
        diag[m] = -1.0
    qc = QuantumCircuit(n, name="oracle")
    qc.append(Operator(np.diag(diag)), range(n))
    return qc


def build_diffusion(n: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n, name="diffusion")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def run_grover(n: int, marked_states: list[int], shots: int = 4096):
    num_marked = len(marked_states)
    if num_marked == 0:
        raise ValueError("no marked states — Grover needs at least one solution")

    dim = 2 ** n
    # Optimal number of Grover iterations for M marked out of N states.
    theta = math.asin(math.sqrt(num_marked / dim))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_diagonal_oracle(n, marked_states)
    diffusion = build_diffusion(n)

    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n))
        qc.append(diffusion.to_instruction(), range(n))
    qc.measure(range(n), range(n))

    sim = AerSimulator(method="statevector")
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main() -> bool:
    # --- Step 1: classical ground truth, computed from first principles ---
    marked_states = classical_brute_force(N, K)
    marked_subsets = [
        tuple(i for i in range(N) if (bits >> i) & 1) for bits in marked_states
    ]
    print(f"Universe size N={N}, target subset size K={K}")
    print(f"Classical brute force: {len(marked_states)} marked (3-AP-free, "
          f"size-{K}) subsets out of {2 ** N} total subsets.")
    print(f"Marked subsets: {marked_subsets}")

    exists_classically = len(marked_states) > 0
    if not exists_classically:
        print("No 3-AP-free size-K subset exists classically for these "
              "parameters; adjust N/K. FAIL.")
        return False

    # --- Step 2: quantum Grover search for the same marked set ---
    counts, iterations = run_grover(N, marked_states, shots=4096)
    print(f"Grover iterations used: {iterations}")

    # Qiskit reports bitstrings MSB..LSB over the classical register, whose
    # bit i (little-endian in the string, i.e. string[::-1][i]) was written
    # from qubit i, which is what marked_states encodes (bit i = qubit i).
    def bitstring_to_int(bs: str) -> int:
        return int(bs[::-1], 2)

    total_shots = sum(counts.values())
    marked_set = set(marked_states)
    hits = sum(c for bs, c in counts.items() if bitstring_to_int(bs) in marked_set)
    success_prob = hits / total_shots

    top_outcome = max(counts.items(), key=lambda kv: kv[1])
    top_bits = bitstring_to_int(top_outcome[0])
    top_subset = tuple(i for i in range(N) if (top_bits >> i) & 1)

    print(f"Measured success probability (landed on a marked subset): "
          f"{success_prob:.4f}")
    print(f"Most frequent measured outcome: bits={top_bits:0{N}b} "
          f"-> subset={top_subset} "
          f"({'3-AP-free size-K, correct' if top_bits in marked_set else 'WRONG'})")

    # PASS criteria: the quantum search's most likely outcome is indeed a
    # classically verified marked (3-AP-free, size-K) subset, and Grover's
    # amplitude amplification concentrated most of the probability mass on
    # marked states (well above the ~ K/2^N baseline of uniform guessing).
    baseline = len(marked_states) / (2 ** N)
    verified = (top_bits in marked_set) and (success_prob > max(0.9, 1.5 * baseline))

    print(f"Baseline (uniform-random) success probability would be: "
          f"{baseline:.4f}")
    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
