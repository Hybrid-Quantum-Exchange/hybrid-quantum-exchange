"""
Erdos problem #869 (per data/problems.yaml in the erdosproblems repository,
https://github.com/manman4/erdosproblems): tags = ["number theory",
"additive basis"], oeis = ["N/A"] (no OEIS sequence id is recorded for this
problem in the source data as of 2026-09-19).

LIMITATION: because problem #869 carries no OEIS id, there is no specific
integer sequence to target with a "is n in the sequence" style query. Rather
than fabricate a sequence membership fact that has no basis in the problem's
actual data, this script instead builds a genuine, problem-relevant quantum
computation around the one substantive piece of metadata that *is* present:
the tag "additive basis". This is a best-effort adjacent construction, not a
verification of problem #869 itself, and should be read as such.

Classical property being tested
--------------------------------
Fix N = 16 and the explicit finite set

    A = [0, 1, 2, 4, 8, 9, 10, 11]   (indexed 0..7, so a 3-bit index each)

We ask: which ordered pairs of indices (i, j) in {0,...,7}^2 satisfy

    (A[i] + A[j]) mod N == t,   where t = 5

This is exactly the elementary question at the heart of "is A an additive
basis (of order 2) for Z_N": for a fixed target residue t, which pairs of
basis elements sum to it. It is finite, fully computable, and small enough
to brute force classically (64 candidate pairs) -- which the script does
first, from first principles, to get the ground truth.

Ground truth (computed in this script, not copied from anywhere):
    solutions = [(1,3), (3,1), (6,7), (7,6)]   (4 of the 64 pairs)
because A[1]+A[3] = 1+4 = 5, A[6]+A[7] = 10+11 = 21 = 5 (mod 16).

Quantum approach
-----------------
Grover search over the 6-qubit space of index pairs (i, j), each 3 bits,
64 basis states total. The oracle is a phase oracle built directly from the
classically-precomputed solution set above (multi-controlled Z gates marking
exactly those 4 basis states whose computational-basis index equals one of
the 4 solutions). This is standard practice for small Grover instances where
the "black box" function A[i]+A[j] mod N == t is evaluated once classically
to build the oracle, exactly as one would derive a boolean formula's oracle
from its truth table.

With M = 4 marked states out of Ns = 64, the optimal number of Grover
iterations is floor(pi/4 * sqrt(Ns/M)) = floor(pi/4 * sqrt(16)) = 3.

The circuit is run on the ideal AerSimulator with 4096 shots. PASS requires
that every one of the 4 most-measured outcomes decodes (via the classical
index -> (i, j) -> A[i]+A[j] mod N) to a value equal to t, and that all 4
classically-known solutions appear among the most-frequent measured outcomes
with combined probability mass well above what uniform random sampling of
64 states would give (uniform baseline ~= 4/64 = 6.25%).
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_ground_truth(A, N, t):
    """Brute-force, from first principles, every (i, j) with A[i]+A[j] == t (mod N)."""
    solutions = []
    for i, j in itertools.product(range(len(A)), range(len(A))):
        if (A[i] + A[j]) % N == t:
            solutions.append((i, j))
    return solutions


def index_pair_to_int(i, j):
    """Pack a (i, j) pair, each in [0,8), into a single 6-bit integer i*8 + j."""
    return i * 8 + j


def int_to_index_pair(n):
    return (n // 8, n % 8)


def build_oracle(marked_ints, n_qubits):
    """Phase oracle flipping the sign of exactly the basis states in marked_ints."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked_ints:
        bits = format(m, f"0{n_qubits}b")
        # Flip qubits that should be 0 so the marked state looks like all-ones,
        # apply a multi-controlled Z (via H-MCX-H on the last qubit), then flip back.
        flip_qubits = [q for q, b in enumerate(reversed(bits)) if b == "0"]
        for q in flip_qubits:
            qc.x(q)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for q in flip_qubits:
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
    A = [0, 1, 2, 4, 8, 9, 10, 11]
    N = 16
    t = 5
    n_qubits = 6  # 3 bits for i, 3 bits for j

    solutions = classical_ground_truth(A, N, t)
    marked_ints = sorted(index_pair_to_int(i, j) for (i, j) in solutions)
    print(f"Classical solutions (i, j) with A[i]+A[j] == {t} (mod {N}): {solutions}")
    print(f"Marked computational-basis integers: {marked_ints}")

    Ns = 2 ** n_qubits
    M = len(marked_ints)
    iterations = max(1, round((math.pi / 4) * math.sqrt(Ns / M)))
    print(f"Search space size = {Ns}, marked states = {M}, Grover iterations = {iterations}")

    oracle = build_oracle(marked_ints, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical register bit order is little-endian in the returned
    # bitstrings (qubit 0 is the rightmost character); int_to_index_pair
    # expects the same convention used when building the oracle (q0..q5 = LSB..MSB).
    decoded_counts = {}
    for bitstring, c in counts.items():
        n = int(bitstring, 2)
        decoded_counts[n] = decoded_counts.get(n, 0) + c

    top4 = sorted(decoded_counts.items(), key=lambda kv: -kv[1])[:4]
    print("Top 4 measured outcomes (int: count):", top4)

    top4_ints = {n for n, _ in top4}
    top4_mass = sum(c for n, c in top4 if n in marked_ints) / shots

    all_top4_are_solutions = all(n in marked_ints for n in top4_ints)
    all_solutions_in_top4 = set(marked_ints) == top4_ints
    # sanity re-check: decode top4 back to (i, j) and re-verify classically
    verified = True
    for n in top4_ints:
        i, j = int_to_index_pair(n)
        if not (0 <= i < len(A) and 0 <= j < len(A)):
            verified = False
            break
        if (A[i] + A[j]) % N != t:
            verified = False
            break

    baseline = M / Ns
    print(f"Fraction of shots landing on a true solution among top-4 outcomes: {top4_mass:.3f} "
          f"(uniform-random baseline would be ~{baseline:.3f})")

    passed = (
        all_top4_are_solutions
        and all_solutions_in_top4
        and verified
        and top4_mass > 3 * baseline
    )

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    main()
