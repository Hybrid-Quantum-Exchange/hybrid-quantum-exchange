"""
Erdos problem #772 (source: manman4/erdosproblems, data/problems.yaml, entry
"number: \"772\""). Status recorded there: proved (2025-08-31), no prize.
Tags: ["number theory", "sidon sets", "additive combinatorics"].

The YAML's `oeis` field for problem 772 is `["possible"]`, which is not an
actual OEIS id -- it looks like placeholder/junk data in that source file,
not a real sequence reference. Since the problem is explicitly about Sidon
sets (also called B2 sets: sets where all pairwise sums a_i + a_j, i < j,
are distinct), this script instead uses the canonical OEIS sequence for
that exact topic: A005282, the Mian-Chowla sequence, the lexicographically
earliest infinite Sidon set of positive integers, built by always appending
the smallest integer that keeps all pairwise sums distinct.
https://oeis.org/A005282 begins: 1, 2, 3, 5, 8, 13, 21, 31, 45, 66, ...

CLASSICAL PROPERTY TESTED (computed from first principles below, not copied
from OEIS): take the first 4 terms of A005282, S = [1, 2, 3, 5]. Form all
C(4,2) = 6 unordered pairs (i, j) with i < j and their sums S[i] + S[j].
The Sidon-set property being exercised is exactly that these 6 sums are
pairwise distinct (this is verified classically in `classical_check()`
below -- it is not assumed). Because the sums are distinct, exactly one of
the 6 pairs achieves any given target sum T that is actually attained. We
pick T = max(sums) = S[2] + S[3] = 3 + 5 = 8, whose unique witness pair is
(i, j) = (2, 3).

QUANTUM CIRCUIT: Grover's search algorithm over a 3-qubit register whose
8 basis states 0..7 are decoded (index_table below) into the 6 valid pairs
(0..5) and 2 unused/invalid states (6, 7). The oracle is a diagonal phase
oracle built as a multi-controlled-Z (with X-conjugation) that flags only
the single basis state whose decoded pair sums to T = 8. One Grover
iteration (optimal for 1 marked item out of 8) is applied, then the
register is measured. The most frequently measured basis state must decode
to the classically-found witness pair (2, 3).

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Classical part: build the first terms of A005282 (Mian-Chowla) from first
# principles (greedy Sidon construction), then find the witness pair.
# ---------------------------------------------------------------------------

def mian_chowla(n_terms):
    """Greedily build the first n_terms of the Mian-Chowla (Sidon) sequence."""
    seq = [1]
    sums_seen = set()  # pairwise sums a_i + a_j, i < j, of terms already chosen
    candidate = 2
    while len(seq) < n_terms:
        # sums this candidate would introduce with every existing term
        new_sums = [candidate + s for s in seq]
        if len(set(new_sums)) == len(new_sums) and sums_seen.isdisjoint(new_sums):
            seq.append(candidate)
            sums_seen.update(new_sums)
        candidate += 1
    return seq


def classical_check():
    """Build S, verify the Sidon property on it, and find the witness pair
    for the target sum T = max pairwise sum. Returns (S, pairs, index_table,
    target_sum, witness_index)."""
    S = mian_chowla(4)
    assert S == [1, 2, 3, 5], f"unexpected Mian-Chowla prefix: {S}"

    pairs = list(combinations(range(len(S)), 2))  # [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    sums = [S[i] + S[j] for (i, j) in pairs]

    # Verify the Sidon (B2) property: all pairwise sums distinct.
    assert len(set(sums)) == len(sums), f"S is not Sidon: sums={sums}"

    target_sum = max(sums)
    witnesses = [idx for idx, s in enumerate(sums) if s == target_sum]
    assert len(witnesses) == 1, "target sum should have a unique witness pair"
    witness_index = witnesses[0]

    index_table = {i: pairs[i] for i in range(len(pairs))}  # 0..5 -> (i,j)

    return S, pairs, sums, index_table, target_sum, witness_index


S, PAIRS, SUMS, INDEX_TABLE, TARGET_SUM, WITNESS_INDEX = classical_check()


# ---------------------------------------------------------------------------
# Quantum part: Grover search over the 3-qubit index register for the one
# basis state (0..7) equal to WITNESS_INDEX.
# ---------------------------------------------------------------------------

N_QUBITS = 3  # 8 basis states, 6 valid pair-indices (0..5), 2 unused (6,7)


def oracle(qc, marked_index, qubits):
    """Flip the phase of |marked_index> among the n-qubit basis states."""
    bits = format(marked_index, f"0{len(qubits)}b")  # MSB..LSB matches qubit order below
    # Qiskit bit ordering: qubits[0] is the least-significant bit of the
    # measured integer, so reverse `bits` to align with qubits[0..n-1].
    bits = bits[::-1]
    zero_positions = [q for q, b in zip(qubits, bits) if b == "0"]

    for q in zero_positions:
        qc.x(q)
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for q in zero_positions:
        qc.x(q)


def diffuser(qc, qubits):
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


def build_grover_circuit(marked_index, n_qubits):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))

    qc.h(qubits)

    # Optimal number of Grover iterations for 1 marked item out of 2**n.
    n_states = 2 ** n_qubits
    iterations = max(1, round((np.pi / 4) * np.sqrt(n_states / 1)))

    for _ in range(iterations):
        oracle(qc, marked_index, qubits)
        diffuser(qc, qubits)

    qc.measure(qubits, qubits)
    return qc, iterations


def run_grover():
    qc, iterations = build_grover_circuit(WITNESS_INDEX, N_QUBITS)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit count keys are big-endian strings of the classical register;
    # our register bit q_i is the i-th bit counting from qubit 0 (LSB) at
    # the *right* end of the string, so reverse to read as q0 q1 q2 (MSB
    # first among our indices) -> easier: just int(key, 2) with key as-is
    # already gives the correct integer since we measured qubits in order
    # (q0 -> clbit0, ... ) and Qiskit prints clbit (n-1) ... clbit0.
    best_key = max(counts, key=counts.get)
    measured_index = int(best_key, 2)

    return measured_index, counts, iterations, shots


def main():
    print("Erdos problem #772 -- Sidon set / Mian-Chowla (A005282) Grover search")
    print(f"  Sequence prefix S (A005282, first 4 terms): {S}")
    print(f"  Pairs (i<j) and sums: "
          + ", ".join(f"{INDEX_TABLE[k]}->{SUMS[k]}" for k in range(len(PAIRS))))
    print(f"  Target sum T = {TARGET_SUM}, classical witness pair index = "
          f"{WITNESS_INDEX} (pair {INDEX_TABLE[WITNESS_INDEX]})")

    measured_index, counts, iterations, shots = run_grover()

    top_count = counts[max(counts, key=counts.get)]
    print(f"  Grover iterations used: {iterations}")
    print(f"  Measurement counts: {counts}")
    print(f"  Most frequent measured index: {measured_index} "
          f"({top_count}/{shots} shots, decoded pair "
          f"{INDEX_TABLE.get(measured_index, 'invalid/unused state')})")

    passed = (measured_index == WITNESS_INDEX)
    if passed:
        print("PASS: quantum Grover search recovered the classical Sidon-pair witness.")
    else:
        print("FAIL: quantum result does not match the classical witness.")

    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
