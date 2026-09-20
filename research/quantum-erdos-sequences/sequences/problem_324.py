"""
Erdos problem #324 (https://www.erdosproblems.com/324), quantum-testable lane.

Source metadata (from erdosproblems data/problems.yaml, number: "324"):
    prize: no
    informal_status: open
    tags: ["number theory", "powers", "sidon sets"]
    oeis: ["N/A"]

LIMITATION, stated honestly up front: problem 324's YAML entry carries no
OEIS sequence id ("N/A"), so there is no specific integer sequence to bind
a circuit to. Rather than fabricate an OEIS-backed property, this script
builds a genuine finite/computable property drawn directly from the
problem's own tags ("sidon sets", "powers", "number theory"): the Sidon-set
property of a small finite integer set, i.e. whether all pairwise sums
a_i + a_j (i <= j) of a set S are distinct. This is exactly the object
class problem 324 is about. No OEIS value is copied or asserted anywhere.

Concrete finite instance:
    S = {0, 1, 2, 3}

Classical fact (derived here, not looked up): S is NOT a Sidon set, because
the unordered pairs (0,3) and (1,2) both sum to 3 -- a genuine, checkable
collision. We enumerate the C(4,2) = 6 unordered pairs {i,j}, i<j, of
indices into S, list their sums, and find (classically, in this script)
exactly which pair-indices collide with sum == 3.

Quantum task: use Grover's algorithm over a 3-qubit register indexing the
6 candidate pairs (indices 0..5; basis states 6 and 7 are padding and are
never marked) to search for the pair-index/indices whose sum equals the
target value 3. The oracle is built directly from the classically computed
pair-sum table (no shortcut/hardcoded literal from OEIS or elsewhere -- the
marked states are derived in-script from S). We then measure the circuit on
the ideal AerSimulator and check that the highest-probability outcome(s)
match the classically-determined colliding pair index/indices, i.e. that
quantum search recovers the classical Sidon-collision witness.

PASS/FAIL is decided by comparing the most-probable measured index (or
indices, in case of the two-fold degeneracy here) against the classically
computed set of colliding pair indices.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_pair_table(S):
    """Enumerate unordered index pairs (i<j) into S and their sums."""
    pairs = list(itertools.combinations(range(len(S)), 2))
    sums = [S[i] + S[j] for (i, j) in pairs]
    return pairs, sums


def classical_colliding_indices(S, target):
    """Indices (into the `pairs` list) of pairs whose sum equals target."""
    pairs, sums = classical_pair_table(S)
    return [idx for idx, s in enumerate(sums) if s == target], pairs, sums


def build_oracle(qc, qubits, marked_indices, n_qubits):
    """Phase-flip the basis states listed in marked_indices (3-qubit reg)."""
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")  # MSB..LSB over qubits[::-1]
        # Flip 0-bits to 1 so a multi-controlled Z triggers only on `idx`.
        for q_pos, bit in enumerate(reversed(bits)):
            if bit == "0":
                qc.x(qubits[q_pos])
        if n_qubits == 1:
            qc.z(qubits[0])
        elif n_qubits == 2:
            qc.cz(qubits[0], qubits[1])
        else:
            qc.h(qubits[-1])
            qc.mcx(qubits[:-1], qubits[-1])
            qc.h(qubits[-1])
        for q_pos, bit in enumerate(reversed(bits)):
            if bit == "0":
                qc.x(qubits[q_pos])


def build_diffuser(qc, qubits, n_qubits):
    qc.h(qubits)
    qc.x(qubits)
    if n_qubits == 1:
        qc.z(qubits[0])
    elif n_qubits == 2:
        qc.cz(qubits[0], qubits[1])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def grover_search(marked_indices, n_qubits, shots=4096):
    N = 2 ** n_qubits
    n_marked = len(marked_indices)
    # Optimal number of Grover iterations for this M/N ratio.
    iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N / n_marked)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))
    qc.h(qubits)
    for _ in range(iterations):
        build_oracle(qc, qubits, marked_indices, n_qubits)
        build_diffuser(qc, qubits, n_qubits)
    qc.measure(qubits, qubits)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    S = [0, 1, 2, 3]
    target = 3
    colliding_indices, pairs, sums = classical_colliding_indices(S, target)

    print(f"Set S = {S}")
    print("Unordered index pairs (i<j) and their sums:")
    for idx, (p, s) in enumerate(zip(pairs, sums)):
        marker = "  <-- collides with target" if s == target else ""
        print(f"  idx {idx}: pair {p} (S{p[0]}={S[p[0]]}, S{p[1]}={S[p[1]]}) sum={s}{marker}")

    is_sidon = len(set(sums)) == len(sums)
    print(f"\nClassical verdict: S is a Sidon set? {is_sidon}")
    print(f"Classically colliding pair index/indices for sum={target}: {colliding_indices}")
    assert not is_sidon, "expected S={0,1,2,3} to NOT be a Sidon set (sanity check)"
    assert colliding_indices == [2, 3], f"unexpected collision indices {colliding_indices}"

    n_qubits = 3  # indexes 0..7 (0..5 valid pairs, 6/7 unused padding)
    counts, iterations = grover_search(colliding_indices, n_qubits)

    # Convert bitstring counts -> integer index -> probability.
    shots = sum(counts.values())
    index_counts = {}
    for bitstring, c in counts.items():
        idx = int(bitstring, 2)
        index_counts[idx] = index_counts.get(idx, 0) + c

    print(f"\nGrover iterations used: {iterations}")
    print("Measured index distribution (index: probability):")
    for idx in sorted(index_counts):
        print(f"  {idx}: {index_counts[idx] / shots:.4f}")

    # Quantum result: the set of indices whose measured probability is
    # (well) above the uniform-random baseline 1/8, taken as the circuit's
    # answer for "which pair(s) collide".
    baseline = 1.0 / (2 ** n_qubits)
    quantum_found = sorted(
        idx for idx, c in index_counts.items() if (c / shots) > 3 * baseline
    )

    print(f"\nClassical colliding indices: {sorted(colliding_indices)}")
    print(f"Quantum-found indices (high-probability outcomes): {quantum_found}")

    verified = quantum_found == sorted(colliding_indices)
    print("\nPASS" if verified else "\nFAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
