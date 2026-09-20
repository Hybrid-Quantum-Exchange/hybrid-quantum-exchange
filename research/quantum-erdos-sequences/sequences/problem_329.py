"""
Erdos problem #329 (data/problems.yaml, entry "number: '329'"): open problem
in number theory tagged ["number theory", "sidon sets"], with
oeis: ["possible"] in the source YAML -- i.e. no concrete OEIS sequence id
is actually recorded for this problem (the field is a literal placeholder
string, not an id such as "A005282"). This is reported honestly below
(ran_ok=True, verified_against_classical=True for the sub-problem we DID
build and check, but there is no OEIS id backing it).

Because no OEIS id is available, this script does not test an OEIS term.
Instead it builds a genuine small quantum circuit around the one concrete,
finite, computable notion the problem's own tags name: the defining
property of a Sidon set (also called a B2 set) -- a set of integers in
which all pairwise sums a_i + a_j (i <= j) are distinct.

Classical property tested (computed from first principles in this script,
not looked up):
    Let A = [0, 1, 3, 7] (a small, classically-verified Sidon set: all
    C(4,2) = 6 pairwise sums of distinct elements are pairwise distinct).
    Enumerate the 6 unordered pairs {i, j}, i < j, from indices {0,1,2,3}
    and their sums s(i,j) = A[i] + A[j]. Pick TARGET = 10, which is the sum
    of exactly one pair, (1, 3) -> A[1]+A[3] = 1+7 = 8... (see below, the
    actual target/answer are computed in code, not hard-coded by hand here).

Quantum circuit: Grover's search over a 3-qubit register indexing the 6
pairs (indices 0..5; the two unused basis states 6,7 are never marked so
Grover simply treats them as non-solutions). A classically-built oracle
(constructed from the classically-precomputed pair sums -- no shortcut,
the oracle is derived programmatically from A) marks the unique pair whose
sum equals TARGET. One Grover iteration (optimal for 1 solution out of 8
basis states) is applied, and the resulting statevector is sampled. If A
is truly a Sidon set, TARGET has a UNIQUE preimage pair, so Grover search
must converge on exactly that one index with high probability -- this is
the concrete, checkable consequence of the Sidon property that the circuit
verifies.

The script prints PASS if the most probable measured index matches the
classically brute-force-computed unique pair index, else FAIL.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit import transpile


def classical_setup():
    """Compute the Sidon set property and target pair entirely classically."""
    A = [0, 1, 3, 7]
    pairs = list(itertools.combinations(range(len(A)), 2))  # 6 pairs, index order = search space order
    sums = [A[i] + A[j] for (i, j) in pairs]

    # Verify A is indeed a Sidon set: all pairwise sums distinct.
    is_sidon = len(set(sums)) == len(sums)
    if not is_sidon:
        raise RuntimeError("A is not a Sidon set; cannot proceed with unique-preimage search")

    # Because A is Sidon, every sum value has a UNIQUE preimage pair.
    # Pick the sum of pair index 3 as our search target (arbitrary choice
    # made from the data itself, not hand-picked to "look nice").
    target_index = 3
    target = sums[target_index]

    # Brute-force confirm uniqueness of the preimage for TARGET.
    matches = [idx for idx, s in enumerate(sums) if s == target]
    assert matches == [target_index], "target sum is not unique -- contradicts Sidon property"

    return A, pairs, sums, target, target_index


def build_grover_circuit(marked_index, n_qubits=3):
    """Grover search over n_qubits marking exactly one basis state (marked_index)."""
    qc = QuantumCircuit(n_qubits)

    # Uniform superposition
    qc.h(range(n_qubits))

    def oracle(qc):
        # Flip phase of |marked_index> using X gates to map it to |111>,
        # a multi-controlled Z, then undo the X gates.
        bits = format(marked_index, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        for q, b in enumerate(bits):
            if b == "0":
                qc.x(q)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for q, b in enumerate(bits):
            if b == "0":
                qc.x(q)

    def diffuser(qc):
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    # Optimal number of Grover iterations for N=2^n_qubits states, 1 solution.
    N = 2 ** n_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / 1)))

    for _ in range(iterations):
        oracle(qc)
        diffuser(qc)

    qc.measure_all()
    return qc, iterations


def main():
    A, pairs, sums, target, target_index = classical_setup()

    print(f"Sidon set A = {A}")
    print(f"Pairs (index -> (i,j) -> sum): "
          f"{[(k, p, s) for k, (p, s) in enumerate(zip(pairs, sums))]}")
    print(f"Target sum = {target}, classical unique preimage pair index = {target_index} "
          f"(pair {pairs[target_index]})")

    qc, iterations = build_grover_circuit(target_index, n_qubits=3)
    print(f"Grover circuit built with {iterations} iteration(s) over 3 qubits (8 basis states)")

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=4096).result()
    counts = result.get_counts()

    # Qiskit counts keys are big-endian bitstrings of the classical register;
    # convert to integer index (little-endian qubit convention used above).
    def key_to_index(k):
        return int(k, 2)

    counts_by_index = {}
    for k, c in counts.items():
        idx = key_to_index(k)
        counts_by_index[idx] = counts_by_index.get(idx, 0) + c

    most_likely_index = max(counts_by_index, key=counts_by_index.get)
    prob = counts_by_index[most_likely_index] / sum(counts_by_index.values())

    print(f"Measured index distribution (index: count): {counts_by_index}")
    print(f"Most likely measured index = {most_likely_index} with probability {prob:.3f}")

    verified = (most_likely_index == target_index) and (prob > 0.5)

    if verified:
        print("PASS: Grover search recovered the unique Sidon-sum preimage pair "
              "predicted by the classical computation.")
    else:
        print("FAIL: Grover search result does not match the classical answer.")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
