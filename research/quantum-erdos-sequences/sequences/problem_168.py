"""
Erdos problem #168 (erdosproblems.com/168, additive combinatorics, tags:
["additive combinatorics"], oeis: A004059, A057561, A094708, A386439).

Problem #168 concerns the greedy Sidon set (greedy B_2 sequence): the
sequence built by starting from 1 and always adjoining the smallest integer
that keeps the set a Sidon set (a set in which all pairwise sums a_i + a_j,
i <= j, are distinct). OEIS A004059 is this greedy Sidon sequence; its first
terms are 1, 2, 4, 8, 13, 21, 31, 45, 66, 81, 97, ...

Classical property tested here (derived and checked from first principles in
this script, not copied from OEIS):

  Take the first 4 terms of A004059: S = [1, 2, 4, 8].
  The Sidon property means every pairwise sum a_i + a_j (i < j) is distinct.
  We verify that classically below (brute force over all 6 unordered pairs),
  which confirms S really is Sidon for this small instance.

  We then pick one specific target sum T = 10 (the sum of the pair (2, 8),
  i.e. indices (1, 3) in S). Because S is Sidon, exactly one of the 6 pairs
  sums to T. This is a genuine finite search problem: "find the unique pair
  of indices (i, j), i < j, in S with S[i] + S[j] == T" -- exactly the kind
  of unstructured search Grover's algorithm solves.

Quantum circuit: we enumerate the 6 unordered index pairs of {0,1,2,3} as
3-qubit basis states 0..5 (states 6 and 7 are unused/never marked). A Grover
oracle marks the one state whose corresponding pair sums to T = 10. We run
Grover's algorithm with the optimal number of iterations for a search space
of size 8 with 1 marked item, on the ideal AerSimulator, and check that the
most frequently measured state is exactly the classically-computed answer.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator


def classical_check():
    """Verify the Sidon property of S=[1,2,4,8] and find the target pair."""
    S = [1, 2, 4, 8]
    pairs = list(combinations(range(len(S)), 2))  # 6 pairs, indices into S
    sums = [S[i] + S[j] for (i, j) in pairs]

    # Sidon property: all pairwise sums distinct.
    assert len(sums) == len(set(sums)), "S is not a Sidon set!"

    T = 10
    matches = [k for k, s in enumerate(sums) if s == T]
    assert len(matches) == 1, "target sum must be uniquely attained (Sidon)"
    target_index = matches[0]  # index into `pairs`/basis-state number
    target_pair = pairs[target_index]

    return S, pairs, sums, T, target_index, target_pair


def build_grover_circuit(n_qubits, marked_index, n_iterations):
    """Grover search over n_qubits marking a single basis state index."""
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Uniform superposition
    qc.h(range(n_qubits))

    # Oracle: flip phase of |marked_index> using X-sandwiched multi-controlled Z
    def apply_oracle(qc, marked_index, n_qubits):
        bits = format(marked_index, f"0{n_qubits}b")[::-1]  # little-endian
        for q, b in enumerate(bits):
            if b == "0":
                qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
            qc.append(mcz, list(range(n_qubits)))
        for q, b in enumerate(bits):
            if b == "0":
                qc.x(q)

    # Diffuser: inversion about the mean
    def apply_diffuser(qc, n_qubits):
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        if n_qubits == 1:
            qc.z(0)
        else:
            mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
            qc.append(mcz, list(range(n_qubits)))
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    for _ in range(n_iterations):
        apply_oracle(qc, marked_index, n_qubits)
        apply_diffuser(qc, n_qubits)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    S, pairs, sums, T, target_index, target_pair = classical_check()

    print(f"Greedy Sidon set (A004059) first terms: {S}")
    print(f"Pairs and sums: {list(zip(pairs, sums))}")
    print(f"Target sum T = {T}")
    print(f"Classical answer: pair index {target_index} -> S{target_pair} = "
          f"{S[target_pair[0]]} + {S[target_pair[1]]} = {T}")

    n_qubits = 3  # search space size 8, covers pair indices 0..5
    N = 2 ** n_qubits
    n_iterations = max(1, round((math.pi / 4) * math.sqrt(N / 1)))
    print(f"Running Grover search with {n_iterations} iteration(s) over "
          f"{N} basis states...")

    qc = build_grover_circuit(n_qubits, target_index, n_iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=4096).result()
    counts = result.get_counts()

    # Most frequent measured bitstring -> integer (qiskit bitstrings are
    # big-endian string of qubit n_qubits-1 ... 0, but we measured in order
    # so int(..., 2) directly gives our little-endian-encoded index value
    # since format() above used the same convention consistently via Qiskit's
    # own little-endian classical register ordering).
    best_bitstring = max(counts, key=counts.get)
    measured_index = int(best_bitstring, 2)

    total_shots = sum(counts.values())
    p_correct = counts.get(best_bitstring, 0) / total_shots

    print(f"Measured counts: {counts}")
    print(f"Most likely measured index: {measured_index} "
          f"(probability {p_correct:.3f})")

    passed = (measured_index == target_index) and (p_correct > 0.5)

    if passed:
        print("PASS")
    else:
        print("FAIL")

    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
