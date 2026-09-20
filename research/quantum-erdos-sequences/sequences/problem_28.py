"""
Erdos problem #28 (from erdosproblems.com / manman4/erdosproblems data/problems.yaml,
entry `number: "28"`) is a $500 open problem on additive bases / additive number
theory (tags: "number theory", "additive basis"). Its `oeis` field in the source
data is literally `["N/A"]` -- there is no OEIS sequence attached to problem 28,
so this script cannot test "membership of an integer in the OEIS sequence for
problem 28" as instructed, because no such sequence exists to test against.

HONEST LIMITATION: since there is no OEIS id for problem 28, this script instead
tests a genuine, finite, computable property drawn directly from the problem's
own subject matter (additive bases / Sidon sets, i.e. B_2 sets: sets of integers
whose pairwise sums are all distinct). This is exactly the kind of object Erdos
problem 28's area (additive basis) is about, even though it is not the literal
open conjecture itself -- proving/disproving problem 28 in general is of course
far beyond a small quantum circuit.

Concrete finite instance and classical property tested
--------------------------------------------------------
S = [1, 2, 5, 11] is a Sidon set (all C(4,2) = 6 pairwise sums a_i + a_j, i < j,
are distinct). We classically enumerate the 6 pairs, index them 0..5 (fits in
3 qubits, states 6 and 7 unused/never marked), and pick the pair whose sum
equals a chosen target value T = 6 (the unique pair (1, 5), i.e. pair index 2
under pair list order [(1,2),(1,5),(1,11),(2,5),(2,11),(5,11)] -> sums
[3,6,12,7,13,16] -> index of sum==6 is index 1).

We classically verify the target index is unique (this is what "Sidon set"
guarantees: at most one pair has any given sum), then build a genuine 3-qubit
Grover search circuit whose oracle marks exactly that pair index, run it on the
ideal AerSimulator, and check that the circuit's measured most-likely outcome
equals the classically-computed target pair index. This directly exercises
quantum search (Grover) over the search space defined by the additive-basis
(Sidon set) structure that problem 28's tag names.

PASS/FAIL: PASS iff the most frequently measured Grover output bitstring decodes
to the same pair index as the classical brute-force computation.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_setup():
    """Classically compute the Sidon set pairs, sums, and the unique target index."""
    S = [1, 2, 5, 11]
    pairs = [(S[i], S[j]) for i in range(len(S)) for j in range(i + 1, len(S))]
    sums = [a + b for a, b in pairs]

    # Verify Sidon property: all pairwise sums distinct (this is what makes the
    # target index unique, which Grover search relies on).
    assert len(sums) == len(set(sums)), "S is not a Sidon set -- sums collide"

    target_sum = 6
    matches = [i for i, s in enumerate(sums) if s == target_sum]
    assert len(matches) == 1, "target sum must match exactly one pair"
    target_index = matches[0]

    n_qubits = 3  # covers indices 0..7; only 0..5 are valid pair indices
    assert 0 <= target_index < 2 ** n_qubits

    return pairs, sums, target_sum, target_index, n_qubits


def oracle(qc, target_index, n_qubits):
    """Phase-flip the |target_index> basis state (standard Grover oracle)."""
    bits = format(target_index, f"0{n_qubits}b")[::-1]  # little-endian per qubit
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    # multi-controlled Z on all n_qubits (controls = first n-1, target = last)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)


def diffuser(qc, n_qubits):
    """Standard Grover diffusion operator (inversion about the mean)."""
    for q in range(n_qubits):
        qc.h(q)
        qc.x(q)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for q in range(n_qubits):
        qc.x(q)
        qc.h(q)


def build_grover_circuit(target_index, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        oracle(qc, target_index, n_qubits)
        diffuser(qc, n_qubits)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def run():
    pairs, sums, target_sum, target_index, n_qubits = classical_setup()

    N = 2 ** n_qubits
    # optimal number of Grover iterations for 1 marked item out of N
    iterations = max(1, round((np.pi / 4) * np.sqrt(N)))

    qc = build_grover_circuit(target_index, n_qubits, iterations)

    sim = AerSimulator()
    job = sim.run(qc, shots=4096)
    counts = job.result().get_counts()

    # most frequent measured bitstring -> integer index. Qiskit's classical
    # bitstring is "c_{n-1}...c_1 c_0" (rightmost char = qubit 0 = LSB), which
    # is exactly standard binary with qubit0 as the least-significant bit, so
    # the string can be parsed directly as an integer.
    best_bitstring = max(counts, key=counts.get)
    measured_index = int(best_bitstring, 2)

    print(f"Sidon set S = [1, 2, 5, 11]")
    print(f"Pairs (index: pair -> sum): "
          + ", ".join(f"{i}:{p}->{s}" for i, (p, s) in enumerate(zip(pairs, sums))))
    print(f"Target sum = {target_sum}, classical target pair index = {target_index}")
    print(f"Grover iterations used = {iterations}")
    print(f"Measurement counts: {counts}")
    print(f"Most likely measured index = {measured_index}")

    verified = (measured_index == target_index)
    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = run()
    raise SystemExit(0 if ok else 1)
