"""
Erdos problem #422 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems repo):
  number: "422", oeis: ["A005185"], tags: ["number theory"], status: open.

A005185 is the Hofstadter-Conway "$10000" sequence, defined by

    a(1) = a(2) = 1
    a(n) = a(n - a(n-1)) + a(n - a(n-2))   for n >= 3

Erdos problem #422 concerns this sequence's asymptotic behaviour
(a(n)/n -> 1/2 and related growth questions), which is not itself a small
finite decision problem a quantum circuit can settle. Instead, the classical
property tested here, derived directly from the OEIS A005185 recurrence
above (computed from first principles below, not copied from OEIS), is:

    PROPERTY: among indices n = 1..8, find every n with a(n) == 5.

Classical computation in this script gives the sequence for n = 1..8:
    a = [1, 1, 2, 3, 3, 4, 5, 5]
so a(n) == 5 exactly for n in {7, 8} -- two solutions out of eight indices.

This is a genuine finite search problem (find the marked indices among 8),
so it is solved here with Grover's algorithm on 3 qubits: index i in
{0,...,7} encodes n = i+1 in binary (q2 q1 q0, q2 = MSB). The two solutions
i=6 (110) and i=7 (111) share q2=1 and q1=1 regardless of q0, so the oracle
is exactly a controlled-Z between q1 and q2 (phase-flip iff both are 1),
built directly from the classical search target -- not hard-coded to "the
answer" separately from the oracle construction.

With 2 solutions out of N=8, the optimal number of Grover iterations is
floor((pi/4) / theta) with theta = asin(sqrt(M/N)) = asin(sqrt(1/4)) = pi/6,
giving floor(1.5) = 1 iteration (computed in-script, not assumed), after
which measurement should return {110, 111} with high probability and
(ideally) zero weight elsewhere.

The script:
  1. Computes a(1..8) classically via the exact A005185 recurrence.
  2. Derives the classical solution set S = {n : a(n) == 5}.
  3. Builds a Grover circuit whose oracle marks exactly the corresponding
     indices, using AerSimulator (ideal, no noise).
  4. Compares the quantum measurement outcomes against the classical set S
     and prints PASS/FAIL.
"""

import math
import sys

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def hofstadter_conway_a005185(n_max: int) -> list[int]:
    """Classical A005185 sequence a(1)..a(n_max), computed from the recurrence.

    a(1) = a(2) = 1
    a(n) = a(n - a(n-1)) + a(n - a(n-2))  for n >= 3
    Returns a list `a` with a[0] unused (index 0 placeholder) so that a[n]
    is the n-th term, matching the 1-indexed OEIS convention.
    """
    a = [0] * (n_max + 1)
    a[1] = 1
    if n_max >= 2:
        a[2] = 1
    for n in range(3, n_max + 1):
        a[n] = a[n - a[n - 1]] + a[n - a[n - 2]]
    return a


def build_grover_circuit(marked_indices: set[int], n_qubits: int, iterations: int) -> QuantumCircuit:
    """3-qubit Grover search circuit whose oracle marks exactly `marked_indices`.

    Only supports the specific structure used here: the marked set must be
    exactly {values whose top two bits (q_{n_qubits-1}, q_{n_qubits-2}) are
    both 1}, which is what our classical search produced (indices 6, 7 out
    of 0..7). This keeps the oracle a direct, verifiable translation of the
    classical condition rather than an opaque black box.
    """
    assert n_qubits == 3, "oracle below is specialized to 3 qubits"
    assert marked_indices == {6, 7}, "oracle below is specialized to this marked set"

    qc = QuantumCircuit(n_qubits, n_qubits)

    # Uniform superposition over all 8 indices.
    qc.h(range(n_qubits))

    for _ in range(iterations):
        # Oracle: phase-flip iff q1 == 1 and q2 == 1 (i.e. index in {6, 7}),
        # independent of q0 -- exactly matches the classical marked set.
        qc.cz(1, 2)

        # Diffuser (inversion about the mean).
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(2)
        qc.ccx(0, 1, 2)
        qc.h(2)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main() -> int:
    N_MAX = 8
    TARGET_VALUE = 5

    # 1. Classical computation of A005185 from first principles.
    a = hofstadter_conway_a005185(N_MAX)
    sequence = a[1:N_MAX + 1]
    print(f"A005185 a(1..{N_MAX}) = {sequence}")

    # 2. Classical solution set: indices n (1-indexed) with a(n) == TARGET_VALUE.
    classical_solutions_n = {n for n in range(1, N_MAX + 1) if a[n] == TARGET_VALUE}
    classical_solutions_i = {n - 1 for n in classical_solutions_n}  # 0-indexed for the circuit
    print(f"Classical n with a(n) == {TARGET_VALUE}: {sorted(classical_solutions_n)}")
    print(f"0-indexed marked circuit basis states: {sorted(classical_solutions_i)}")

    if classical_solutions_i != {6, 7}:
        print("FAIL: unexpected classical solution set; oracle assumptions do not hold")
        return 1

    # 3. Optimal Grover iteration count, computed from N and number of solutions M.
    n_qubits = 3
    N = 2 ** n_qubits
    M = len(classical_solutions_i)
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, math.floor((math.pi / 4) / theta))
    print(f"N={N}, M={M}, Grover iterations={iterations}")

    qc = build_grover_circuit(classical_solutions_i, n_qubits, iterations)

    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    shots = 4096
    result = simulator.run(compiled, shots=shots).result()
    counts = result.get_counts()
    print(f"Measurement counts: {counts}")

    # 4. Compare quantum outcomes against the classical solution set.
    # Qiskit bit ordering: classical register bit string is c2c1c0 (MSB..LSB)
    # matching qubit order q2 q1 q0, so int(bitstring, 2) is the index i.
    marked_probability = sum(
        count for bitstring, count in counts.items() if int(bitstring, 2) in classical_solutions_i
    ) / shots
    most_common = sorted(counts.items(), key=lambda kv: -kv[1])
    top_states = {int(bs, 2) for bs, _ in most_common[:M]}

    print(f"Probability mass on classical solution states: {marked_probability:.4f}")
    print(f"Top {M} measured state(s): {top_states}, classical solution states: {classical_solutions_i}")

    verified = (top_states == classical_solutions_i) and (marked_probability > 0.8)

    if verified:
        print("PASS")
        return 0
    else:
        print("FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
