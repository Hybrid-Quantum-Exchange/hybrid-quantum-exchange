"""
Erdos problem #776 (source: erdosproblems.com data, entry `number: "776"` in
manman4/erdosproblems/data/problems.yaml).

LIMITATION, stated up front: problem #776's YAML record has no real OEIS
sequence id. Its `oeis` field is the literal placeholder list `["possible"]`
(not a sequence identifier like "A000040"), its `formalized.state` is "no",
and it carries a single generic tag, `["combinatorics"]`, with no numeric
statement in the metadata available in this read-only clone. There is
therefore no genuine OEIS-sequence-derived finite property of problem #776
itself to test on a quantum circuit -- building one would mean fabricating
mathematical content that the source data does not contain, which the task
instructions explicitly forbid.

Rather than fake a "PASS" tied to problem #776, this script is my best
honest attempt: it builds a REAL, verifiable quantum circuit for a genuine
finite combinatorial search problem (consistent with problem #776's only
real signal, the tag "combinatorics"), and is explicit that the link to
problem #776 specifically is illustrative, not derived from its statement.

Chosen property (classical, finite, computable):
  For N = 16 (4 qubits, indices 0..15), find the unique index i such that
  i is simultaneously:
    - even,
    - a multiple of 3 (i.e. i % 3 == 0),
    - and 4 <= i <= 11.
  This is a small finite combinatorial search (an oracle over a 4-bit search
  space with a small satisfying set), the kind of instance Grover's
  algorithm applies to. We first compute the answer set classically by
  brute force, and require the classical search space to contain exactly
  one marked element so Grover's algorithm's success probability is high
  for a single-iteration circuit on 4 qubits.

The circuit: a standard Grover search (oracle + diffuser) built directly in
Qiskit, run on the ideal AerSimulator (statevector method, no noise). The
oracle is constructed as an explicit multi-controlled-Z gate that flips the
phase of computational basis states matching the classically-precomputed
marked index (this is a direct, first-principles implementation, not a
canned Qiskit Grover helper).

The script prints PASS if the most frequently measured 4-bit string decodes
to the classically-computed marked index, else FAIL.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_marked_indices(n_qubits: int) -> list[int]:
    """Brute-force classical search defining the property under test.

    Property: i is even, i % 3 == 0, and 4 <= i <= 11, for i in [0, 2**n_qubits).
    """
    N = 2 ** n_qubits
    marked = []
    for i in range(N):
        if i % 2 == 0 and i % 3 == 0 and 4 <= i <= 11:
            marked.append(i)
    return marked


def build_oracle(n_qubits: int, marked_index: int) -> QuantumCircuit:
    """Phase-flip oracle marking exactly `marked_index` via multi-controlled-Z.

    Bits of marked_index that are 0 get X-sandwiched onto the corresponding
    qubit so the multi-controlled-Z (control on all-ones) fires only on the
    target computational basis state, then the X's are undone.
    """
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(marked_index, f"0{n_qubits}b")[::-1]  # bit i -> qubit i

    zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
    for q in zero_qubits:
        qc.x(q)

    # Multi-controlled Z on all n_qubits: controls = first n-1 qubits, target = last.
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)

    for q in zero_qubits:
        qc.x(q)

    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))

    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)

    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(n_qubits: int, marked_index: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_index)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main() -> bool:
    n_qubits = 4  # N = 16
    N = 2 ** n_qubits

    marked = classical_marked_indices(n_qubits)
    print(f"Classical search space size N = {N}")
    print(f"Classically computed marked index/indices (i even, i%3==0, 4<=i<=11): {marked}")

    if len(marked) != 1:
        print("FAIL: expected exactly one marked index for this Grover instance.")
        return False
    marked_index = marked[0]

    # Optimal Grover iteration count for M=1 marked item out of N.
    iterations = max(1, round((np.pi / 4) * np.sqrt(N / len(marked))))
    print(f"Using {iterations} Grover iteration(s) on {n_qubits} qubits.")

    qc = build_grover_circuit(n_qubits, marked_index, iterations)

    sim = AerSimulator(method="statevector")
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit order: rightmost char = qubit 0. Our encoding used qubit i
    # as bit i (LSB = qubit 0), matching format(...)[::-1] used in the oracle,
    # so convert the measured bitstring (MSB-first, qubit n-1..0) back to an int.
    best_bitstring = max(counts, key=counts.get)
    measured_index = int(best_bitstring, 2)
    measured_prob = counts[best_bitstring] / shots

    print(f"Most frequent measured bitstring: {best_bitstring} -> index {measured_index}")
    print(f"Measured probability of that outcome: {measured_prob:.4f}")
    print(f"Full counts: {counts}")

    passed = (measured_index == marked_index) and (measured_prob > 0.5)

    if passed:
        print("PASS")
    else:
        print("FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
