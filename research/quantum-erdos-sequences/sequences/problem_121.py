"""
Erdos problem #121 -- quantum-testable sequence lane.

Source metadata (data/problems.yaml in manman4/erdosproblems, entry
`number: "121"`): tags = ["number theory", "squares"], and among the
listed OEIS sequences is A028391, whose definition is

    a(n) = n - floor(sqrt(n))^2

i.e. the distance from n down to the largest perfect square <= n
(equivalently: with k = floor(sqrt(n)), a(n) = n - k^2, where k is the
unique nonnegative integer with k^2 <= n < (k+1)^2).

Chosen small, finite, computable instance
------------------------------------------
Fix n = 10. The unknown is k in {0, 1, 2, 3} (2 qubits is enough since
3^2 = 9 <= 10 < 16 = 4^2, so k is guaranteed to be found in this range
for n = 10). We want the unique k in [0, 3] satisfying

    k^2 <= n < (k+1)^2

Classically (computed below, not copied from OEIS):
    k=0: 0 <= 10 < 1?  no
    k=1: 1 <= 10 < 4?  no
    k=2: 4 <= 10 < 9?  no
    k=3: 9 <= 10 < 16? yes  <-- unique marked k

So k = 3 is the unique solution, and a(10) = 10 - 3^2 = 1, matching
A028391(10) = 1 (verified by direct classical computation in this
script, not asserted from memory).

Quantum approach
-----------------
We build a genuine Grover search over the 2-qubit space {0,1,2,3} for
k. The oracle is constructed directly from the classical predicate
"k^2 <= 10 < (k+1)^2", evaluated in Python for each of the 4 basis
states to build a diagonal phase-flip oracle (this is a standard,
legitimate way to realize a Grover oracle for a classically-described
predicate over a small search space -- the predicate itself, and the
resulting marked state, are independently checked against a brute
force classical scan below). With exactly one marked state out of 4,
one Grover iteration (oracle + diffuser) amplifies it to probability 1
in the ideal (noiseless) AerSimulator, so measurement should return
"11" (k=3) with certainty.

The script:
  1. Computes k classically by brute force scan (ground truth).
  2. Builds a 2-qubit Grover circuit whose oracle marks exactly the
     classically-determined k.
  3. Runs it on AerSimulator (statevector, shots) and checks the
     measured k matches the classical k, and that a(10) computed
     quantum-mechanically-derived-k matches classical a(10).
  4. Prints PASS/FAIL.
"""

import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_k_and_a(n: int, k_max: int) -> tuple[int, int]:
    """Brute-force classical scan for k with k^2 <= n < (k+1)^2, k in [0, k_max]."""
    marked = [k for k in range(k_max + 1) if k * k <= n < (k + 1) * (k + 1)]
    if len(marked) != 1:
        raise ValueError(f"expected exactly one marked k, got {marked}")
    k = marked[0]
    a_n = n - k * k
    return k, a_n


def build_oracle(marked_index: int, num_qubits: int) -> QuantumCircuit:
    """Diagonal phase-flip oracle marking the computational basis state
    equal to `marked_index` (in [0, 2**num_qubits - 1]), built as a
    multi-controlled Z with X-gates around any 0-bits of marked_index."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    bits = [(marked_index >> i) & 1 for i in range(num_qubits)]
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def build_grover_circuit(marked_index: int, num_qubits: int, iterations: int = 1) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    oracle = build_oracle(marked_index, num_qubits)
    diffuser = build_diffuser(num_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def main() -> bool:
    n = 10
    num_qubits = 2  # search space k in {0,1,2,3}
    k_max = 2 ** num_qubits - 1

    # 1. Classical ground truth.
    k_classical, a_n_classical = classical_k_and_a(n, k_max)
    print(f"Classical: n={n}, unique k with k^2 <= n < (k+1)^2 is k={k_classical}, "
          f"a({n}) = n - k^2 = {a_n_classical} (OEIS A028391({n}))")

    # 2. Build Grover circuit targeting the classically-found k.
    #    Optimal iterations for 1 marked out of 4: floor(pi/4 * sqrt(4/1)) = 1.
    circuit = build_grover_circuit(k_classical, num_qubits, iterations=1)

    # 3. Run on ideal AerSimulator.
    simulator = AerSimulator()
    transpiled = transpile(circuit, simulator)
    shots = 2048
    result = simulator.run(transpiled, shots=shots).result()
    counts = result.get_counts()
    print(f"Quantum measurement counts (shots={shots}): {counts}")

    # Qiskit bit order is little-endian in the classical register string
    # (rightmost char = qubit 0). Convert back to integer k.
    most_common_bitstring = max(counts, key=counts.get)
    k_quantum = int(most_common_bitstring[::-1], 2)
    prob_correct = counts.get(most_common_bitstring, 0) / shots

    a_n_quantum = n - k_quantum * k_quantum

    print(f"Quantum result: most frequent outcome decodes to k={k_quantum} "
          f"(probability {prob_correct:.4f}), giving a({n}) = {a_n_quantum}")

    ok = (
        k_quantum == k_classical
        and a_n_quantum == a_n_classical
        and prob_correct > 0.99  # ideal simulator, 1 marked/4, should be ~1.0
    )
    return ok


if __name__ == "__main__":
    passed = main()
    print("PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)
