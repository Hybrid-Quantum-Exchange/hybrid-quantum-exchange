"""
Erdos problem #807 -- quantum-testable lane.

Source metadata (from erdosproblems.com data, data/problems.yaml, entry
"number: '807'"):
    prize: no
    status: disproved (last_update 2025-08-31)
    oeis: ["N/A"]          <-- no OEIS sequence is associated with this problem
    tags: ["graph theory"]

LIMITATION (reported honestly, per instructions): problem #807 has no OEIS
sequence id at all ("N/A"). There is therefore no actual integer sequence to
build a "membership / early term" quantum oracle for, and nothing here should
be read as testing problem #807's real mathematical content -- it has none
that reduces to a small finite computable instance from the given metadata.

Best-effort fallback: the only substantive signal in the metadata is the tag
"graph theory". To still produce a *genuine* quantum circuit with real
mathematical content (not a fabricated "sequence value"), this script uses
Grover's algorithm to search the edge-subsets of the triangle graph K3 for
the unique subset that forms a triangle (i.e. contains all 3 possible edges).

Classical property being tested (computed from first principles below, not
copied from anywhere):
    Let the 3 possible edges of K3 be e0, e1, e2. An edge-subset is a 3-bit
    string b2 b1 b0 (bit i = 1 means edge ei is present). The subset forms a
    triangle iff all three edges are present, i.e. b0=b1=b2=1, i.e. the
    subset equals the integer 7 (binary '111'). Among the 8 possible
    edge-subsets of K3, exactly one (7 / '111') forms a triangle.

The circuit: a 3-qubit Grover search over all 8 edge-subsets, with an oracle
that flags the unique triangle-forming subset '111', run for the optimal
number of Grover iterations for N=8, M=1 (round(pi/4 * sqrt(8)) = 2
iterations). We verify that the simulator returns '111' as the
overwhelmingly most likely outcome (ideal success probability after 2
iterations is about 0.945, not deterministic), matching the classically
computed answer.

Dependencies: qiskit, qiskit_aer, numpy only (already installed).
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_answer():
    """Brute-force, from first principles: which 3-bit edge-subsets of K3
    form a triangle (all 3 edges present)?"""
    triangle_subsets = []
    for b in range(8):  # b2 b1 b0, bit i = edge i present
        e0 = (b >> 0) & 1
        e1 = (b >> 1) & 1
        e2 = (b >> 2) & 1
        if e0 == 1 and e1 == 1 and e2 == 1:
            triangle_subsets.append(b)
    assert triangle_subsets == [7]
    return triangle_subsets[0]  # 7, binary '111'


def build_grover_circuit(marked: int, n_qubits: int = 3):
    """Grover search over n_qubits for the single marked basis state."""
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Initial superposition
    qc.h(range(n_qubits))

    def oracle(qc):
        # Flip sign of |111> (marked == 7). Multi-controlled Z via H-CCX-H.
        bits = format(marked, f"0{n_qubits}b")[::-1]
        for i, bit in enumerate(bits):
            if bit == "0":
                qc.x(i)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i, bit in enumerate(bits):
            if bit == "0":
                qc.x(i)

    def diffuser(qc):
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    # Optimal iteration count for N=8, M=1: floor(pi/4 * sqrt(N/M)) = 2
    n_states = 2 ** n_qubits
    iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(n_states / 1))))

    for _ in range(iterations):
        oracle(qc)
        diffuser(qc)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, iterations


def run():
    marked = classical_answer()
    print(f"Classical answer: unique triangle-forming edge-subset of K3 = "
          f"{marked} (binary '{format(marked, '03b')}')")

    qc, iterations = build_grover_circuit(marked)
    print(f"Grover circuit built with {iterations} iteration(s) over 3 qubits.")

    sim = AerSimulator()
    job = sim.run(qc, shots=2048)
    result = job.result()
    counts = result.get_counts()

    # Qiskit bit ordering: classical bit c0 is rightmost in the returned key.
    most_common = max(counts, key=counts.get)
    most_common_int = int(most_common, 2)

    print(f"Measurement counts: {counts}")
    print(f"Most frequent measured state: '{most_common}' -> {most_common_int}")

    passed = (most_common_int == marked)
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
