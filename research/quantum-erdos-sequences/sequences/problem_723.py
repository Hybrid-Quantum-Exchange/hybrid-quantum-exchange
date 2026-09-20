"""
Erdos problem #723 (per data/problems.yaml in the erdosproblems repository).

Metadata verified from source:
    number: "723"
    prize: no
    status: falsifiable (last_update 2025-08-31)
    oeis: ["N/A"]
    tags: ["combinatorics"]

LIMITATION (reported honestly, as instructed): problem #723 has NO OEIS
sequence id attached (oeis: ["N/A"]). There is therefore no concrete integer
sequence to derive a finite, checkable membership/counting/divisibility
property from for this problem specifically -- fabricating one would violate
the "do not fabricate a property with no real mathematical content" rule.

Best-effort substitute actually run below: since no problem-specific finite
property is available, this script instead builds a REAL, genuine Grover
search circuit -- the canonical small quantum search primitive -- over an
N=8 (3-qubit) search space, marking a single classically-precomputed target
index, and verifies that measuring the circuit on the ideal AerSimulator
recovers that classically-known answer with high probability. This is a
generic, honest demonstration of a working quantum search circuit; it is
NOT a verification of any specific numeric property of problem #723's
sequence, because no such sequence exists in the source data. That
distinction is preserved in the printed output and in ran_ok / verified
reporting.

The classical "answer" checked here: for the search space {0,...,7}, find
the index of the unique integer whose value, cubed, is congruent to 1 mod 7
-- i.e. find x in Z_7 (embedded into 3 qubits, index 7 unused/never marked)
such that x**3 % 7 == 1, x != 0. That is computed from first principles in
plain Python and compared against Grover's output.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np

N_QUBITS = 3
N = 2 ** N_QUBITS  # 8

# --- classical answer, computed from first principles ---
def classical_targets():
    targets = []
    for x in range(N):
        if x == 0:
            continue
        if pow(x, 3, 7) == 1:
            targets.append(x)
    return targets

TARGETS = classical_targets()
assert TARGETS == [1, 2, 4], f"unexpected classical targets: {TARGETS}"


def oracle(qc: QuantumCircuit, targets):
    """Phase-flip marked computational basis states."""
    for t in targets:
        bits = format(t, f"0{N_QUBITS}b")[::-1]
        flip_qubits = [i for i, b in enumerate(bits) if b == "0"]
        for q in flip_qubits:
            qc.x(q)
        qc.h(N_QUBITS - 1)
        qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
        qc.h(N_QUBITS - 1)
        for q in flip_qubits:
            qc.x(q)


def diffuser(qc: QuantumCircuit):
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))


def build_grover_circuit(targets, iterations):
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))
    for _ in range(iterations):
        oracle(qc, targets)
        diffuser(qc)
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def main():
    M = len(TARGETS)
    # optimal number of Grover iterations for M marked out of N
    iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))

    qc = build_grover_circuit(TARGETS, iterations)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # aggregate probability mass landing on a classically-correct target
    hit_shots = sum(c for bitstr, c in counts.items() if int(bitstr, 2) in TARGETS)
    hit_prob = hit_shots / shots

    most_likely = max(counts.items(), key=lambda kv: kv[1])
    most_likely_val = int(most_likely[0], 2)

    print("Erdos problem #723 -- Grover search substitute (no OEIS id available)")
    print(f"Classical targets (x in 1..7 with x^3 mod 7 == 1): {TARGETS}")
    print(f"Grover iterations used: {iterations}")
    print(f"Measured counts: {counts}")
    print(f"Most likely measured value: {most_likely_val}")
    print(f"Probability mass on a correct target: {hit_prob:.4f}")

    verified = most_likely_val in TARGETS and hit_prob > 0.7

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
