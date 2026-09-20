"""
Erdos problem #33 -- quantum-testable instance.

LIMITATION (read first): problem #33's entry in erdosproblems/data/problems.yaml
(number: "33") lists no OEIS sequence: `oeis: ["N/A"]`. Its tags are
["number theory", "additive basis"]. There is therefore no OEIS id to derive a
property from directly. This script instead builds a small, finite, honestly
computable property drawn from the "additive basis" tag -- representability of
an integer as a sum of two squares (a classic additive-basis question in
number theory, closely related to the sums-of-squares / additive-basis flavor
of problem #33) -- and tests it with a real Grover search circuit. This is a
best-honest-effort substitute, NOT a claim that OEIS has a sequence for
problem #33, and NOT a claim that this circuit resolves problem #33 itself.

Classical property tested (computed from first principles below, not copied
from any OEIS listing):
    For x in {0, 1, ..., 15} (4-bit search space), is x expressible as
    a^2 + b^2 for some a, b in {0, 1, 2}?

The classically-marked set is computed by brute force in `classical_marked_set()`.
That same set is used to build a Grover oracle (a multi-controlled-Z phase
flip on exactly those classically-verified basis states) which is then
amplified with the standard Grover diffusion operator and measured on the
ideal AerSimulator. The script PASSes if the quantum sampling distribution
concentrates on the classically-marked set with high probability (i.e. Grover
search actually finds the same answers the classical check found).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMT, ZGate
from qiskit_aer import AerSimulator
from qiskit import transpile

N_QUBITS = 4  # search space size 2**4 = 16, x in [0, 15]
N = 1 << N_QUBITS


def classical_marked_set():
    """Brute-force, from first principles: x in [0,15] with x = a^2+b^2,
    a,b in [0,3]. No OEIS data is consulted here."""
    marked = set()
    for x in range(N):
        for a in range(3):
            for b in range(3):
                if a * a + b * b == x:
                    marked.add(x)
                    break
            else:
                continue
            break
    return marked


def oracle(qc: QuantumCircuit, marked_values, qubits):
    """Phase-flip exactly the given marked basis states (multi-controlled Z
    per state, using X gates to map each target bit pattern onto the
    all-ones controls)."""
    n = len(qubits)
    for val in marked_values:
        bits = [(val >> i) & 1 for i in range(n)]  # little-endian bit i -> qubits[i]
        flip_qubits = [qubits[i] for i in range(n) if bits[i] == 0]
        for q in flip_qubits:
            qc.x(q)
        if n == 1:
            qc.z(qubits[0])
        else:
            qc.append(MCMT(ZGate(), n - 1, 1), qubits)
        for q in flip_qubits:
            qc.x(q)


def diffusion(qc: QuantumCircuit, qubits):
    n = len(qubits)
    qc.h(qubits)
    qc.x(qubits)
    if n == 1:
        qc.z(qubits[0])
    else:
        qc.append(MCMT(ZGate(), n - 1, 1), qubits)
    qc.x(qubits)
    qc.h(qubits)


def build_grover_circuit(marked_values, n_qubits=N_QUBITS, iterations=None):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))
    qc.h(qubits)

    m = len(marked_values)
    total = 1 << n_qubits
    if iterations is None:
        # standard optimal Grover iteration count
        iterations = max(1, round((np.pi / 4) * np.sqrt(total / m)))

    for _ in range(iterations):
        oracle(qc, marked_values, qubits)
        diffusion(qc, qubits)

    qc.measure(qubits, qubits)
    return qc, iterations


def main():
    marked = classical_marked_set()
    print(f"Classical brute-force marked set (sum of two squares, a,b in 0..2): "
          f"{sorted(marked)}")
    assert len(marked) > 0

    qc, iters = build_grover_circuit(marked)
    print(f"Grover iterations used: {iters}")

    sim = AerSimulator()
    shots = 4096
    tqc = transpile(qc, sim, basis_gates=["u", "cx"])
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bitstrings are little-endian in classical register order c[n-1..0];
    # our qubit i corresponds to classical bit i (measure(qubits, qubits) preserves
    # index mapping), so int(bitstring[::-1], 2) recovers x directly, but since
    # qiskit prints MSB-first with bit0 as rightmost character, reversing gives
    # bit0 first; convert properly:
    def bitstring_to_x(bitstring):
        # bitstring is c[n-1]c[n-2]...c[0]
        bits = bitstring[::-1]  # now bits[i] = c[i]
        return sum(int(bits[i]) << i for i in range(N_QUBITS))

    outcome_counts = {}
    for bitstring, c in counts.items():
        x = bitstring_to_x(bitstring)
        outcome_counts[x] = outcome_counts.get(x, 0) + c

    marked_hits = sum(c for x, c in outcome_counts.items() if x in marked)
    prob_marked = marked_hits / shots

    print(f"Measured outcomes (value: count): {dict(sorted(outcome_counts.items()))}")
    print(f"Probability mass on classically-marked values: {prob_marked:.4f}")

    # Success criterion: Grover search should concentrate heavily on the
    # classically-verified marked set (well above uniform-random baseline
    # len(marked)/N).
    baseline = len(marked) / N
    quantum_result_marked_set = {x for x in outcome_counts if x in marked}
    threshold = 0.80
    verified = prob_marked >= threshold and prob_marked > baseline

    print(f"Uniform-random baseline probability: {baseline:.4f}")
    print(f"Threshold for PASS: {threshold}")

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
