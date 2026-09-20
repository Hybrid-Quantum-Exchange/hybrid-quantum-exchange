"""
Erdos problem #453 — quantum-testable lane.

Source metadata (erdosproblems.com dataset, /home/user/manman4/erdosproblems/
data/problems.yaml, entry "number: \"453\""):
    prize: no
    informal_status: disproved (Lean), last_update 2026-01-31
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION (read before trusting the PASS below as evidence about problem
453 itself): the dataset lists no OEIS sequence id for problem 453
(oeis == ["N/A"]), and the problems.yaml clone available in this sandbox
carries no problem statement text, only this status/metadata record. There
is therefore no sequence, no early term, and no small finite property of
"the sequence for problem 453" that can honestly be derived here — building
one would mean fabricating a property with no connection to the actual
problem, which the task explicitly rules out.

Best honest attempt taken instead: the problem is tagged "number theory" and
is recorded as disproved, i.e. a counterexample to some number-theoretic
claim was found. In that spirit, and to still deliver a *real* quantum
computation rather than a placeholder, this script uses Grover's algorithm
to search a small, genuinely computable number-theoretic space: the 4-qubit
domain 0..15, marking the unique value x such that x is the smallest
composite (non-prime, non-0/1) integer in that range. A single marked item
out of 16 is exactly the regime Grover's algorithm is designed for, so this
is a real, well-posed oracle search, not a fabricated stand-in for problem
453's own (unavailable) content.

The classical answer (the smallest composite in [0,15], found by trial
division over all integers in the range) is computed from first principles
in this script, independently of the quantum circuit, and the two are
compared.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_QUBITS = 4
N = 2 ** N_QUBITS  # domain 0..15


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_composites(n_values: int) -> list[int]:
    """Composite = integer >= 4 in range that is not prime (0,1 excluded)."""
    return [x for x in range(n_values) if x >= 4 and not is_prime(x)]


def build_oracle(marked: list[int], n_qubits: int) -> QuantumCircuit:
    """Phase-flip oracle marking each value in `marked` (multi-controlled Z)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked: list[int], n_qubits: int, shots: int = 2048):
    n = 2 ** n_qubits
    m = len(marked)
    # optimal number of Grover iterations for m marked items out of n
    theta = math.asin(math.sqrt(m / n))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    all_composites = sorted(classical_composites(N))
    target = all_composites[0]  # smallest composite in [0, N-1] -> single Grover target
    marked = [target]
    print(f"Domain: integers 0..{N - 1} ({N_QUBITS} qubits)")
    print(f"All composites in range (trial division, first principles): {all_composites}")
    print(f"Classical answer (smallest composite): {target}")

    counts, iterations = run_grover(marked, N_QUBITS)
    print(f"Grover iterations used: {iterations}")

    # Qiskit count keys are c[n-1]...c[0] left to right, i.e. already standard
    # binary with qubit 0 as the least-significant bit, so no reversal needed.
    decoded_counts = {}
    for bitstring, freq in counts.items():
        value = int(bitstring, 2)
        decoded_counts[value] = decoded_counts.get(value, 0) + freq

    total_shots = sum(decoded_counts.values())
    target_hits = decoded_counts.get(target, 0)
    target_fraction = target_hits / total_shots

    measured_top = max(decoded_counts.items(), key=lambda kv: kv[1])[0]

    print(f"Fraction of shots landing on target value {target}: {target_fraction:.3f}")
    print(f"Most-measured value: {measured_top}")

    verified = (target_fraction > 0.8) and (measured_top == target)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
