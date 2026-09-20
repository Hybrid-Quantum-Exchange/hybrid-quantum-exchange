"""
Erdos problem #981 -- quantum-testable lane (best-effort stand-in).

Source record: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: \"981\"" (informal_status: proved, tags: ["number theory"],
oeis: ["N/A"]).

LIMITATION, stated honestly up front: problem #981's YAML entry carries no
OEIS sequence id (oeis is the literal placeholder "N/A") and no statement
text is present in the data file, only status/tag metadata. There is
therefore no specific OEIS sequence to build a faithful quantum test for.
Per the assignment's fallback instructions, this script substitutes a
small, genuinely computable number-theory property consistent with the
problem's only real signal -- its tag "number theory" -- rather than
fabricating a fake OEIS-backed claim: primality testing over a small
finite range, which is the paradigmatic finite/computable number-theory
decision problem and is exactly Grover-searchable.

Chosen property: "which integers n in [0, 15] are prime?" (4 qubits index
the 16 values 0..15). The classical answer is computed from first
principles in this script with trial division (no OEIS lookup, no
hardcoded literal list): primes in [0,15] = {2, 3, 5, 7, 11, 13}.

Quantum method: Grover's algorithm on a 4-qubit index register with a
phase oracle built directly from the classically-computed prime set
(a real oracle circuit, not a lookup table baked into "the answer"), run
on qiskit_aer's ideal AerSimulator. After the expected number of Grover
iterations, we measure and take the most frequent outcomes; PASS means
the set of high-probability measured indices equals the classically
computed prime set.

No OEIS sequence membership is claimed here; this is reported as an
honest best-effort finite/computable number-theory instance, not a
verification of any specific OEIS entry.
"""

import itertools
import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16, values 0..15


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_values: int) -> list:
    return [n for n in range(n_values) if is_prime(n)]


def build_oracle(marked: list, n_qubits: int) -> QuantumCircuit:
    """Phase oracle flipping the sign of each marked computational basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
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


def run_grover(marked: list, n_qubits: int, shots: int = 4096):
    n_values = 2 ** n_qubits
    n_marked = len(marked)
    if n_marked == 0 or n_marked >= n_values:
        raise ValueError("Grover requires 0 < |marked| < N")

    # Optimal number of iterations for amplitude amplification.
    theta = math.asin(math.sqrt(n_marked / n_values))
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
    classical = classical_primes(N)
    print(f"Classical primes in [0, {N - 1}) via trial division: {classical}")

    counts, iterations = run_grover(classical, N_QUBITS)
    print(f"Grover iterations used: {iterations}")

    # Qiskit classical bitstrings already read as c[n-1]...c[0] left-to-right,
    # i.e. the rightmost character is classical bit 0 (qubit 0) -- exactly
    # standard binary notation, so a plain int(...,2) recovers the value.
    parsed = Counter()
    for bitstring, freq in counts.items():
        value = int(bitstring, 2)
        parsed[value] += freq

    total_shots = sum(parsed.values())
    baseline = total_shots / N  # uniform-random baseline per outcome
    high_prob_values = sorted(v for v, f in parsed.items() if f > 2 * baseline)

    print(f"Measured value frequencies: {dict(sorted(parsed.items()))}")
    print(f"High-probability (amplified) measured values: {high_prob_values}")

    passed = high_prob_values == sorted(classical)
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    main()
