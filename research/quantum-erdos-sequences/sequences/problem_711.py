"""
Erdos problem #711 -- quantum-testable sequence lane.

Source record: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"- number: \"711\"" (prize INR 1000, tags: ["number theory"], informal
status "open" as of 2025-08-31).

LIMITATION (read this before trusting the "PASS" below as evidence about
problem #711 itself): the YAML record for #711 does not carry a concrete
OEIS sequence id. Its `oeis` field is the literal placeholder string
"possible" -- meaning "an OEIS entry may exist" -- not an actual A-number,
and the problem statement itself is not present in this metadata file (only
prize/status/tags). There is therefore no citable, checkable sequence
definition to build a genuine problem-711-specific oracle from without
fabricating one, which the task brief explicitly forbids ("do not fabricate
a property with no real mathematical content").

Honest fallback actually implemented: problem #711 is tagged "number
theory", so this script builds a REAL, self-contained number-theoretic
decision problem in that same family -- primality -- and verifies it with a
genuine Grover search circuit run on AerSimulator. This is NOT a derivation
of problem #711's actual content; it is the best-effort, honestly-labeled
substitute the brief calls for when no real per-problem sequence is
available. Do not read the PASS below as validating anything about Erdos
problem #711 specifically.

Property tested: for N = 16 (4 qubits, search space {0, ..., 15}), find the
set of prime integers via a classical sieve computed from first principles
in this script, then build a Grover oracle that phase-flips exactly the
computational basis states corresponding to primes in that range, and run
enough Grover iterations to amplify those states so that the most frequent
simulator measurement outcomes match the classical prime set.

Classical answer for N = 16, computed here (not copied from OEIS):
primes in [0, 15] = {2, 3, 5, 7, 11, 13}  (this matches OEIS A000040 by
construction of the classical sieve below, but the sieve computes it itself
rather than looking A000040 up).
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator


def classical_primes(n):
    """Sieve of Eratosthenes over [0, n-1], computed from first principles."""
    is_prime = [False, False] + [True] * (n - 2)
    for p in range(2, int(n ** 0.5) + 1):
        if is_prime[p]:
            for multiple in range(p * p, n, p):
                is_prime[multiple] = False
    return sorted(i for i, prime in enumerate(is_prime) if prime)


def build_oracle(num_qubits, marked_values):
    """Phase-flip exactly the basis states in `marked_values`."""
    oracle = QuantumCircuit(num_qubits, name="Oracle")
    for value in marked_values:
        bits = format(value, f"0{num_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            oracle.x(zero_positions)
        if num_qubits == 1:
            oracle.z(0)
        else:
            oracle.append(MCMTGate(ZGate(), num_qubits - 1, 1), list(range(num_qubits)))
        if zero_positions:
            oracle.x(zero_positions)
    return oracle


def build_diffuser(num_qubits):
    """Standard Grover diffuser (inversion about the mean)."""
    diffuser = QuantumCircuit(num_qubits, name="Diffuser")
    diffuser.h(range(num_qubits))
    diffuser.x(range(num_qubits))
    if num_qubits == 1:
        diffuser.z(0)
    else:
        diffuser.append(MCMTGate(ZGate(), num_qubits - 1, 1), list(range(num_qubits)))
    diffuser.x(range(num_qubits))
    diffuser.h(range(num_qubits))
    return diffuser


def run_grover(num_qubits, marked_values, shots=4096):
    n_total = 2 ** num_qubits
    oracle = build_oracle(num_qubits, marked_values)
    diffuser = build_diffuser(num_qubits)

    # Optimal number of Grover iterations for this M-out-of-N search.
    m = len(marked_values)
    theta = np.arcsin(np.sqrt(m / n_total))
    iterations = max(1, int(np.floor((np.pi / (4 * theta)) - 0.5)))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(num_qubits))
        qc.append(diffuser.to_gate(), range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))

    qc = qc.decompose().decompose()

    backend = AerSimulator()
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical-bit strings are already MSB-first (c_{n-1}...c_0),
    # i.e. int(bitstring, 2) gives the same integer value as the circuit's
    # little-endian qubit-index convention used in build_oracle.
    int_counts = {}
    for bitstring, freq in counts.items():
        value = int(bitstring, 2)
        int_counts[value] = int_counts.get(value, 0) + freq
    return int_counts, iterations


def main():
    num_qubits = 4
    n = 2 ** num_qubits  # 16

    classical_answer = classical_primes(n)
    print(f"Classical primes in [0, {n - 1}] (computed by sieve): {classical_answer}")

    counts, iterations = run_grover(num_qubits, classical_answer)
    total_shots = sum(counts.values())

    # Quantum answer: the states measured more often than the uniform-random
    # baseline (1 / n of shots) are the ones Grover amplified, i.e. the
    # circuit's answer to "which basis states are marked".
    baseline = total_shots / n
    quantum_answer = sorted(v for v, c in counts.items() if c > 2 * baseline)

    print(f"Grover iterations used: {iterations}")
    print(f"Quantum-amplified states (from {total_shots} shots): {quantum_answer}")

    marked_shots = sum(c for v, c in counts.items() if v in classical_answer)
    hit_rate = marked_shots / total_shots
    print(f"Fraction of shots landing on a true prime state: {hit_rate:.3f}")

    passed = (
        quantum_answer == classical_answer
        and hit_rate > 0.8  # Grover should concentrate most probability on marked states
    )

    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
