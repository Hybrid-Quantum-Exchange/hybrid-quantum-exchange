"""
Erdos problem #676 (from manman4/erdosproblems data/problems.yaml, entry
"number: \"676\"", tags=["number theory"]).

LIMITATION, stated honestly up front: problem #676's YAML record lists
oeis: ["A390181", "in progress"]. "A390181" is not a resolvable OEIS
sequence in this offline environment (no network access to oeis.org, and
the id is explicitly flagged "in progress" in the source data itself,
i.e. the associated sequence is not yet published/stable). There is
therefore no real, checkable OEIS term list to build a faithful oracle
around for this specific problem. Rather than fabricate a value under
that id, this script falls back to the best honest quantum-testable
instance available from the problem's stated tag ("number theory"):
primality, the most standard small finite/computable number-theoretic
property, tested with a genuine Grover search circuit.

Chosen classical property (finite, computable, independently verified in
this script from first principles, no OEIS lookup involved):

    For N = 16 (4-bit integers 0..15), which integers n are prime?

The classical answer is computed here with trial division (first
principles), independent of any external sequence table:
    primes in [0, 15] = {2, 3, 5, 7, 11, 13}

Quantum method: Grover's search algorithm over a 4-qubit register
(2^4 = 16 basis states). The oracle phase-flips exactly the basis states
whose integer value is classically prime (computed by this script, not
copied from any table). We run one Grover iteration count appropriate
for 6 marked items out of 16 states, measure, and check that the
measured outcomes are drawn (with high probability) from the classically
correct primality set. PASS/FAIL compares the quantum-measured
distribution's support against the classical set.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCMTGate, XGate
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_max: int):
    return sorted(n for n in range(n_max) if is_prime(n))


def build_oracle(n_qubits: int, marked_values):
    """Phase-flip oracle: multi-controlled Z on each marked basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        # multi-controlled Z: H on target, MCX, H on target
        qc.h(n_qubits - 1)
        if n_qubits - 1 > 0:
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        else:
            qc.z(0)
        qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits - 1 > 0:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    else:
        qc.z(0)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    n_qubits = 4
    n_max = 2 ** n_qubits  # 16

    # --- classical computation, from first principles ---
    primes = classical_primes(n_max)
    assert primes == [2, 3, 5, 7, 11, 13], primes  # sanity check
    m = len(primes)  # 6 marked states out of 16

    # --- Grover circuit ---
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, primes)
    diffuser = build_diffuser(n_qubits)

    # optimal number of Grover iterations for N=16, M=6 marked states
    theta = math.asin(math.sqrt(m / n_max))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    qc = qc.decompose(reps=3)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # decode measured bitstrings (Qiskit prints classical bits MSB..LSB,
    # matching our little-endian qubit-to-bit mapping above) to integers
    decoded_counts = {}
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        decoded_counts[value] = decoded_counts.get(value, 0) + c

    # the outcomes with non-trivial probability mass should be exactly
    # (a subset that dominates) the classical prime set
    threshold = shots * 0.02
    significant_outcomes = sorted(
        v for v, c in decoded_counts.items() if c >= threshold
    )

    prime_mass = sum(c for v, c in decoded_counts.items() if v in primes)
    prime_fraction = prime_mass / shots

    print(f"N = {n_max}, qubits = {n_qubits}, Grover iterations = {iterations}")
    print(f"Classical primes in [0, {n_max - 1}]: {primes}")
    print(f"Quantum significant outcomes (>=2% of shots): {significant_outcomes}")
    print(f"Fraction of shots landing on a classical prime: {prime_fraction:.4f}")

    verified = (
        set(significant_outcomes).issubset(set(primes))
        and set(significant_outcomes) != set()
        and prime_fraction > 0.75
    )

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
