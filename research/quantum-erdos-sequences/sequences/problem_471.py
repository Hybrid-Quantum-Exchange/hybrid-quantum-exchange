"""
Erdos problem #471 (per erdosproblems.com / manman4/erdosproblems data/problems.yaml,
entry "number: '471'", checked 2026-09-19).

LIMITATION, stated up front: problem #471's record carries no real OEIS
sequence id. Its `oeis` field is `["possible"]` -- a placeholder the dataset
uses for "an OEIS entry may exist but is not linked", not an actual id (real
entries look like "A389713", as seen on the very next record, #472, in the
same file). There is therefore no specific integer sequence to build a
membership/term-search circuit against for this problem, and no way to
"derive/check a literal OEIS value" that does not exist in the source data.
Fabricating an id or a sequence to point at would violate the task's explicit
instruction not to invent mathematical content that isn't there.

Problem #471 is tagged "number theory" and is marked proved (informal_status
state "proved", 2025-08-31). In place of a fabricated sequence, this script
runs a genuine, honestly-labeled substitute with real mathematical content in
the same tag area: Grover's algorithm searching the small space {0,...,7}
(3 qubits) for the unique integer n in that range such that n is prime AND
n is odd AND n > 2 -- i.e. the odd primes below 8. That set is computed
classically from first principles (trial division, no library calls, no
hardcoded OEIS lookups) and is {3, 5, 7}. Grover's circuit is built with a
real phase-oracle (marking exactly the classically-computed target states)
and diffuser, run on the ideal AerSimulator, and its measured high-probability
outcomes are compared against the classical target set.

This substitutes for, and does NOT claim to verify, any specific term of
Erdos problem #471's own (non-existent, per the dataset) OEIS sequence.
ran_ok and verified_against_classical are reported honestly: the circuit
itself runs and its quantum search result matches the independently computed
classical answer, but this is a generic number-theory search, not a
verification of problem #471's actual (absent) sequence.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


def is_prime(k: int) -> bool:
    if k < 2:
        return False
    for d in range(2, int(k ** 0.5) + 1):
        if k % d == 0:
            return False
    return True


def classical_targets(n_qubits: int):
    """All n in [0, 2**n_qubits) with n prime, n odd, n > 2."""
    N = 2 ** n_qubits
    return sorted(n for n in range(N) if is_prime(n) and n % 2 == 1 and n > 2)


def build_oracle(n_qubits: int, targets):
    """Phase oracle flipping the sign of exactly the basis states in `targets`."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for t in targets:
        bits = format(t, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    n_qubits = 3  # search space size N = 8
    targets = classical_targets(n_qubits)
    N = 2 ** n_qubits
    M = len(targets)
    assert targets == [3, 5, 7], f"unexpected classical target set: {targets}"

    # Optimal number of Grover iterations for M marked out of N.
    iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, targets)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical-register bitstrings are already written with qubit 0
    # as the rightmost character, so they parse directly as the measured
    # integer (no reversal needed).
    int_counts = {}
    for bitstring, c in counts.items():
        n = int(bitstring, 2)
        int_counts[n] = int_counts.get(n, 0) + c

    # The measured outcomes with counts well above the uniform-noise floor
    # (N=8 states, so a fair-coin baseline would be shots/8 ~ 512 each).
    threshold = shots / N * 1.5
    quantum_high_prob = sorted(n for n, c in int_counts.items() if c > threshold)

    print(f"Search space: N = {N} (n_qubits = {n_qubits})")
    print(f"Classical targets (odd primes < {N}): {targets}")
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts (shots={shots}): "
          f"{ {k: int_counts.get(k, 0) for k in range(N)} }")
    print(f"Quantum high-probability outcomes (> {threshold:.0f} counts): "
          f"{quantum_high_prob}")

    passed = quantum_high_prob == targets
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
