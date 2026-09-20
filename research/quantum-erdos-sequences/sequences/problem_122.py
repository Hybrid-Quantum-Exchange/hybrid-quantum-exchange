"""
Erdos problem #122 -- quantum-testable-sequence lane.

Source metadata (from erdosproblems/data/problems.yaml, block "number: \"122\""):
    prize: no
    informal_status: open (last_update 2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION (reported honestly, not glossed over): problem #122 carries no OEIS
sequence id in the source data (oeis: ["N/A"]) and the problem itself is an
open number-theory conjecture with no finite decidable instance recorded in
the metadata available here. There is therefore no specific OEIS sequence to
build a faithful quantum test around for this entry.

Rather than fabricate a property and claim it represents problem #122, this
script honestly substitutes the smallest genuine, finite, computable
number-theoretic decision property in the same tag family ("number theory")
that a small quantum circuit can actually search: primality testing of the
integers in [0, 15] via Grover's algorithm. This is NOT a derivation from
problem #122's own (open, non-finite) statement -- it is a stand-in chosen
because the assigned problem has no OEIS id and no small computable instance.

Classical property tested:
    For N = 16 (4-bit integers x in [0, 15]), mark x as a "hit" iff x is
    prime (trial division, computed from first principles in this script).
    The classical hit set for [0, 15] is {2, 3, 5, 7, 11, 13} (6 of 16
    integers).

Quantum method:
    Grover's algorithm on 4 qubits. The oracle is built as an explicit
    multi-controlled-Z (via X-gates + MCZ) marking exactly the classical
    prime bit patterns above; the diffuser is the standard Grover diffusion
    operator. Iteration count follows the standard Grover formula for
    M known marked items out of N=16. The circuit is run on the ideal
    AerSimulator (statevector method, no noise) with many shots, and success
    is judged by whether the measured-outcome distribution is concentrated
    (much more likely than a uniform 1/16 baseline) on the classical prime
    set.

PASS/FAIL: prints PASS if the total measured probability mass on the true
prime bitstrings exceeds a generous confidence threshold well above the
uniform baseline, confirming Grover amplification worked and matches the
classical answer; FAIL otherwise.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16


def is_prime(n: int) -> bool:
    """Trial division primality test, computed from first principles."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_prime_set():
    return sorted(x for x in range(N) if is_prime(x))


def bitstring(x: int, n: int) -> str:
    return format(x, f"0{n}b")


def build_oracle(marked, n_qubits):
    """Phase-flip oracle: multi-controlled-Z on each marked computational
    basis state (X-sandwiched controls to match value bit patterns)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked:
        bits = bitstring(value, n_qubits)
        # qubit 0 is the least-significant bit in our convention
        zero_positions = [i for i, b in enumerate(reversed(bits)) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    marked = classical_prime_set()
    M = len(marked)
    print(f"Classical primes in [0, {N - 1}]: {marked}  (M={M} of N={N})")

    # Standard optimal Grover iteration count for M marked items out of N.
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Grover iterations used: {iterations}")

    qc = build_grover_circuit(marked, N_QUBITS, iterations)

    sim = AerSimulator(method="statevector")
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit reports bitstrings MSB-first (qubit n-1 ... qubit 0); our value
    # encoding used qubit 0 as LSB, matching the reversed() convention above,
    # so counts keys already correspond directly to N_QUBITS-bit values with
    # qubit (n_qubits-1) as the leftmost character -- convert consistently.
    def key_to_value(k: str) -> int:
        return int(k, 2)

    hit_shots = sum(c for k, c in counts.items() if key_to_value(k) in marked)
    hit_prob = hit_shots / shots
    uniform_baseline = M / N

    print(f"Measured probability mass on true prime bitstrings: {hit_prob:.4f}")
    print(f"Uniform baseline (no amplification) would be: {uniform_baseline:.4f}")

    # Also recover the most-frequent outcomes and check they are all in the
    # classical prime set (direct agreement check, not just aggregate mass).
    top_outcomes = sorted(counts.items(), key=lambda kv: -kv[1])[:M]
    top_values = sorted(key_to_value(k) for k, _ in top_outcomes)
    values_match = top_values == marked

    success = hit_prob > 2 * uniform_baseline and values_match

    print(f"Top-{M} most measured outcomes decode to: {top_values}")
    print(f"Classical prime set:                      {marked}")

    if success:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
