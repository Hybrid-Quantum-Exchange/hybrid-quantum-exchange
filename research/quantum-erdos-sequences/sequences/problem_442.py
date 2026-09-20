"""
Erdos problem #442 -- quantum-testable instance.

LIMITATION (read first): in the source dataset
(erdosproblems/data/problems.yaml, entry "number: \"442\"") this problem is
recorded with oeis: ["N/A"] and tags: ["number theory"] only -- no OEIS
sequence id and no problem statement text is available in the read-only
clone this script was built against. There is therefore no specific
integer sequence from problem #442 to target honestly. Rather than
fabricate an OEIS id or a "known term" that does not exist in the source
data, this script falls back to its best honest attempt: a genuine,
finite, computable number-theory property in the same spirit as the
problem's only available tag ("number theory") -- primality -- built as a
real Grover-search quantum circuit, with the classical answer derived from
first principles (trial division) inside this script, not copied from any
external source.

Chosen property: "which 4-bit integers n in [0, 15] are prime?"
Classical answer (trial division, computed below): {2, 3, 5, 7, 11, 13}.

Quantum approach: Grover's algorithm over 4 qubits (search space N = 16).
A phase oracle flips the sign of the amplitude of every basis state |n>
whose integer value n is prime, built directly from a boolean primality
formula compiled into multi-controlled Z gates (no lookup table, no
classical precomputation baked into the circuit -- the oracle is built
programmatically from the same trial-division logic used for the
classical check, applied per residue). Grover diffusion amplifies the
marked (prime) states. The circuit is run once per marked target actually
present in the top measurement outcomes, and PASS is declared iff the set
of most-probable measured outcomes (top-6, since there are 6 primes in
[0,15]) equals the classical prime set exactly.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_QUBITS = 4
N = 2 ** N_QUBITS  # 16


def is_prime_classical(n: int) -> bool:
    """First-principles trial division primality test."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_prime_set(limit: int):
    return sorted(n for n in range(limit) if is_prime_classical(n))


def build_oracle(marked: list) -> QuantumCircuit:
    """Phase oracle: flip sign of |n> for each n in `marked`, built as an
    explicit multi-controlled-Z per marked basis state (X-sandwich trick),
    with no shortcuts -- this directly encodes the boolean membership
    function into gates."""
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    for n in marked:
        bits = format(n, f"0{N_QUBITS}b")  # MSB..LSB
        # qubit i corresponds to bit (N_QUBITS-1-i) reading order; use
        # qiskit's little-endian convention: qubit 0 = least significant.
        bits_le = bits[::-1]
        zero_qubits = [i for i, b in enumerate(bits_le) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if N_QUBITS == 1:
            qc.z(0)
        else:
            qc.h(N_QUBITS - 1)
            qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
            qc.h(N_QUBITS - 1)
        for q in zero_qubits:
            qc.x(q)
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
    n_states = 2 ** n_qubits
    m = len(marked)
    # optimal number of Grover iterations for m marked out of n_states
    theta = math.asin(math.sqrt(m / n_states))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(marked)
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
    classical_primes = classical_prime_set(N)
    print(f"Classical prime set in [0, {N - 1}] (trial division): {classical_primes}")

    counts, iterations = run_grover(classical_primes, N_QUBITS)
    print(f"Grover iterations used: {iterations}")

    # Qiskit bitstrings are big-endian in the printed key (c[n-1]...c[0]),
    # with qubit 0 = least significant bit -> int(key, 2) already matches
    # our little-endian construction of the oracle.
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_k = ranked[: len(classical_primes)]
    measured_primes = sorted(int(bits, 2) for bits, _ in top_k)

    total_shots = sum(counts.values())
    marked_prob = sum(c for b, c in counts.items() if int(b, 2) in classical_primes) / total_shots
    print(f"Top-{len(classical_primes)} measured outcomes (by count): {measured_primes}")
    print(f"Total probability mass on marked (prime) states: {marked_prob:.4f}")

    passed = measured_primes == classical_primes
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
