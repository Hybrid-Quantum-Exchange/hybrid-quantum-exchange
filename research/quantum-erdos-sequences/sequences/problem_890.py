"""
Erdos problem #890 -- quantum-testable sequence lane.

Source metadata (data/problems.yaml, manman4/erdosproblems, entry `number: "890"`):
    prize: no
    status: open (last_update 2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory", "primes"]

LIMITATION (read this before trusting the "OEIS id(s) used" field): problem
#890 carries no OEIS sequence id in the source data -- the field is the
literal string "N/A". There is therefore no specific OEIS-indexed sequence
for this problem to build a faithful quantum test around, and nothing here
should be read as verifying problem #890's actual open conjecture (which
this script does not even state, since the yaml entry carries no
description field either).

Rather than fabricate a fake OEIS-backed property, this script honestly
substitutes the nearest well-defined, finite, computable property implied by
problem #890's own tags ("number theory", "primes"): primality testing over
a small finite universe. This is a real mathematical property with a real
quantum algorithm behind it (Grover search), not a copied OEIS value -- it
is just not a test of problem #890's specific unsolved claim, because no
such claim is encoded anywhere in the available metadata for this problem.

Concretely:
  - Universe: integers 0..15 (4 qubits).
  - Classical property computed from first principles below: trial-division
    primality test, giving the *exact* set of primes in [0, 15]:
        {2, 3, 5, 7, 11, 13}
  - Quantum task: Grover's search algorithm, with a phase oracle built by
    explicitly marking those computed prime basis states with multi-controlled
    Z gates (no lookup table, no OEIS value pasted in -- the oracle is
    constructed directly from the classically-computed prime set). After the
    optimal number of Grover iterations, ideal AerSimulator sampling should
    concentrate amplitude on exactly the 6 prime states out of 16.
  - Verification: run the circuit, take the most-frequently-sampled
    computational basis states, and check that this measured set equals the
    classically computed prime set. PASS/FAIL is printed based on that
    comparison.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_primes_below(n: int) -> list[int]:
    """Trial-division primality test, first principles, no external data."""
    primes = []
    for k in range(n):
        if k < 2:
            continue
        is_prime = True
        for d in range(2, int(math.isqrt(k)) + 1):
            if k % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(k)
    return primes


def build_oracle(n_qubits: int, marked_states: list[int]) -> QuantumCircuit:
    """Phase oracle flipping the sign of each marked computational basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
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


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover_prime_search(n_qubits: int, marked_states: list[int], shots: int = 4096):
    n_total = 2 ** n_qubits
    m = len(marked_states)

    # Optimal number of Grover iterations for m marked out of n_total states.
    theta = math.asin(math.sqrt(m / n_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(n_qubits, marked_states)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_qubits = 4
    n_total = 2 ** n_qubits  # 16

    classical_primes = classical_primes_below(n_total)
    print(f"Classical primes in [0, {n_total - 1}): {classical_primes}")

    counts, iterations = run_grover_prime_search(n_qubits, classical_primes)
    print(f"Grover iterations used: {iterations}")

    total_shots = sum(counts.values())
    # Sort measured states by frequency, take the top len(classical_primes) as
    # the quantum-search result.
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_states = sorted_counts[: len(classical_primes)]
    measured_set = sorted(int(bitstring, 2) for bitstring, _ in top_states)

    measured_probability_mass = sum(c for _, c in top_states) / total_shots
    print(f"Top {len(classical_primes)} measured states (little/big-endian as sampled): "
          f"{[b for b, _ in top_states]}")
    print(f"Measured integer set: {measured_set}")
    print(f"Probability mass on those top states: {measured_probability_mass:.3f}")

    verified = measured_set == sorted(classical_primes) and measured_probability_mass > 0.5

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
