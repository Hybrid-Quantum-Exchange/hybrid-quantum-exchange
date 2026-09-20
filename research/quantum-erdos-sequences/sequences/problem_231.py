"""
Erdos problem #231 -- quantum-testable sequence lane.

Source metadata (from erdosproblems.com data, /home/user/manman4/erdosproblems/
data/problems.yaml, entry `number: "231"`):
    prize: no
    informal_status: disproved (Lean-formalized 2026-05-14)
    oeis: ["N/A"]
    tags: ["combinatorics"]

LIMITATION (reported honestly, as instructed): problem 231 carries no OEIS
sequence id -- its `oeis` field is literally `["N/A"]` -- and no problem
description/statement file exists in the cloned repository under any name
containing "231". There is therefore no OEIS-derived integer sequence to
build a quantum-testable property from for this specific problem, and no
problem text to derive a bespoke combinatorial property from either. Per the
task's own fallback instructions, this script is a best-honest-effort
substitute: a genuinely finite, computable, quantum-searchable number-
theoretic property in the same spirit as the "combinatorics" tag, rather
than a manufactured claim of relevance to problem 231's actual (unavailable)
content. Treat `verified_against_classical=True` as validating the circuit
construction, not as validating any connection to problem 231 itself.

Chosen property (self-contained, no OEIS dependency):
    "Which 6-bit integers x in [0, 63] are prime?"
This is a small, finite, exactly-computable search problem (64 items, 6
qubits) well suited to Grover's algorithm. The classical answer -- the exact
set of primes in [0, 63] -- is computed from first principles in this script
via trial division (no external tables, no hardcoded OEIS values), and used
both to build the oracle and to check the quantum search's output.

Circuit: a 6-qubit Grover search over all 64 basis states |x>, with an
oracle built as a phase-flip diagonal matrix (Diagonal gate) that marks
exactly the primes found classically, followed by the standard diffusion
operator, iterated the Grover-optimal number of times for
sqrt(64/#primes). The circuit is run on the ideal AerSimulator, and PASS/FAIL
is decided by checking that the most-frequently-measured outcomes are all in
the classically-computed prime set.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_qubits: int):
    N = 2 ** n_qubits
    return sorted(x for x in range(N) if is_prime(x))


def build_grover_circuit(n_qubits: int, marked: list) -> QuantumCircuit:
    N = 2 ** n_qubits

    # --- Oracle: diagonal phase flip on marked states ---
    diag = np.ones(N, dtype=complex)
    for m in marked:
        diag[m] = -1.0

    oracle = DiagonalGate(list(diag))

    # --- Diffusion operator (inversion about the mean) ---
    diffusion = QuantumCircuit(n_qubits, name="Diffusion")
    diffusion.h(range(n_qubits))
    diffusion.x(range(n_qubits))
    diffusion.h(n_qubits - 1)
    diffusion.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    diffusion.h(n_qubits - 1)
    diffusion.x(range(n_qubits))
    diffusion.h(range(n_qubits))

    num_marked = len(marked)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / num_marked)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle, range(n_qubits))
        qc.append(diffusion.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc, iterations


def main():
    n_qubits = 6  # search space [0, 63]
    N = 2 ** n_qubits

    primes = classical_primes(n_qubits)
    print(f"Classical property: primes in [0, {N - 1}] (computed by trial division)")
    print(f"Classical answer ({len(primes)} primes): {primes}")

    qc, iterations = build_grover_circuit(n_qubits, primes)
    print(f"Grover circuit built: {n_qubits} qubits, {iterations} iteration(s)")

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Convert bitstrings (qiskit little-endian) to integers.
    int_counts = Counter()
    for bitstring, c in counts.items():
        x = int(bitstring, 2)
        int_counts[x] += c

    # Take the outcomes that together account for the top mass of shots,
    # up to as many distinct outcomes as there are primes, and check they
    # are all in the classically-computed prime set.
    top = [x for x, _ in int_counts.most_common(len(primes))]
    top_mass = sum(int_counts[x] for x in top) / shots

    print(f"Top {len(top)} measured outcomes (by frequency): {sorted(top)}")
    print(f"Fraction of shots landing on those outcomes: {top_mass:.3f}")

    all_marked_are_prime = all(x in set(primes) for x in top)
    amplification_worked = top_mass > 0.5  # much better than uniform baseline

    passed = all_marked_are_prime and amplification_worked

    print(f"All top outcomes are prime: {all_marked_are_prime}")
    print(f"Amplification above uniform baseline: {amplification_worked}")
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
