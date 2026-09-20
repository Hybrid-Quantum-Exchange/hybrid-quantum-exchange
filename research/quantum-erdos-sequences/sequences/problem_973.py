"""
Erdos problem #973 -- quantum-testable-sequence lane.

Source metadata (from erdosproblems/data/problems.yaml, entry `number: "973"`):
    prize:           no
    informal_status: open (last_update 2025-08-31)
    formal_status:   unformalized
    oeis:             ["N/A"]
    tags:             ["analysis"]

LIMITATION (reported honestly, per the task's fallback instructions):
Problem #973 carries no OEIS sequence id at all (oeis == ["N/A"]). Without a
concrete integer sequence there is no membership/divisibility/counting
property of "the sequence" to encode in an oracle -- there is nothing to
derive or check classically that ties back to this specific Erdos problem's
mathematical content. Fabricating an OEIS-flavored property here would
violate the task's explicit instruction not to invent one with no real
mathematical content, so no genuine circuit *for problem 973's sequence* is
possible from the available metadata.

BEST-EFFORT FALLBACK: to still deliver a real, runnable, verifiable quantum
circuit in this lane (rather than an empty file), this script implements a
standalone but mathematically genuine small computation: Grover's search
algorithm over the 3-bit integers 0..7 to find the primes among them. This
property (primality of small integers) is at least the same *flavor* of
finite, computable number-theoretic property that a real OEIS-linked lane
would test (e.g. OEIS A000040, primes) -- but it is NOT derived from problem
973's own (nonexistent) OEIS entry, and must not be read as verifying
anything about Erdos problem 973 itself. This is disclosed explicitly here
and in the final report.

Classical answer (computed from first principles below, trial division):
    primes in {0,...,7} = {2, 3, 5, 7}

Quantum approach:
    Grover's algorithm with a 3-qubit register (values 0..7), a phase oracle
    that flags primes, and the standard diffusion operator, run on the ideal
    AerSimulator. We run enough Grover iterations for the 4-of-8 marked-item
    case (optimal iterations = 1) and check that measurement is heavily
    concentrated on the classically-computed prime set.

PASS/FAIL: the script prints PASS if the quantum circuit's most-probable
measurement outcomes match the classical prime set, and FAIL otherwise.
"""

import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_bits: int) -> set:
    """First-principles classical computation of primes in [0, 2**n_bits)."""
    return {n for n in range(2 ** n_bits) if is_prime(n)}


def build_oracle(n_bits: int, marked: set) -> QuantumCircuit:
    """Phase oracle: flips the sign of the amplitude for each marked basis state."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_bits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all n_bits qubits (control = first n_bits-1, target = last)
        if n_bits == 1:
            qc.z(0)
        else:
            qc.h(n_bits - 1)
            qc.mcx(list(range(n_bits - 1)), n_bits - 1)
            qc.h(n_bits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    qc.h(n_bits - 1)
    if n_bits == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
    qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def grover_find_primes(n_bits: int, marked: set, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))
    oracle = build_oracle(n_bits, marked)
    diffuser = build_diffuser(n_bits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_bits), range(n_bits))
    return qc


def best_iteration_count(n_bits: int, marked: set, max_iter: int = 6) -> int:
    """Pick the iteration count (by exact statevector simulation) that maximizes
    the total probability mass on the marked (prime) states."""
    from qiskit.quantum_info import Statevector

    best_k, best_p = 0, -1.0
    for k in range(0, max_iter + 1):
        qc = QuantumCircuit(n_bits)
        qc.h(range(n_bits))
        oracle = build_oracle(n_bits, marked)
        diffuser = build_diffuser(n_bits)
        for _ in range(k):
            qc.compose(oracle, inplace=True)
            qc.compose(diffuser, inplace=True)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()
        p_marked = sum(probs[m] for m in marked)
        if p_marked > best_p:
            best_p, best_k = p_marked, k
    return best_k


def main():
    n_bits = 4
    N = 2 ** n_bits  # 16

    marked = classical_primes(n_bits)
    print(f"Classical primes in [0, {N}): {sorted(marked)}")

    M = len(marked)
    iterations = best_iteration_count(n_bits, marked)
    print(f"Grover iterations used: {iterations} (N={N}, M={M}, chosen by exact amplitude search)")

    qc = grover_find_primes(n_bits, marked, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Convert bitstrings (qiskit reports MSB..LSB, little-endian qubit 0 = rightmost char)
    outcome_counts = {}
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        outcome_counts[value] = outcome_counts.get(value, 0) + c

    print("Measurement distribution (value: count):")
    for v in sorted(outcome_counts, key=lambda x: -outcome_counts[x]):
        print(f"  {v}: {outcome_counts[v]}")

    # Take the top-M most frequent outcomes as the circuit's "found" set.
    top_values = sorted(outcome_counts, key=lambda x: -outcome_counts[x])[:M]
    quantum_primes = set(top_values)

    verified = quantum_primes == marked
    print(f"Quantum top-{M} outcomes: {sorted(quantum_primes)}")
    print(f"Classical primes:         {sorted(marked)}")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
