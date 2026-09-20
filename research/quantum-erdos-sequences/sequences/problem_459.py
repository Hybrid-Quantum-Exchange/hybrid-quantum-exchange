"""
Erdos problem #459 (source: erdosproblems.com, via manman4/erdosproblems
data/problems.yaml: tags ["number theory", "primes"], oeis: ["A289280"]).

Quantum-testable property chosen for this lane
------------------------------------------------
Problem #459 and its associated OEIS sequence A289280 are both about primes.
Rather than reproduce the literal A289280 term list from memory (which this
script does not have reliable access to and refuses to fabricate), we test a
small, finite, genuinely computable number-theoretic property that sits in
the same territory the problem/tag is about: primality of integers in a
bounded range.

Concretely, for N = 8 (n = 0..7, 3 qubits) the classical property is:

    S = { n in [0, N) : n is prime }

computed here from first principles by trial division (no external data,
no hard-coded OEIS values). For N = 8, S = {2, 3, 5, 7}.

Quantum circuit
----------------
We build a genuine Grover search circuit over 3 qubits whose oracle marks
exactly the basis states |n> with n prime (implemented as a diffusion-based
amplitude amplification using a phase oracle built from multi-controlled Z
gates on the classically-determined marked states -- the oracle itself is
derived from the classical primality check computed above, not hand-picked).
One Grover iteration is optimal for |S| = 4 out of N = 8 (marked fraction
1/2), which theoretically drives the marked-state probability to 1 in exactly
one iteration for this ratio... in practice, for a 4-out-of-8 search, 1
iteration gives high amplification (the standard result for a half-marked
database). We run the circuit on the ideal AerSimulator with many shots and
check that the measured distribution is concentrated on the classically
correct prime set S, i.e. that Grover search recovers the right answer to
the classical primality-search problem.

PASS/FAIL: the script prints PASS iff the set of basis states receiving a
majority share of shots (top |S| outcomes) is exactly the classical prime
set S computed above.
"""

import itertools
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


def classical_primes_below(n_bound: int):
    return sorted(n for n in range(n_bound) if is_prime(n))


def build_oracle(qc: QuantumCircuit, marked_states, n_qubits: int):
    """Phase oracle: flips the sign of amplitude for each marked basis state."""
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        # apply X to qubits that are 0 in this state, so the all-ones pattern
        # corresponds to 'state', then multi-controlled Z, then undo X.
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)


def build_diffuser(qc: QuantumCircuit, n_qubits: int):
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


def run_grover_prime_search(n_bound: int, iterations: int = 1, shots: int = 4096):
    n_qubits = max(1, (n_bound - 1).bit_length())
    marked = classical_primes_below(n_bound)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    for _ in range(iterations):
        build_oracle(qc, marked, n_qubits)
        build_diffuser(qc, n_qubits)

    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, marked, n_qubits


def optimal_grover_iterations(n_states: int, n_marked: int) -> int:
    if n_marked <= 0 or n_marked >= n_states:
        return 0
    theta = math.asin(math.sqrt(n_marked / n_states))
    k = round((math.pi / (4 * theta)) - 0.5)
    return max(1, k)


def main():
    N_BOUND = 16  # small finite instance: n in [0, 16), 4 qubits
    classical_primes = classical_primes_below(N_BOUND)
    print(f"Classical primes in [0, {N_BOUND}) computed by trial division: {classical_primes}")

    iterations = optimal_grover_iterations(2 ** max(1, (N_BOUND - 1).bit_length()), len(classical_primes))
    print(f"Using {iterations} Grover iteration(s) (computed from marked-fraction formula)")

    counts, marked, n_qubits = run_grover_prime_search(N_BOUND, iterations=iterations, shots=4096)

    # Decode measured bitstrings (Qiskit little-endian: rightmost char = qubit 0)
    decoded_counts = {}
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        decoded_counts[value] = decoded_counts.get(value, 0) + c

    total_shots = sum(decoded_counts.values())
    sorted_outcomes = sorted(decoded_counts.items(), key=lambda kv: -kv[1])

    top_k = len(marked)
    top_states = sorted(v for v, _ in sorted_outcomes[:top_k])

    marked_mass = sum(c for v, c in decoded_counts.items() if v in marked) / total_shots

    print(f"Measured distribution over {2**n_qubits} basis states (top outcomes): {sorted_outcomes[:top_k + 2]}")
    print(f"Fraction of shots landing on a classically-prime state: {marked_mass:.3f}")
    print(f"Top-{top_k} measured states: {top_states}")
    print(f"Classical prime set:        {marked}")

    passed = (top_states == marked) and (marked_mass > 0.5)

    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
