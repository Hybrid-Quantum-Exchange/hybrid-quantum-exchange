"""
Erdos problem #429 -- quantum-testable sequence entry.

LIMITATION (read first): problem #429's metadata in
erdosproblems/data/problems.yaml records oeis: ["N/A"] -- it has no
associated OEIS sequence. It is tagged only ["number theory"] and its
status is "disproved (Lean)" with no further problem statement available
in the cloned repository (no docs/problems/429.* file exists). There is
therefore no real sequence to derive a quantum-testable property FROM for
this specific problem: any claim of "the OEIS sequence for #429" would be
fabricated.

Per instructions, this is the best honest attempt rather than a faked
pass: since the only real signal available for #429 is its tag
"number theory", this script builds a genuine, verifiable, finite
number-theory search problem -- Grover's algorithm searching the 4-bit
space {0, ..., 15} for the primes among the N=16 integers -- and checks
the quantum search result against a classical (trial-division) computation
of the same primality property, done from first principles in this script.

This is NOT a computation of an actual Erdos-429 OEIS term (none exists).
It is reported honestly below as: ran_ok=True (the circuit runs and
produces the correct answer), but verified_against_classical is true only
for the substitute number-theory property, not for problem #429's own
sequence, because problem #429 has no OEIS sequence to verify against.

Classical computation for the record (from first principles, trial
division, N = 16, i.e. 4-bit integers 0..15):
    primes in [0, 15] = {2, 3, 5, 7, 11, 13}   (6 of the 16 integers)

Quantum approach: Grover's algorithm with a 4-qubit register and a
primality oracle built from classical (reversible, via multi-controlled
Z / phase-flip) marking of the 6 known prime basis states, run on the
ideal AerSimulator. Grover amplifies the marked (prime) states; we check
that the measured distribution is concentrated on the classically-correct
prime set.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_primes_below(n: int) -> list[int]:
    """Trial-division primality test, first principles, for 0..n-1."""
    primes = []
    for k in range(2, n):
        is_prime = True
        for d in range(2, int(k ** 0.5) + 1):
            if k % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(k)
    return primes


def build_oracle(qc: QuantumCircuit, qubits, marked_states, n_qubits):
    """Phase-flip oracle: multi-controlled Z on each marked computational
    basis state (X-sandwich trick), applied to the given qubits."""
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian
        flip_qubits = [qubits[i] for i, b in enumerate(bits) if b == "0"]
        for q in flip_qubits:
            qc.x(q)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        for q in flip_qubits:
            qc.x(q)


def build_diffuser(qc: QuantumCircuit, qubits, n_qubits):
    """Standard Grover diffusion operator (inversion about the mean)."""
    for q in qubits:
        qc.h(q)
    for q in qubits:
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
    for q in qubits:
        qc.h(q)


def run_grover_prime_search(n_qubits: int, marked_states: list[int], shots: int = 4096):
    n = 2 ** n_qubits
    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))

    # Uniform superposition.
    for q in qubits:
        qc.h(q)

    m = len(marked_states)
    # Optimal number of Grover iterations for m marked out of n.
    iterations = max(1, round((np.pi / 4) * np.sqrt(n / m)))

    for _ in range(iterations):
        build_oracle(qc, qubits, marked_states, n_qubits)
        build_diffuser(qc, qubits, n_qubits)

    qc.measure(qubits, qubits)

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_qubits = 4
    n = 2 ** n_qubits  # N = 16

    classical_prime_set = set(classical_primes_below(n))
    print(f"Classical primes in [0, {n - 1}] (trial division): "
          f"{sorted(classical_prime_set)}")

    counts, iterations = run_grover_prime_search(
        n_qubits, sorted(classical_prime_set)
    )
    print(f"Grover iterations used: {iterations}")

    # Sum measurement probability landing on classically-correct primes.
    total_shots = sum(counts.values())
    hit_shots = 0
    for bitstring, c in counts.items():
        # Qiskit prints classical-register bitstrings as c[n-1]...c[0],
        # and clbit i was filled directly from qubit i (qc.measure(qubits,
        # qubits)), so the printed string is already qubit3..qubit0,
        # i.e. the standard MSB-first binary value.
        value = int(bitstring, 2)
        if value in classical_prime_set:
            hit_shots += c

    hit_fraction = hit_shots / total_shots
    print(f"Fraction of shots landing on a classically-verified prime: "
          f"{hit_fraction:.4f}")

    # Grover success criterion: heavily concentrated on marked states.
    # With 6 primes marked out of N=16 and 1 iteration, the ideal (noiseless)
    # success probability is sin(3*theta)^2 ~= 0.844, so a 0.85 threshold
    # could never pass even under perfect simulation; 0.75 keeps comfortable
    # margin below that ideal value while still requiring amplification far
    # above the 6/16 = 0.375 uniform baseline.
    quantum_agrees = hit_fraction > 0.75

    print()
    if quantum_agrees:
        print("PASS")
    else:
        print("FAIL")

    return quantum_agrees


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
