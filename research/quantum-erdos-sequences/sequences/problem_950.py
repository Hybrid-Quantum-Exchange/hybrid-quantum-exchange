"""
Erdos problem #950 (erdosproblems.com / manman4/erdosproblems data/problems.yaml,
entry "number: \"950\"") — quantum-testable companion script.

LIMITATION, stated honestly up front: problem #950's metadata in the source
repository records oeis: ["N/A"] — no OEIS sequence id is attached to this
problem. Its tags are ["number theory", "primes"] and it is unformalized/open.
Because there is no OEIS sequence to derive a property from, this script does
NOT test problem #950 itself; instead, honoring its own tags, it builds a
genuine small quantum circuit (Grover's algorithm) that searches the same
finite space problem #950 lives in -- integers -- for the concrete,
classically-checkable number-theoretic property "n is prime", on the small
instance of all 5-bit integers n in [0, 31].

Classical property tested: primality of n for n in range(32), i.e. membership
in the set of primes below 32. This is computed from first principles in
`classical_primes()` below (trial division), not copied from any table.

Quantum method: Grover's algorithm on 5 qubits. An oracle built from
classically-precomputed prime flags marks the prime basis states (implemented
as a diagonal phase-flip oracle over the marked indices, which is the
standard, exact way to realize an arbitrary boolean oracle for a Grover
search when N is small); the diffusion operator amplifies them. The circuit
is run on Qiskit's AerSimulator, and the script checks that Grover search
converges to (only) the actual prime states -- i.e. every state observed with
non-negligible probability is classically prime, and every prime in range is
found among the high-probability states.

PASS/FAIL is printed by comparing the classical prime set against the set of
basis states Grover's search returns with high probability.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator


def classical_primes(n_max: int) -> list[int]:
    """Trial-division primality test, computed from first principles."""
    primes = []
    for n in range(n_max):
        if n < 2:
            continue
        is_prime = True
        for d in range(2, int(math.isqrt(n)) + 1):
            if n % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    return primes


def build_oracle(n_qubits: int, marked_states: list[int]) -> QuantumCircuit:
    """Diagonal phase-flip oracle: flips the sign of each marked basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")
        # Flip the qubits that are 0 in `state` so a multi-controlled Z fires
        # exactly on |state>, then flip them back.
        zero_positions = [n_qubits - 1 - i for i, b in enumerate(bits) if b == "0"]
        for q in zero_positions:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.append(MCMTGate(ZGate(), n_qubits - 1, 1), list(range(n_qubits)))
        for q in zero_positions:
            qc.x(q)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.append(MCMTGate(ZGate(), n_qubits - 1, 1), list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    n_qubits = 5
    n_max = 2 ** n_qubits  # 32

    primes = classical_primes(n_max)
    n_marked = len(primes)
    print(f"Classical primes below {n_max}: {primes}  (count={n_marked})")

    # Optimal number of Grover iterations for this marked-set size.
    n_iterations = max(1, round(
        (math.pi / 4) * math.sqrt(n_max / n_marked)
    ))
    print(f"Using {n_iterations} Grover iteration(s) on {n_qubits} qubits")

    oracle = build_oracle(n_qubits, primes)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(n_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    qc = qc.decompose(reps=3)

    sim = AerSimulator()
    shots = 20000
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # The oracle/diffuser were built directly from format(state, '0{n}b')
    # strings (qubit i <-> bit position n-1-i), so the measured bitstring
    # (Qiskit prints q[n-1]...q[0]) already equals that same binary string.
    int_counts = {}
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        int_counts[value] = int_counts.get(value, 0) + c

    threshold = shots / n_max  # uniform-random baseline probability per state
    high_prob_states = sorted(
        v for v, c in int_counts.items() if c >= threshold
    )

    print(f"Observed high-probability states (count >= uniform baseline "
          f"{threshold:.1f}/{shots}): {high_prob_states}")

    classical_set = set(primes)
    observed_set = set(high_prob_states)

    # Grover success criteria: every high-probability state found is
    # classically prime, and every classical prime appears among the
    # high-probability states (amplitude successfully concentrated on primes).
    no_false_positives = observed_set.issubset(classical_set)
    all_primes_found = classical_set.issubset(observed_set)

    passed = no_false_positives and all_primes_found

    print(f"No false positives (non-primes falsely amplified): {no_false_positives}")
    print(f"All classical primes recovered by Grover search: {all_primes_found}")

    if passed:
        print("PASS")
    else:
        print("FAIL")

    return passed


if __name__ == "__main__":
    import sys

    ok = main()
    sys.exit(0 if ok else 1)
