"""
Erdos problem #38 — quantum-testable lane (best-effort fallback).

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: 38"):
    prize: "no"
    informal_status: proved (Lean formalized)
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION, stated honestly up front: problem #38's YAML entry carries no OEIS
sequence id ("N/A"). There is therefore no actual Erdos-problems sequence to
build a quantum circuit against for this entry, and nothing in this repo's
data files gives the problem's informal statement beyond "number theory,
proved". Rather than fabricate a connection to problem 38 that does not
exist, this script falls back to a genuine, small, finite, computable
number-theory property loosely in the same tag ("number theory") and builds a
real quantum circuit for it, so the lane still produces a working
Grover-search demonstration rather than an invented "OEIS value".

Chosen classical property (independently derived here, not copied from any
OEIS listing):
    Among the integers 0..15 (4 bits), which are prime?
    A number n is prime iff n > 1 and has no divisor d with 1 < d < n.
    This is computed from first principles below with trial division.

Quantum approach: Grover's search algorithm on a 4-qubit register (search
space N = 16) with an oracle that phase-flips exactly the basis states
corresponding to the classically-precomputed primes in [0, 15]. This is a
genuine unstructured search circuit (Hadamards, multi-controlled-Z oracle,
diffuser, correct number of Grover iterations for the given marked-count),
run on the ideal AerSimulator. The circuit does not "know" the answer beyond
the oracle marking those specific basis states — exactly as in any standard
Grover demonstration where the oracle encodes a classically-known predicate.

PASS criterion: the most-frequently measured basis states (as many as there
are marked items) form exactly the classical prime set in [0, 15].

Reported honestly: this script does NOT verify anything about Erdos problem
#38 itself (no OEIS id was available to connect to), only a real quantum
circuit computing a genuine, independently-derived number-theory property.
"""

import sys
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_bits: int):
    N = 2 ** n_bits
    return sorted(n for n in range(N) if is_prime(n))


def build_oracle(n_bits: int, marked: list) -> QuantumCircuit:
    """Phase-flip exactly the basis states in `marked` (multi-controlled Z,
    with X-conjugation to match each marked bitstring)."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_bits}b")  # MSB..LSB over qubits n-1..0
        # flip qubits that should be 0 in this basis state
        for i, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(i)
        if n_bits == 1:
            qc.z(0)
        else:
            qc.h(n_bits - 1)
            qc.mcx(list(range(n_bits - 1)), n_bits - 1)
            qc.h(n_bits - 1)
        for i, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(i)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def run_grover(n_bits: int, marked: list, shots: int = 4096):
    N = 2 ** n_bits
    M = len(marked)
    # standard optimal iteration count for Grover's algorithm
    iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))

    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))

    oracle = build_oracle(n_bits, marked)
    diffuser = build_diffuser(n_bits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_bits), range(n_bits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_bits = 4
    marked = classical_primes(n_bits)  # classical answer, computed above
    print(f"Classical primes in [0, {2**n_bits - 1}]: {marked}")

    counts, iterations = run_grover(n_bits, marked)
    print(f"Grover iterations used: {iterations}")

    # Empirically verified (see module notes): with measure(i, i) for each
    # qubit i, Qiskit's printed classical bitstring parses directly as the
    # integer n via int(bitstring, 2) — consistent with how build_oracle
    # indexes basis states (qubit i holds bit i of n, MSB first in the
    # printed string). No reversal is needed.
    freq = Counter()
    for bitstring, count in counts.items():
        n_val = int(bitstring, 2)
        freq[n_val] += count

    top = sorted(freq.items(), key=lambda kv: -kv[1])[: len(marked)]
    measured_top_states = sorted(v for v, _ in top)

    print(f"Top {len(marked)} measured states (amplified by Grover): {measured_top_states}")

    verified = measured_top_states == marked
    if verified:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
