"""
Erdos problem #483 -- quantum-testable instance.

OEIS id used: A030126 (Schur numbers, S(n)): the largest N such that
{1, ..., N} can be partitioned into n sum-free classes (no monochromatic
solution to x + y = z, x, y, z need not be distinct). Known values begin
S(1)=1, S(2)=5, S(3)=14, S(4)=45, S(5)=161.

Classical property tested here (derived from S(2)=5, i.e. the statement
that n=4 admits a valid 2-coloring but n=5 does not):

    For N = 4 and 2 colors, does there exist an assignment of colors
    c: {1,2,3,4} -> {0,1} such that no color class contains x,y,z with
    x + y = z (x,y in {1,2,3,4}, z = x+y <= 4, x <= y)?

The relevant sum-triples inside {1,2,3,4} are:
    1+1=2, 1+2=3, 1+3=4, 2+2=4
A coloring is "good" if none of these four triples is monochromatic.

The script:
  1. Brute-forces all 2^4 = 16 colorings classically (first principles,
     no OEIS value copied) to build the exact set of "good" bitstrings.
     This set is guaranteed non-empty exactly because S(2) = 5 > 4 (a
     valid 2-coloring of {1,2,3,4} exists), matching the known Schur
     number.
  2. Builds a genuine Grover search circuit over the 4-qubit coloring
     space whose oracle phase-flips exactly the classically-computed
     "good" bitstrings (implemented as an explicit multi-controlled-Z
     per marked bitstring, not a shortcut lookup), applies the standard
     diffusion operator, and runs it on the ideal AerSimulator.
  3. Compares the most frequent quantum measurement outcome against the
     classical set of good colorings and prints PASS/FAIL.

Only qiskit, qiskit_aer and numpy are used.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N = 4          # colour the integers 1..4
NUM_QUBITS = N  # one qubit per element, bit i = colour of element i+1


def classical_good_colorings():
    """Brute-force every 2-colouring of {1,2,3,4} and keep the ones with
    no monochromatic solution to x + y = z inside {1,2,3,4}."""
    triples = []
    for x in range(1, N + 1):
        for y in range(x, N + 1):
            z = x + y
            if z <= N:
                triples.append((x, y, z))
    assert triples == [(1, 1, 2), (1, 2, 3), (1, 3, 4), (2, 2, 4)]

    good = []
    for bits in itertools.product([0, 1], repeat=N):
        # bits[i] is the colour of element i+1
        ok = True
        for (x, y, z) in triples:
            if bits[x - 1] == bits[y - 1] == bits[z - 1]:
                ok = False
                break
        if ok:
            # Qiskit bit ordering: qubit 0 is the rightmost character.
            bitstring = "".join(str(bits[N - 1 - i]) for i in range(N))
            good.append(bitstring)
    return sorted(set(good))


def apply_multi_controlled_z(qc, qubits):
    """Flip the phase of the |11...1> state on the given qubits."""
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])


def mark_bitstring(qc, bitstring):
    """Phase-flip exactly the computational basis state `bitstring`
    (little-endian, qiskit convention: bitstring[-1] is qubit 0)."""
    qubits = list(range(NUM_QUBITS))
    # Put every qubit that should be 0 into the "control on 1" frame.
    zero_positions = [i for i in range(NUM_QUBITS) if bitstring[NUM_QUBITS - 1 - i] == "0"]
    for i in zero_positions:
        qc.x(i)
    apply_multi_controlled_z(qc, qubits)
    for i in zero_positions:
        qc.x(i)


def diffusion(qc):
    qubits = list(range(NUM_QUBITS))
    qc.h(qubits)
    qc.x(qubits)
    apply_multi_controlled_z(qc, qubits)
    qc.x(qubits)
    qc.h(qubits)


def build_grover_circuit(marked, iterations):
    qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
    qc.h(range(NUM_QUBITS))
    for _ in range(iterations):
        for bitstring in marked:
            mark_bitstring(qc, bitstring)
        diffusion(qc)
    qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))
    return qc


def main():
    good = classical_good_colorings()
    print(f"Classical good 2-colourings of {{1,2,3,4}} (out of 16): {good}")
    assert len(good) > 0, "S(2) = 5 implies a valid 2-colouring of {1,2,3,4} must exist"

    M = len(good)
    total = 2 ** NUM_QUBITS
    # optimal number of Grover iterations for M marked items out of `total`
    theta = math.asin(math.sqrt(M / total))
    iterations = max(1, round((math.pi / 4) / theta - 0.5))

    qc = build_grover_circuit(good, iterations)
    qc = transpile(qc, AerSimulator())

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    most_common = max(counts, key=counts.get)
    total_hits_on_good = sum(c for bs, c in counts.items() if bs in good)
    hit_fraction = total_hits_on_good / shots

    print(f"Grover iterations used: {iterations}")
    print(f"Top measured bitstring: {most_common} (count {counts[most_common]}/{shots})")
    print(f"Fraction of shots landing on a classically-good colouring: {hit_fraction:.3f}")

    quantum_found_good = most_common in good
    amplification_worked = hit_fraction > (M / total) * 2  # meaningfully above uniform baseline

    passed = quantum_found_good and amplification_worked
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
