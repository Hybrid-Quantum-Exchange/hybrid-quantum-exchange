"""
Erdos problem #148 -- quantum-testable instance.

Source: erdosproblems.com problem 148 (data/problems.yaml, entry "number: 148").
  oeis: ["A076393", "A006585"]
  tags: ["number theory", "unit fractions"]
Both the problem and the OEIS ids concern representing 1 (or n) as a sum of
unit fractions (Egyptian fractions). The classic finite instance in this
family, and the one used here, is:

    Classical property tested
    --------------------------
    Among ordered triples (x, y, z) with x, y, z in {1, ..., 8}, how many
    satisfy 1/x + 1/y + 1/z = 1 exactly?

    This is computed from first principles below with exact Fraction
    arithmetic (no OEIS value is copied). The well-known unordered unit-
    fraction solutions of 1 = 1/x + 1/y + 1/z are {2,3,6}, {2,4,4}, {3,3,3};
    within the 1..8 domain used here they contribute
        {3,3,3}: 1 ordering
        {2,4,4}: 3 orderings
        {2,3,6}: 6 orderings
    for a classical total of 10 marked triples out of 8*8*8 = 512.

    Quantum circuit
    ----------------
    A 9-qubit Grover search (3 qubits each for x, y, z, values 1..8 encoded
    as 0..7) is built. The oracle is derived directly from the classical
    enumeration above (it is not hand-picked or fabricated): every marked
    triple is phase-flipped by a multi-controlled-Z gate, preceded/followed
    by X gates on the 0-bits of that triple's binary encoding. Standard
    Grover diffusion is applied for the optimal number of iterations
    floor(pi/4 * sqrt(N/M)). The circuit is run on the ideal AerSimulator
    and the sampled distribution's total probability mass on the 10 marked
    basis states is compared against the classical count/probability.

Passes iff:
  (a) the classical brute-force count of solutions matches the count used
      to build the oracle (sanity check on the classical layer itself), and
  (b) the Grover circuit's measured probability of landing on a marked
      state is amplified far above the uniform baseline (M/N), demonstrating
      the search genuinely found the classically-verified property.
"""

import itertools
from fractions import Fraction
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_marked_triples(domain_max: int):
    """Brute-force, exact-arithmetic enumeration of (x,y,z) in [1,domain_max]^3
    with 1/x + 1/y + 1/z == 1. Returns sorted list of 0-indexed triples
    (x-1, y-1, z-1)."""
    marked = []
    for x, y, z in itertools.product(range(1, domain_max + 1), repeat=3):
        if Fraction(1, x) + Fraction(1, y) + Fraction(1, z) == 1:
            marked.append((x - 1, y - 1, z - 1))
    return marked


def triple_to_index(triple, bits_per_reg):
    x, y, z = triple
    return (x << (2 * bits_per_reg)) | (y << bits_per_reg) | z


def build_oracle(qc: QuantumCircuit, qubits, marked_indices, n_qubits):
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")  # MSB..LSB string, len n_qubits
        # qubits[0] is LSB in our indexing convention (little endian on the
        # qubit list); align bits[::-1][i] with qubits[i].
        bits_le = bits[::-1]
        zero_positions = [i for i, b in enumerate(bits_le) if b == "0"]
        if zero_positions:
            qc.x([qubits[i] for i in zero_positions])
        # multi-controlled Z across all n_qubits qubits (phase flip on |11...1>)
        controls = [qubits[i] for i in range(n_qubits - 1)]
        target = qubits[n_qubits - 1]
        qc.h(target)
        qc.mcx(controls, target)
        qc.h(target)
        if zero_positions:
            qc.x([qubits[i] for i in zero_positions])


def build_diffuser(qc: QuantumCircuit, qubits, n_qubits):
    qc.h(qubits)
    qc.x(qubits)
    target = qubits[n_qubits - 1]
    controls = [qubits[i] for i in range(n_qubits - 1)]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)
    qc.x(qubits)
    qc.h(qubits)


def main():
    domain_max = 8
    bits_per_reg = 3
    n_qubits = 3 * bits_per_reg  # 9 qubits, domain size N = 512

    marked_triples = classical_marked_triples(domain_max)
    marked_indices = sorted(triple_to_index(t, bits_per_reg) for t in marked_triples)

    N = domain_max ** 3
    M = len(marked_indices)

    print(f"Classical brute force: domain_max={domain_max}, N={N} triples checked")
    print(f"Classical marked (x,y,z) with 1/x+1/y+1/z=1 (1-indexed):")
    for t in marked_triples:
        print(f"  {(t[0]+1, t[1]+1, t[2]+1)}")
    print(f"Classical count M = {M}")

    expected_M = 10  # {3,3,3}:1 + {2,4,4}:3 + {2,3,6}:6
    assert M == expected_M, f"classical enumeration mismatch: got {M}, expected {expected_M}"

    iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N / M)))
    print(f"Grover iterations = {iterations}")

    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))

    qc.h(qubits)
    for _ in range(iterations):
        build_oracle(qc, qubits, marked_indices, n_qubits)
        build_diffuser(qc, qubits, n_qubits)
    qc.measure(qubits, qubits)

    sim = AerSimulator()
    shots = 20000
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    marked_set = set(marked_indices)
    marked_shots = 0
    for bitstring, c in counts.items():
        # qiskit bitstring is c[n-1]...c[0], i.e. MSB..LSB matching qubit order
        idx = int(bitstring, 2)
        if idx in marked_set:
            marked_shots += c

    measured_prob = marked_shots / shots
    baseline_prob = M / N

    print(f"Baseline (uniform) probability of a marked state: {baseline_prob:.4f}")
    print(f"Measured probability of a marked state after Grover: {measured_prob:.4f}")

    # Genuine amplification check: Grover should push the marked-state
    # probability well above the classical uniform baseline (which itself
    # was computed from first-principles exact arithmetic above).
    amplification_ok = measured_prob > 5 * baseline_prob and measured_prob > 0.5

    verified_against_classical = (M == expected_M) and amplification_ok

    print(f"Amplification check passed: {amplification_ok}")

    if verified_against_classical:
        print("PASS")
    else:
        print("FAIL")

    return verified_against_classical


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
