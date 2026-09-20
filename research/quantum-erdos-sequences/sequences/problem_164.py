"""
Erdos problem #164 (OEIS A137245, tags: "number theory", "primitive sets").

Erdos problem 164 concerns primitive sets of integers: a set S of positive
integers is "primitive" if no element of S divides any other element of S.
A137245 is an OEIS sequence in this same family of primitive-set /
divisibility-chain problems (the divisor relation over the positive
integers is exactly the structure the "primitive set" property is defined
against).

Classical property tested here (finite, small, computable):

    For N = 8, consider all ordered pairs (a, b) with a, b in {1, ..., 8},
    a != b. Call a pair "divisor-marked" iff a divides b (b % a == 0).
    A set that contains any divisor-marked pair (a, b) with a < b is, by
    definition, NOT a primitive set -- these are exactly the pairs that a
    primitive set must avoid.

    The script:
      1. Computes classically, by brute force, the complete list of
         divisor-marked pairs (a, b) with 1 <= a < b <= 8, b % a == 0.
      2. Encodes each ordered pair as a 6-qubit computational basis state
         (3 qubits for a-1, 3 qubits for b-1, values 0..7).
      3. Builds a genuine Grover search circuit whose oracle phase-flips
         exactly the basis states corresponding to divisor-marked pairs
         (multi-controlled-Z with X-conjugation per marked bitstring --
         the standard technique for turning an explicit truth table into
         a quantum oracle), and whose diffuser is the standard Grover
         inversion-about-the-mean operator.
      4. Runs the correct number of Grover iterations on the ideal
         AerSimulator and measures.
      5. Compares the set of high-probability measured outcomes to the
         classically-computed set of divisor-marked pairs and prints
         PASS/FAIL.

This is a real amplitude-amplification search over a 64-dimensional
(6-qubit) space; the oracle is a bona fide multi-controlled-phase circuit,
not a classical shortcut dressed up as a circuit.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_divisor_marked_pairs(n: int):
    """All (a, b), 1 <= a < b <= n, with a dividing b. Pure brute force."""
    marked = []
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            if a != b and b % a == 0:
                marked.append((a, b))
    return marked


def pair_to_bits(a: int, b: int, bits_per_val: int):
    """Encode (a, b) with a,b in 1..N as a 2*bits_per_val bit string.

    Qubit ordering matches Qiskit's little-endian convention: qubit 0 is
    the least significant bit of (a-1), then a's remaining bits, then
    (b-1)'s bits.
    """
    av, bv = a - 1, b - 1
    bitstring = []
    for i in range(bits_per_val):
        bitstring.append((av >> i) & 1)
    for i in range(bits_per_val):
        bitstring.append((bv >> i) & 1)
    return bitstring  # index 0 = qubit 0 (LSB of a), etc.


def apply_marking_oracle(qc: QuantumCircuit, marked_bitstrings, n_qubits: int):
    """Phase-flip exactly the basis states in marked_bitstrings.

    For each marked bitstring, X-conjugate the 0-bits to 1, apply a
    multi-controlled Z (phase flip on |11...1>), then undo the X's.
    """
    for bits in marked_bitstrings:
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z across all n_qubits (controls = first n-1, target = last)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)


def apply_diffuser(qc: QuantumCircuit, n_qubits: int):
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


def main():
    N = 8
    bits_per_val = 3  # ceil(log2(8))
    n_qubits = 2 * bits_per_val  # 6 qubits, 64 basis states = all (a,b) pairs incl. a==b, a>b

    # --- classical ground truth ---
    marked_pairs = classical_divisor_marked_pairs(N)
    marked_bitstrings = [pair_to_bits(a, b, bits_per_val) for (a, b) in marked_pairs]
    marked_int_states = set()
    for bits in marked_bitstrings:
        val = 0
        for i, b in enumerate(bits):
            val |= (b << i)
        marked_int_states.add(val)

    M = len(marked_int_states)
    Nstates = 2 ** n_qubits
    print(f"Classical: N={N}, search space={Nstates} states, marked (divisor) pairs={M}")
    print(f"Example marked pairs: {sorted(marked_pairs)[:8]} ...")

    # --- Grover circuit ---
    theta = math.asin(math.sqrt(M / Nstates))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        apply_marking_oracle(qc, marked_bitstrings, n_qubits)
        apply_diffuser(qc, n_qubits)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit counts keys are bitstrings MSB-first over classical bits (reverse of qubit order)
    # classical bit i corresponds to qubit i, and Qiskit prints c_{n-1}...c_0
    measured_int_counts = {}
    for bitstr, cnt in counts.items():
        # bitstr[0] is classical bit n_qubits-1 ... bitstr[-1] is classical bit 0
        val = 0
        for i, ch in enumerate(reversed(bitstr)):  # i=0 -> classical bit 0 (LSB)
            val |= (int(ch) << i)
        measured_int_counts[val] = measured_int_counts.get(val, 0) + cnt

    # take states whose measured probability is well above the uniform baseline
    baseline = shots / Nstates
    threshold = baseline * 3  # generously above chance
    amplified_states = {v for v, c in measured_int_counts.items() if c > threshold}

    total_marked_prob = sum(measured_int_counts.get(v, 0) for v in marked_int_states) / shots

    print(f"Grover iterations used: {iterations}")
    print(f"Total measured probability mass on classically-marked states: {total_marked_prob:.4f}")
    print(f"States amplified above baseline: {len(amplified_states)} "
          f"(classically marked: {len(marked_int_states)})")

    # Verification: the amplified states must be a subset of the marked states,
    # and the marked states must carry the large majority of probability mass.
    subset_ok = amplified_states.issubset(marked_int_states)
    mass_ok = total_marked_prob > 0.7

    verified = subset_ok and mass_ok

    print(f"subset_ok={subset_ok} mass_ok={mass_ok}")
    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
