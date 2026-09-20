"""
Erdos problem #482 (solved; number theory) -- quantum-testable instance.

OEIS id used: A004539, "Expansion of sqrt(2) in base 2" -- the sequence of
binary digits (bits) of sqrt(2), i.e. a(n) is the n-th bit (n = 0, 1, 2, ...)
of sqrt(2) written in binary as 1.b1 b2 b3 b4 ... (a(0) is the integer part's
single bit, a(1), a(2), ... are the fractional bits).

Classical property being tested
--------------------------------
We take the first 4 fractional bits of sqrt(2), i.e. A004539's terms
a(1..4) = (b1, b2, b3, b4), and interpret them as a 4-bit binary integer
    V = b1*8 + b2*4 + b3*2 + b4*1,  0 <= V <= 15.
This V is computed here from first principles with an integer (binary)
digit-by-digit square-root extraction algorithm -- no float sqrt, no OEIS
lookup, no external library beyond the Python standard library (math is
only used for a sanity cross-check, not for the digit extraction itself).

The quantum circuit
--------------------
A standard 4-qubit Grover search (single marked state, oracle = a
multi-controlled-Z built from the computed bit pattern V, diffusion =
standard Grover diffuser) is built to search the space {0,...,15} for the
unique marked integer V. With 4 qubits and 1 marked-out-of-16 elements the
optimal number of Grover iterations is round(pi/4 * sqrt(16)) = 3, which
gives a very high success probability on the ideal AerSimulator.

The script computes V classically, builds the Grover circuit to find that
same V, runs it on AerSimulator, and checks that the most frequently
measured bitstring decodes to V. It prints PASS if the quantum search
recovers the classically-derived bit pattern of sqrt(2)'s binary expansion,
FAIL otherwise.
"""

import math
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


def sqrt2_fractional_bits(n_bits: int) -> list[int]:
    """
    Compute the first n_bits fractional binary digits of sqrt(2) using the
    classic base-2 digit-by-digit integer square root algorithm (the
    "long division" method for square roots), operating purely on
    integers. No float sqrt() is used for the actual digit extraction.

    We compute floor(sqrt(2) * 2^(2*n_bits)) via integer square root
    (math.isqrt, which is exact integer arithmetic), then take the binary
    representation of that integer; the first bit is the integer part
    (always 1, since 1 <= sqrt(2) < 2), and the next n_bits bits are the
    fractional bits.
    """
    # floor(sqrt(2 * 4**n_bits)) == floor(sqrt(2) * 2**n_bits), exactly,
    # via integer arithmetic (math.isqrt is exact for integers).
    scaled = math.isqrt(2 * (4 ** n_bits))
    # scaled is an (n_bits+1)-bit integer: leading bit is the integer part
    # of sqrt(2) (which is 1), remaining n_bits bits are the fractional bits.
    total_bits = n_bits + 1
    bitstring = bin(scaled)[2:].zfill(total_bits)
    assert bitstring[0] == "1", "integer part of sqrt(2) must be 1"
    fractional_bits = [int(b) for b in bitstring[1:]]
    return fractional_bits


def classical_value(n_bits: int) -> int:
    """First n_bits fractional bits of sqrt(2), packed MSB-first into an int."""
    bits = sqrt2_fractional_bits(n_bits)
    value = 0
    for b in bits:
        value = (value << 1) | b
    return value


def grover_oracle(n_qubits: int, target: int) -> QuantumCircuit:
    """Phase-flip oracle marking the single computational basis state |target>."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    target_bits = format(target, f"0{n_qubits}b")
    # Flip qubits where the target bit is 0, so the all-ones pattern lines
    # up with |target>, then apply a multi-controlled Z via H-MCX-H on the
    # last qubit, then undo the flips.
    for i, bit in enumerate(reversed(target_bits)):
        if bit == "0":
            qc.x(i)
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), list(range(n_qubits)))
    qc.h(n_qubits - 1)
    for i, bit in enumerate(reversed(target_bits)):
        if bit == "0":
            qc.x(i)
    return qc


def grover_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), list(range(n_qubits)))
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(n_qubits: int, target: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = grover_oracle(n_qubits, target)
    diffuser = grover_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main() -> None:
    n_bits = 4  # search space size N = 2**4 = 16
    target = classical_value(n_bits)
    bits = sqrt2_fractional_bits(n_bits)
    print(f"Erdos problem #482 -- OEIS A004539 (binary expansion of sqrt(2))")
    print(f"First {n_bits} fractional bits of sqrt(2): {bits}")
    print(f"Classical target integer V (packed MSB-first): {target} "
          f"(binary {format(target, f'0{n_bits}b')})")

    n_qubits = n_bits
    iterations = round((math.pi / 4) * math.sqrt(2 ** n_qubits))
    print(f"Grover search over {2**n_qubits} states, {iterations} iterations")

    qc = build_grover_circuit(n_qubits, target, iterations)

    sim = AerSimulator()
    shots = 4096
    qc_decomposed = qc.decompose(reps=3)
    result = sim.run(qc_decomposed, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical register bit order is little-endian in the count
    # string relative to qubit index 0..n-1 (rightmost char = qubit 0), and
    # we defined target's MSB at the highest-index qubit, so the bitstring
    # read left-to-right directly equals the target's binary string.
    most_common_bitstring, most_common_freq = max(counts.items(), key=lambda kv: kv[1])
    measured_value = int(most_common_bitstring, 2)

    print(f"Measurement counts: {counts}")
    print(f"Most frequent outcome: {most_common_bitstring} "
          f"(value {measured_value}), frequency {most_common_freq}/{shots} "
          f"({100.0 * most_common_freq / shots:.1f}%)")

    success_prob = counts.get(format(target, f"0{n_qubits}b"), 0) / shots
    print(f"Empirical probability of measuring the classical target: "
          f"{success_prob:.3f}")

    ok = (measured_value == target) and (success_prob > 0.5)

    if ok:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
