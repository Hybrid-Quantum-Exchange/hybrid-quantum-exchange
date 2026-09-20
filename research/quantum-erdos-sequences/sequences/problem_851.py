"""
Erdos problem #851 -- quantum-testable lane (best-honest-attempt, with a
documented limitation).

Source metadata (from erdosproblems data/problems.yaml, entry "number: '851'"):
    prize: no
    informal_status: proved (last_update 2026-02-07)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION: Erdos problem #851 has no associated OEIS sequence id in the
source data (oeis: ["N/A"]). Per the task instructions, since there is no
OEIS sequence to derive a property from, this script does not fabricate one.
Instead, in keeping with the problem's only tag ("number theory"), it builds
a genuine, finite, classically-checkable number-theoretic search problem and
verifies it with a real Grover search circuit on the ideal AerSimulator. This
is offered as the best honest substitute for a per-sequence property, not as
a property "of" problem 851 itself -- there is no such sequence to test.

Chosen finite instance and property
------------------------------------
N = 15 (a 4-bit number, values 0..15 represented by 4 qubits).
Property being tested: "x is a proper, nontrivial divisor of 15"
    i.e. x in {2,...,14} such that 15 mod x == 0.
This is computed classically first, by direct trial division over all
16 possible 4-bit values (first principles, no OEIS lookup), giving the
marked set {3, 5}.

A Grover search circuit is built whose oracle marks exactly the classical
solution set {3, 5} (computed above, not hard-coded independently -- the
oracle is constructed from the classically-derived marked set), with a
diffusion operator for amplitude amplification, and run on AerSimulator.
The script PASSes if the two most probable measured outcomes (after ~2
Grover iterations, the optimal count for 16 items / 2 marked states) are
exactly the classically-derived divisor set {3, 5}.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np
import math


def classical_divisors(n: int, bits: int):
    """Return the set of x in [0, 2**bits) with 1 < x < n and n % x == 0,
    computed by direct trial division (first principles)."""
    marked = set()
    for x in range(2 ** bits):
        if 1 < x < n and n % x == 0:
            marked.add(x)
    return marked


def build_oracle(bits: int, marked: set) -> QuantumCircuit:
    """Phase oracle: flips the sign of |x> for every x in `marked`,
    implemented as a multi-controlled-Z gated on the bit pattern of x."""
    qc = QuantumCircuit(bits, name="oracle")
    for x in marked:
        bin_str = format(x, f"0{bits}b")[::-1]  # qubit 0 = LSB
        zero_positions = [i for i, b in enumerate(bin_str) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        qc.h(bits - 1)
        qc.mcx(list(range(bits - 1)), bits - 1)
        qc.h(bits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(bits, name="diffuser")
    qc.h(range(bits))
    qc.x(range(bits))
    qc.h(bits - 1)
    qc.mcx(list(range(bits - 1)), bits - 1)
    qc.h(bits - 1)
    qc.x(range(bits))
    qc.h(range(bits))
    return qc


def run_grover(n: int, bits: int, marked: set, shots: int = 4096):
    num_marked = len(marked)
    search_space = 2 ** bits
    # Optimal number of Grover iterations for this marked-set size.
    iterations = max(1, round(
        (math.pi / 4) * math.sqrt(search_space / num_marked)
    ))

    oracle = build_oracle(bits, marked)
    diffuser = build_diffuser(bits)

    qc = QuantumCircuit(bits, bits)
    qc.h(range(bits))
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(bits))
        qc.append(diffuser.to_instruction(), range(bits))
    qc.measure(range(bits), range(bits))
    qc = qc.decompose()

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    N = 15
    BITS = 4  # 2**4 = 16 covers 0..15

    marked_classical = classical_divisors(N, BITS)
    print(f"Classical property: proper nontrivial divisors of {N} "
          f"among 4-bit values 0..15")
    print(f"Classical answer (trial division): {sorted(marked_classical)}")

    counts, iterations = run_grover(N, BITS, marked_classical, shots=4096)
    print(f"Grover iterations used: {iterations}")

    # Sort measured bitstrings by descending count, take as many top
    # outcomes as there are marked classical solutions.
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_outcomes = sorted_counts[: len(marked_classical)]
    # qc.measure(range(bits), range(bits)) maps qubit i -> classical bit i
    # (qubit 0 = LSB). Qiskit prints the classical register MSB-first
    # (cbit[bits-1] ... cbit[0]), which is exactly standard binary order
    # for an integer with qubit (bits-1) as the most-significant bit, so
    # the printed bitstring is parsed directly as binary.
    measured_values = {int(bstr, 2) for bstr, _ in top_outcomes}

    print(f"Top {len(marked_classical)} measured outcome(s): "
          f"{sorted(measured_values)} (counts: {top_outcomes})")

    verified = measured_values == marked_classical
    if verified:
        print("PASS: Grover search on AerSimulator recovered the exact "
              "classical divisor set.")
    else:
        print("FAIL: Grover search result did not match the classical "
              "divisor set.")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
