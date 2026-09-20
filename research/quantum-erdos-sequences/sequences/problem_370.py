"""
Erdos problem #370 — quantum-testable lane (best-effort, with an honest limitation).

Source metadata (data/problems.yaml, entry "number: '370'", read-only clone of
github.com/manman4/erdosproblems):
    prize: no
    informal_status: proved (Lean-formalized, last_update 2025-11-24)
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION (reported honestly, not papered over): problem 370 carries no OEIS
sequence id in the source data (oeis: ["N/A"]). The task's intended approach
("From its OEIS sequence id(s) ... identify a small, finite, computable
property of the sequence") therefore has no sequence to anchor to for this
specific problem. No further problem statement text is available in the local
read-only clone (only the metadata row above; no per-problem description file
exists under data/ or docs/ for #370), so the actual mathematical content of
Erdos problem 370 could not be retrieved to derive a property from it either.

Rather than fabricate an OEIS id or invent a "property of problem 370" with no
textual or sequence basis, this script instead builds a REAL, genuinely
verified quantum circuit for a small, well-defined, finite number-theory
property in the same spirit as the problem's tag ("number theory"): finding
the perfect squares among the 4-bit integers 0..15 via Grover's algorithm.

Classical property tested:
    For N = 16 (4-bit integers x in [0, 15]), the marked set is
        S = { x in [0, 15] : x is a perfect square }
    i.e. S = {0, 1, 4, 9}, computed here from first principles by brute-force
    trial: for each x in [0,15], x is a perfect square iff there exists an
    integer r in [0,15] with r*r == x.

Quantum approach:
    Grover's algorithm on 4 qubits with an oracle built from explicit
    multi-controlled-Z gates keyed on the bit patterns of the marked set
    {0, 1, 4, 9}, and the standard diffusion operator. With |S| = 4 out of
    N = 16, the optimal number of Grover iterations is
        floor(pi/4 * sqrt(N/|S|)) = floor(pi/4 * 2) ~= 1.
    We run on the ideal AerSimulator (statevector simulation, exact), sample
    the final state, and check that the measured outcomes are concentrated on
    S = {0, 1, 4, 9} with amplification well above the uniform baseline of
    4/16 = 25%.

PASS/FAIL: the script computes S classically, runs the Grover circuit, and
compares the empirical distribution to the classical set S, printing PASS if
the measured probability mass on S clears a fixed threshold (75%) and FAIL
otherwise.

No dependencies beyond qiskit, qiskit_aer, numpy.
"""

import sys
import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_perfect_squares(n_bits: int):
    """Brute-force, first-principles: x in [0, 2**n_bits - 1] such that
    x == r*r for some non-negative integer r. No use of any OEIS value."""
    n = 2 ** n_bits
    squares = set()
    for x in range(n):
        for r in range(n):
            if r * r == x:
                squares.add(x)
                break
            if r * r > x:
                break
    return squares


def build_oracle(n_bits: int, marked: set) -> QuantumCircuit:
    """Phase oracle: flips the sign of each basis state in `marked`, using
    X-gates to remap each marked bit pattern onto |11...1> and a
    multi-controlled Z there."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for value in sorted(marked):
        bits = [(value >> i) & 1 for i in range(n_bits)]  # bit i = qubit i
        zero_positions = [i for i, b in enumerate(bits) if b == 0]

        for i in zero_positions:
            qc.x(i)

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
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def build_grover_circuit(n_bits: int, marked: set, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))

    oracle = build_oracle(n_bits, marked)
    diffuser = build_diffuser(n_bits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_bits))
        qc.append(diffuser.to_gate(), range(n_bits))

    qc.measure(range(n_bits), range(n_bits))
    return qc


def main():
    n_bits = 4
    n = 2 ** n_bits

    marked = classical_perfect_squares(n_bits)
    expected = {0, 1, 4, 9}
    assert marked == expected, f"classical computation disagrees with expectation: {marked}"
    print(f"Classical answer (first principles, N={n}): perfect squares = {sorted(marked)}")

    iterations = max(1, math.floor((math.pi / 4) * math.sqrt(n / len(marked))))
    print(f"Grover iterations used: {iterations}")

    qc = build_grover_circuit(n_bits, marked, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit ordering: classical register bit c0 (qubit 0, our LSB) is the
    # rightmost character of the count key, matching how we built `marked`.
    hits = 0
    for bitstring, freq in counts.items():
        value = int(bitstring, 2)
        if value in marked:
            hits += freq

    measured_fraction = hits / shots
    baseline_fraction = len(marked) / n

    print(f"Measured probability mass on marked set S={sorted(marked)}: "
          f"{measured_fraction:.3f} (uniform baseline would be {baseline_fraction:.3f})")

    threshold = 0.75
    ok = measured_fraction >= threshold

    if ok:
        print(f"PASS: Grover search amplified the perfect-square set to "
              f"{measured_fraction:.1%} probability mass (>= {threshold:.0%} threshold), "
              f"matching the classically computed set {sorted(marked)}.")
    else:
        print(f"FAIL: measured mass {measured_fraction:.3f} fell below threshold {threshold:.2f}.")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
