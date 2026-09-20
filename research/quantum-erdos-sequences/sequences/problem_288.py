"""
Erdos problem #288 (see erdosproblems.com/288) — quantum-testable companion script.

Source metadata (from manman4/erdosproblems data/problems.yaml, verified read-only
on 2026-09-19): problem 288 is an OPEN number-theory problem tagged
["number theory", "unit fractions"], with oeis: ["N/A"]. There is NO OEIS sequence
id attached to this problem, so this script cannot test "membership of a term in
OEIS sequence such-and-such" the way most entries in this library can — that would
require fabricating an OEIS id that does not exist in the source data.

LIMITATION (stated honestly, per the task's own fallback instructions): because
problem 288 has no associated OEIS id, this script instead tests a small, finite,
classically-checkable UNIT-FRACTION property in the same mathematical family as
problem 288's tag ("unit fractions"), which is a well-known and genuinely
finite/computable question:

    Property tested: for a, b each ranging over {1, 2, 3, 4} (encoded as 2-bit
    integers 0..3, representing a-1 and b-1), find the pair(s) (a, b) such that
        1/a + 1/b == 1
    i.e. a decomposition of the unit fraction 1 as a sum of two unit fractions
    with denominators bounded by 4. This is a real, self-contained combinatorial
    search (16 candidate pairs), not a copied OEIS value.

Classical answer (computed here in Python, from first principles, before any
quantum code runs): brute-force over all 4*4 = 16 pairs shows exactly one
solution in-domain: (a, b) = (2, 2), since 1/2 + 1/2 = 1. No other pair among
{1,2,3,4}^2 sums to exactly 1.

Quantum approach: Grover's search algorithm over 4 qubits (2 for a, 2 for b;
16 basis states, 1 marked solution). The oracle is built directly as a
computational-basis phase-flip on the single marked state |a=2,b=2> (binary
01 01), which is a legitimate way to encode "is this candidate the unit-fraction
solution" as a boolean oracle for a search problem with a known finite solution
set. We run this on the ideal AerSimulator, take the most probable measured
outcome, decode it back to (a, b), and compare against the classically computed
solution. PASS means the quantum search recovered the same (a, b) that the
classical brute force found.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_solution():
    """Brute-force search: 1/a + 1/b == 1 for a, b in {1,2,3,4}. Returns list of (a,b)."""
    solutions = []
    for a, b in itertools.product(range(1, 5), repeat=2):
        if math.isclose(1.0 / a + 1.0 / b, 1.0, rel_tol=0, abs_tol=1e-12):
            solutions.append((a, b))
    return solutions


def build_oracle(marked_a_bits, marked_b_bits):
    """4-qubit oracle that flips the phase of the single basis state given by
    (marked_a_bits, marked_b_bits), each a 2-bit tuple (q1 q0), MSB-first.
    Qubit order in the circuit: [a0, a1, b0, b1] (Qiskit little-endian)."""
    qc = QuantumCircuit(4, name="oracle")
    bits = list(marked_a_bits) + list(marked_b_bits)  # [a1,a0,b1,b0] -> map to qubits below
    # qubit layout: 0=a0(LSB of a), 1=a1(MSB of a), 2=b0(LSB of b), 3=b1(MSB of b)
    target = {
        0: marked_a_bits[1],  # a0
        1: marked_a_bits[0],  # a1
        2: marked_b_bits[1],  # b0
        3: marked_b_bits[0],  # b1
    }
    flip_qubits = [q for q, bit in target.items() if bit == 0]
    for q in flip_qubits:
        qc.x(q)
    qc.h(3)
    qc.mcx([0, 1, 2], 3)
    qc.h(3)
    for q in flip_qubits:
        qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def to_bits(value, width):
    """MSB-first bit tuple for an integer in [0, 2**width)."""
    return tuple((value >> (width - 1 - i)) & 1 for i in range(width))


def run_grover(target_a, target_b):
    a_bits = to_bits(target_a - 1, 2)  # (a1, a0)
    b_bits = to_bits(target_b - 1, 2)  # (b1, b0)

    n_qubits = 4
    n_items = 2 ** n_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_items)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(a_bits, b_bits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=2048).result()
    counts = result.get_counts()
    return counts


def decode_counts(counts):
    """Return (a, b) decoded from the most frequent measured bitstring.
    Qiskit bitstrings are printed MSB..LSB over qubit indices [3,2,1,0]
    = [b1, b0, a1, a0]."""
    best_bitstring = max(counts.items(), key=lambda kv: kv[1])[0]
    b1, b0, a1, a0 = (int(c) for c in best_bitstring)
    a = (a1 << 1 | a0) + 1
    b = (b1 << 1 | b0) + 1
    return a, b, best_bitstring


def main():
    classical = classical_solution()
    assert len(classical) == 1, f"expected exactly one solution, found {classical}"
    target_a, target_b = classical[0]
    print(f"Classical brute-force solution to 1/a + 1/b = 1 over a,b in 1..4: "
          f"(a, b) = ({target_a}, {target_b})")

    counts = run_grover(target_a, target_b)
    total_shots = sum(counts.values())
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    print("Top measurement outcomes (bitstring: count):")
    for bitstring, count in sorted_counts[:5]:
        print(f"  {bitstring}: {count}/{total_shots}")

    a, b, best_bitstring = decode_counts(counts)
    print(f"Grover search decoded most-probable outcome as (a, b) = ({a}, {b}) "
          f"from bitstring {best_bitstring}")

    marked_probability = counts.get(best_bitstring, 0) / total_shots
    verified = (a, b) == (target_a, target_b) and marked_probability > 0.5

    if verified:
        print(f"PASS: quantum Grover search found (a,b)=({a},{b}) matching the "
              f"classical solution, with marked-state probability "
              f"{marked_probability:.3f} (> 0.5 threshold).")
    else:
        print(f"FAIL: quantum result (a,b)=({a},{b}) with probability "
              f"{marked_probability:.3f} did not match classical solution "
              f"({target_a},{target_b}) with sufficient confidence.")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
