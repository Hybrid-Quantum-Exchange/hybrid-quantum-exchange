"""
Erdos problem #846 — quantum-testable sequence attempt.

LIMITATION (read first): the source metadata for problem 846
(/home/user/manman4/erdosproblems/data/problems.yaml, entry "number: '846'")
gives no OEIS id — its `oeis` field is `["N/A"]` — and no problem statement
text, only bookkeeping fields:

    number: 846
    prize: no
    informal_status: disproved (2026-02-25)
    formal_status: Lean (2026-02-25)
    tags: [geometry]

With no OEIS sequence attached, there is no concrete "term of a sequence"
property to derive and check classically, so the task's actual request
(identify a small computable property of *the* OEIS sequence for #846 and
build a real Grover/QPE/QAE circuit around it) cannot be honestly satisfied
for this problem. Fabricating a sequence or an OEIS id would misrepresent
the source data, which the task explicitly forbids.

What this script does instead, as the best honest fallback: it builds a
REAL, self-contained Grover search circuit on AerSimulator for a small,
genuinely computed finite search problem in the same subject area as the
problem's tag ("geometry") — finding which residues x in Z_15 satisfy the
classic Pythagorean-triple-style geometric predicate

    x^2 + 3^2 == 5^2  (mod 15)   i.e.  x^2 == 16 == 1 (mod 15)

over the search space N = 16 (4 qubits). This is NOT problem 846's own
sequence (none exists in the source), and this script's PASS/FAIL result
says nothing about problem 846 itself — it only demonstrates that a real
oracle-based Grover circuit can be constructed and verified against an
independently, classically computed answer, which is the most honest
substitute available given the missing OEIS id.

Classical ground truth is computed first, from first principles, by brute
force over the full search space, and the quantum result is compared
against it below.

Reported accurately: ran_ok reflects whether this script executes without
error; verified_against_classical reflects whether the Grover circuit's
top measurement outcome(s) match the independently computed classical
solution set for this toy predicate — NOT whether problem 846's true
sequence was reproduced, since no such sequence is available to test.
"""

import math
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_solutions(n_bits: int, modulus: int) -> list[int]:
    """Brute-force, first-principles classical search.

    Predicate: x^2 mod modulus == 1 mod modulus, over x in [0, 2**n_bits).
    """
    target = 1 % modulus
    sols = []
    for x in range(2 ** n_bits):
        if x < modulus and (x * x) % modulus == target:
            sols.append(x)
    return sols


def build_oracle(n_bits: int, marked: list[int]) -> QuantumCircuit:
    """Phase-flip oracle marking each state in `marked` (multi-controlled Z)."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_bits}b")[::-1]  # little-endian per qubit index
        flip_qubits = [i for i, b in enumerate(bits) if b == "0"]
        for q in flip_qubits:
            qc.x(q)
        if n_bits == 1:
            qc.z(0)
        else:
            qc.h(n_bits - 1)
            qc.mcx(list(range(n_bits - 1)), n_bits - 1)
            qc.h(n_bits - 1)
        for q in flip_qubits:
            qc.x(q)
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


def run_grover(n_bits: int, marked: list[int], shots: int = 2048):
    N = 2 ** n_bits
    M = len(marked)
    if M == 0 or M == N:
        raise ValueError("Grover requires 0 < M < N marked items")

    # Optimal number of Grover iterations for this M, N (floor, not round,
    # since rounding up overshoots the peak success probability badly for
    # small instances such as this one).
    iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N / M)))

    oracle = build_oracle(n_bits, marked)
    diffuser = build_diffuser(n_bits)

    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))
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
    n_bits = 4          # search space N = 16
    modulus = 15         # x^2 == 1 (mod 15)

    marked = classical_solutions(n_bits, modulus)
    print(f"Classical brute-force solutions (x^2 == 1 mod {modulus}, "
          f"0<=x<{2**n_bits}): {marked}")

    counts, iterations = run_grover(n_bits, marked)
    print(f"Grover iterations used: {iterations}")

    # Sort measured bitstrings by frequency, take as many top outcomes as
    # there are marked solutions, and compare the *set* of top outcomes to
    # the classical solution set.
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_k = sorted_counts[: len(marked)]
    measured_top = sorted(int(bs, 2) for bs, _ in top_k)

    total_shots = sum(counts.values())
    marked_shots = sum(c for bs, c in counts.items() if int(bs, 2) in marked)
    marked_fraction = marked_shots / total_shots

    print(f"Measured top-{len(marked)} outcomes (as integers): {measured_top}")
    print(f"Fraction of shots landing on a marked (classically correct) "
          f"state: {marked_fraction:.3f}")

    verified = (
        set(measured_top) == set(marked)
        and marked_fraction > 0.8  # Grover should concentrate probability
    )

    if verified:
        print("PASS")
    else:
        print("FAIL")

    print(
        "\nNOTE: this PASS/FAIL verifies a real Grover circuit against an "
        "independently computed classical answer for a toy geometric "
        "predicate. It does NOT verify Erdos problem 846's own sequence, "
        "because problem 846 has no OEIS id in the source data (oeis: "
        "['N/A']) and therefore no sequence exists to test."
    )

    sys.exit(0 if verified else 1)


if __name__ == "__main__":
    main()
