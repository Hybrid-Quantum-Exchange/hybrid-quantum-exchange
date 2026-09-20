"""Quantum-testable sequence entry: Erdos problem #128.

Source metadata (from erdosproblems.com data, problems.yaml entry
`number: "128"`, tags: ["graph theory"], prize: "$250"):

    oeis: ["N/A"]

LIMITATION, stated honestly up front: problem #128 has NO OEIS sequence
attached (oeis is the literal string "N/A" in the source data, not a real
id), and the problems.yaml row carries no statement text either -- only
metadata (prize, status, tags). There is therefore no OEIS sequence to
build a quantum-testable membership/term property from, as the task
brief asks for. Rather than fabricate an OEIS id or invent a property
falsely attributed to problem #128's actual (unknown, to this script)
statement, this script falls back to the one honest thing the metadata
does tell us: the problem's `tags` field is "graph theory". So this is a
best-effort substitute, clearly not a verification of problem #128
itself: a small, finite, genuinely computable graph-theory decision
property -- proper vertex 2-colorability of a path graph P3 (3 vertices,
2 edges: v0-v1, v1-v2) -- solved with a real Grover search circuit on
AerSimulator, and checked against a brute-force classical enumeration
computed in this script from first principles.

Classical property under test
------------------------------
Let v0, v1, v2 in {0, 1} be a colour assignment to the 3 vertices of the
path graph P3 (edges v0-v1 and v1-v2). A colouring is PROPER iff
v0 != v1 AND v1 != v2 (adjacent vertices get different colours).

Brute-force classical enumeration over all 2^3 = 8 assignments gives
exactly 2 proper colourings:
    (v0, v1, v2) = (0, 1, 0)
    (v0, v1, v2) = (1, 0, 1)

This script builds a Grover search circuit over 3 qubits (search space
N = 8) whose oracle marks exactly those 2 proper-colouring states, runs
the expected ~1 Grover iteration (optimal for N=8, M=2), and checks that
the two classically-correct bitstrings dominate the measured output
distribution.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles (brute force).
# ---------------------------------------------------------------------------

def is_proper_p3_colouring(v0: int, v1: int, v2: int) -> bool:
    """P3 has edges v0-v1 and v1-v2; proper iff both edges are bichromatic."""
    return (v0 != v1) and (v1 != v2)


def classical_solutions():
    sols = []
    for v0, v1, v2 in itertools.product((0, 1), repeat=3):
        if is_proper_p3_colouring(v0, v1, v2):
            sols.append((v0, v1, v2))
    return sols


CLASSICAL_SOLUTIONS = classical_solutions()
assert CLASSICAL_SOLUTIONS == [(0, 1, 0), (1, 0, 1)], CLASSICAL_SOLUTIONS

N = 8          # search space size, 2^3
M = len(CLASSICAL_SOLUTIONS)  # number of marked (proper-colouring) states


# Qiskit qubit order is little-endian in bitstrings (q0 is the rightmost
# character). We map q0 -> v0, q1 -> v1, q2 -> v2, so a bitstring
# "v2 v1 v0" read left-to-right corresponds to Qiskit's classical register
# output string directly.
def bitstring_for(v0: int, v1: int, v2: int) -> str:
    return f"{v2}{v1}{v0}"


MARKED_BITSTRINGS = {bitstring_for(*s) for s in CLASSICAL_SOLUTIONS}


# ---------------------------------------------------------------------------
# 2. Grover oracle: phase-flip states where v0 != v1 AND v1 != v2.
# ---------------------------------------------------------------------------

def build_oracle() -> QuantumCircuit:
    # qubits 0,1,2 = v0,v1,v2 ; qubits 3,4 = ancillas holding XORs
    qc = QuantumCircuit(5, name="oracle")

    # a1 (qubit 3) = v0 XOR v1
    qc.cx(0, 3)
    qc.cx(1, 3)
    # a2 (qubit 4) = v1 XOR v2
    qc.cx(1, 4)
    qc.cx(2, 4)

    # Phase flip exactly when a1 == 1 and a2 == 1 (both edges bichromatic).
    qc.cz(3, 4)

    # Uncompute ancillas.
    qc.cx(1, 4)
    qc.cx(2, 4)
    qc.cx(0, 3)
    qc.cx(1, 3)

    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(iterations: int) -> QuantumCircuit:
    n_search = 3
    n_total = 5  # 3 search qubits + 2 ancilla
    qc = QuantumCircuit(n_total, n_search)

    qc.h(range(n_search))

    oracle = build_oracle()
    diffuser = build_diffuser(n_search)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(5))
        qc.append(diffuser.to_instruction(), range(n_search))

    qc.measure(range(n_search), range(n_search))
    return qc


# ---------------------------------------------------------------------------
# 3. Run on AerSimulator and compare to the classical ground truth.
# ---------------------------------------------------------------------------

def main():
    theta = math.asin(math.sqrt(M / N))
    optimal_iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    circuit = build_grover_circuit(optimal_iterations)

    sim = AerSimulator()
    compiled = transpile(circuit, sim)
    shots = 4096
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()

    marked_shots = sum(c for bs, c in counts.items() if bs in MARKED_BITSTRINGS)
    marked_fraction = marked_shots / shots

    top_bitstring = max(counts, key=counts.get)

    print("Erdos problem #128 -- quantum-testable sequence entry")
    print("Note: problem #128 has no OEIS id (oeis: ['N/A']); this script")
    print("uses a substitute finite graph-theory property (proper 2-colouring")
    print("of path graph P3) derived from the problem's 'graph theory' tag,")
    print("NOT a verification of problem #128's actual (unavailable) statement.")
    print()
    print(f"Classical proper colourings of P3 (brute force): {CLASSICAL_SOLUTIONS}")
    print(f"Search space N={N}, marked states M={M}, Grover iterations={optimal_iterations}")
    print(f"Measured counts: {counts}")
    print(f"Most frequent measured bitstring: {top_bitstring} "
          f"(marked={top_bitstring in MARKED_BITSTRINGS})")
    print(f"Fraction of shots landing on a classically-correct marked state: "
          f"{marked_fraction:.4f}")

    # Success criteria: the two classically-correct states should dominate
    # (Grover amplifies them well above the uniform 2/8 = 0.25 baseline).
    verified = (
        marked_fraction > 0.7
        and top_bitstring in MARKED_BITSTRINGS
        and set(MARKED_BITSTRINGS).issubset(set(counts.keys()))
    )

    print()
    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
