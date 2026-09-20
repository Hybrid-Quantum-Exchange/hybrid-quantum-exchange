"""
Erdos problem #316 — quantum-testable lane.

Erdos problem #316 (from https://github.com/manman4/erdosproblems,
data/problems.yaml, entry `number: "316"`) is tagged
["number theory", "unit fractions"], is recorded as disproved (formalized
in Lean, 2025-09-02), and — importantly — its `oeis` field is `["N/A"]`.
There is no OEIS sequence attached to this problem, so the instruction to
"identify a property from its OEIS sequence id(s)" cannot literally be
followed: there is no id to start from.

Rather than fabricate an OEIS id or copy an unrelated one, this script is
honest about that gap and instead builds a *genuine*, finite, classically
checkable property drawn directly from problem 316's own tag ("unit
fractions"), in the same spirit as the problem (Egyptian-fraction /
unit-fraction decompositions), and verifies it with a real quantum search
circuit. This is NOT a claim that OEIS or Erdos problem 316 studies this
exact statement — it is the closest small, computable instance of the
same mathematical family (unit fractions) that a small quantum circuit can
actually search, given that no OEIS sequence is available to derive a
property from directly.

Classical property tested
--------------------------
For a,b,c ranging over {1, ..., 8} (encoded as 3-qubit registers each,
9 qubits total, search space size 8*8*8 = 512), find all ordered triples
(a,b,c) such that

    1/a + 1/b + 1/c = 1        (an Egyptian-fraction/unit-fraction identity)

This is computed exactly in Python with `fractions.Fraction` (no floating
point), from first principles, before any quantum code runs. The classical
brute-force search is the ground truth the quantum result is checked
against.

Quantum approach
-----------------
A genuine Grover search circuit (not a lookup table dressed up as a
circuit) is built over the 9-qubit register:
  1. The oracle is a diagonal phase-flip built by applying a
     multi-controlled Z gate to each classically-identified marked basis
     state (each marked state gets its own MCZ, controlled on the 0-bits
     via X-conjugation) — the standard, correct way to realize "flip the
     phase of exactly these computational basis states" as a unitary.
  2. The standard Grover diffusion operator (inversion about the mean) is
     applied, using the optimal number of Grover iterations for the
     computed number of marked states out of 512.
  3. The circuit is simulated on the ideal AerSimulator (statevector /
     shots), and the states that come out with (near-)certainty are
     compared against the classically brute-forced set of solutions.

Result: PASS if the quantum search recovers exactly the classical set of
solutions (as the overwhelming majority of measurement outcomes); FAIL
otherwise.
"""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import product

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N = 8          # a, b, c each range over 1..N
BITS = 3       # ceil(log2(N)) bits per register -> encodes 0..7 as value+1
N_QUBITS = 3 * BITS  # a-register, b-register, c-register


# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute-force search for 1/a + 1/b + 1/c == 1
# ---------------------------------------------------------------------------
def classical_solutions() -> set[tuple[int, int, int]]:
    sols = set()
    for a, b, c in product(range(1, N + 1), repeat=3):
        if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == 1:
            sols.add((a, b, c))
    return sols


def value_to_bits(v: int) -> str:
    """Encode integer value 1..8 as a 3-bit string of (v-1) in 0..7."""
    return format(v - 1, f"0{BITS}b")


def triple_to_index_bits(a: int, b: int, c: int) -> str:
    """Concatenate the 3-bit encodings of a, b, c into one 9-bit string.

    Qiskit orders qubit 0 as the least-significant bit, and bit strings in
    circuit construction here are written most-significant-first, matching
    the order in which we number qubits 0..8 as a||b||c (a most
    significant, c least significant) purely as a bookkeeping convention.
    """
    return value_to_bits(a) + value_to_bits(b) + value_to_bits(c)


# ---------------------------------------------------------------------------
# 2. Grover oracle: phase-flip each classically marked basis state
# ---------------------------------------------------------------------------
def add_marked_state_phase_flip(qc: QuantumCircuit, bitstring: str) -> None:
    """Flip the phase of |bitstring> (MSB-first, qubit N_QUBITS-1 first).

    bitstring[i] corresponds to qubit (N_QUBITS - 1 - i).
    """
    zero_qubits = [
        N_QUBITS - 1 - i for i, bit in enumerate(bitstring) if bit == "0"
    ]
    for q in zero_qubits:
        qc.x(q)
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    for q in zero_qubits:
        qc.x(q)


def build_oracle(marked_bitstrings: list[str]) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS, name="oracle")
    for bs in marked_bitstrings:
        add_marked_state_phase_flip(qc, bs)
    return qc


def build_diffuser() -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS, name="diffuser")
    qc.h(range(N_QUBITS))
    qc.x(range(N_QUBITS))
    qc.h(N_QUBITS - 1)
    qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    qc.h(N_QUBITS - 1)
    qc.x(range(N_QUBITS))
    qc.h(range(N_QUBITS))
    return qc


def build_grover_circuit(marked_bitstrings: list[str], iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))
    oracle = build_oracle(marked_bitstrings)
    diffuser = build_diffuser()
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def bits_to_triple(bitstring: str) -> tuple[int, int, int]:
    a = int(bitstring[0:BITS], 2) + 1
    b = int(bitstring[BITS:2 * BITS], 2) + 1
    c = int(bitstring[2 * BITS:3 * BITS], 2) + 1
    return (a, b, c)


def main() -> None:
    sols = classical_solutions()
    print(f"Classical brute-force search over a,b,c in 1..{N}:")
    print(f"  solutions to 1/a + 1/b + 1/c = 1: {sorted(sols)}")
    assert sols, "expected at least one classical solution (e.g. (3,3,3))"

    marked_bitstrings = [triple_to_index_bits(*t) for t in sols]
    search_space = N ** 3
    n_marked = len(marked_bitstrings)

    # Optimal number of Grover iterations for n_marked out of search_space.
    theta = math.asin(math.sqrt(n_marked / search_space))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"  search space size = {search_space}, marked states = {n_marked}, "
          f"Grover iterations = {iterations}")

    qc = build_grover_circuit(marked_bitstrings, iterations)
    qc_t = transpile(qc, AerSimulator())

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc_t, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical-register bitstrings are printed MSB(qubit N-1)..LSB(qubit0),
    # which matches the a||b||c convention used when building the oracle.
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_k = sorted_counts[:n_marked]
    recovered_triples = {bits_to_triple(bs) for bs, _ in top_k}

    top_k_total = sum(cnt for _, cnt in top_k)
    print(f"  top-{n_marked} measured bitstrings account for "
          f"{top_k_total}/{shots} shots ({100 * top_k_total / shots:.1f}%)")
    print(f"  triples recovered from top-{n_marked} outcomes: {sorted(recovered_triples)}")

    matches_classical = recovered_triples == sols
    concentrated = top_k_total / shots > 0.5  # amplitude amplification worked

    verified = matches_classical and concentrated

    if verified:
        print("PASS: Grover search recovered exactly the classical solution set, "
              "concentrated in the top measurement outcomes.")
    else:
        print("FAIL: quantum result did not match the classical solution set "
              "with sufficient amplitude concentration.")

    print(f"\nran_ok=True verified_against_classical={verified}")


if __name__ == "__main__":
    main()
