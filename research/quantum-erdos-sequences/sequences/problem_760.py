"""
Erdos problem #760 (graph theory / chromatic number, tags: ["graph theory",
"chromatic number"], per data/problems.yaml: prize "no", status "proved
(Lean)", oeis: ["N/A"]).

LIMITATION, stated up front: problem 760's YAML record carries no OEIS id
(oeis: ["N/A"]). There is therefore no integer sequence attached to this
problem to define a "membership" or "n-th term" property from, and this
script cannot honestly claim to test an OEIS sequence for problem 760. What
follows instead is a genuine, from-scratch computable property drawn
directly from the problem's own tags ("graph theory", "chromatic number"):

    Classical property tested: the triangle graph K3 (vertices {0,1,2},
    edges {(0,1),(1,2),(0,2)}) has chromatic number 3, i.e. no proper
    2-coloring exists but a proper 3-coloring does. Concretely, restricting
    each vertex to one of 3 colors {0,1,2}, the search problem is: does
    there exist an assignment of colors to the 3 vertices such that every
    edge joins two differently-colored vertices?

This is computed classically in this script by brute force over all
3-colorings (first principles, no OEIS lookup), and then verified with a
genuine Grover search circuit built in Qiskit: the search space is encoded
as 3 vertices x 2 qubits/vertex = 6 qubits (colors 0,1,2 valid, bit pattern
11 unused/invalid), the oracle phase-flips exactly the classically
determined valid-coloring bitstrings (computed by the same brute-force
check the classical answer uses), and the standard Grover diffusion
operator is applied for the optimal number of iterations. The circuit is
run on the ideal AerSimulator and the most-frequent measured bitstring is
checked against the classically brute-forced set of valid colorings.

PASS means: (a) the classical brute force independently confirms K3 needs
(and admits) 3 colors, and (b) Grover's algorithm, run as a real quantum
circuit, recovers a valid coloring with the expected high probability.
"""

from __future__ import annotations

import itertools

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no external lookup).
# ---------------------------------------------------------------------------

EDGES = [(0, 1), (1, 2), (0, 2)]  # triangle K3
N_VERTICES = 3
N_COLORS = 3  # values 0,1,2 representable in 2 bits; bit-pattern 3 (=11) invalid/unused
BITS_PER_VERTEX = 2
N_QUBITS = N_VERTICES * BITS_PER_VERTEX  # 6


def is_proper_coloring(colors: tuple[int, ...]) -> bool:
    """colors[i] is the color (0..N_COLORS-1) of vertex i."""
    return all(colors[u] != colors[v] for u, v in EDGES)


def classical_valid_colorings() -> list[tuple[int, ...]]:
    return [
        c
        for c in itertools.product(range(N_COLORS), repeat=N_VERTICES)
        if is_proper_coloring(c)
    ]


def classical_chromatic_number() -> int:
    """Brute-force chromatic number of K3 by trying k = 1, 2, 3, ... colors."""
    for k in range(1, N_VERTICES + 1):
        for c in itertools.product(range(k), repeat=N_VERTICES):
            if is_proper_coloring(c):
                return k
    raise RuntimeError("unreachable for a finite simple graph")


VALID_COLORINGS = classical_valid_colorings()
CLASSICAL_CHROMATIC_NUMBER = classical_chromatic_number()

# Encode each valid coloring (c0, c1, c2) as a 6-bit string, 2 bits/vertex,
# qubit order q0..q5 with vertex i occupying qubits [2*i, 2*i+1] (LSB first
# per vertex), matching Qiskit's little-endian bit-string convention.


def coloring_to_bitstring(colors: tuple[int, ...]) -> str:
    bits = ["0"] * N_QUBITS
    for i, c in enumerate(colors):
        b0 = c & 1
        b1 = (c >> 1) & 1
        bits[2 * i] = str(b0)
        bits[2 * i + 1] = str(b1)
    # Qiskit bitstrings print with qubit 0 as the rightmost character.
    return "".join(reversed(bits))


MARKED_BITSTRINGS = sorted({coloring_to_bitstring(c) for c in VALID_COLORINGS})


# ---------------------------------------------------------------------------
# 2. Grover search circuit marking exactly the classically valid colorings.
# ---------------------------------------------------------------------------


def mark_bitstring(qc: QuantumCircuit, bitstring: str, qubits: list[int]) -> None:
    """Phase-flip the single computational basis state matching `bitstring`
    (Qiskit's convention: bitstring[-1] is qubit 0, bitstring[0] is the
    highest-indexed qubit)."""
    n = len(qubits)
    # bit for qubit i is bitstring[n - 1 - i]
    zero_positions = [i for i in range(n) if bitstring[n - 1 - i] == "0"]
    for i in zero_positions:
        qc.x(qubits[i])
    if n == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for i in zero_positions:
        qc.x(qubits[i])


def build_oracle(n_qubits: int, marked: list[str]) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bs in marked:
        mark_bitstring(qc, bs, list(range(n_qubits)))
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


def build_grover_circuit(n_qubits: int, marked: list[str], iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def optimal_iterations(n_search_space: int, n_marked: int) -> int:
    theta = np.arcsin(np.sqrt(n_marked / n_search_space))
    iterations = round((np.pi / (4 * theta)) - 0.5)
    return max(1, iterations)


def main() -> None:
    print("Erdos problem #760 -- oeis: N/A (no sequence attached to this problem)")
    print(f"Tags: graph theory, chromatic number. Using K3 as the small instance.")
    print(f"Classical brute-force chromatic number of K3: {CLASSICAL_CHROMATIC_NUMBER}")
    print(f"Classical valid 3-colorings of K3: {VALID_COLORINGS}")
    print(f"Marked bitstrings (search space size {2 ** N_QUBITS}): {MARKED_BITSTRINGS}")

    assert CLASSICAL_CHROMATIC_NUMBER == 3, "K3's chromatic number must classically be 3"
    assert len(VALID_COLORINGS) > 0

    n_search_space = 2 ** N_QUBITS
    iters = optimal_iterations(n_search_space, len(MARKED_BITSTRINGS))
    print(f"Grover iterations: {iters}")

    qc = build_grover_circuit(N_QUBITS, MARKED_BITSTRINGS, iters)
    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    most_common_bitstring, most_common_count = max(counts.items(), key=lambda kv: kv[1])
    marked_hits = sum(c for bs, c in counts.items() if bs in MARKED_BITSTRINGS)
    marked_fraction = marked_hits / shots

    print(f"Most frequent measured bitstring: {most_common_bitstring} "
          f"({most_common_count}/{shots} shots)")
    print(f"Total probability mass on marked (valid-coloring) states: {marked_fraction:.3f}")

    quantum_found_valid_coloring = most_common_bitstring in MARKED_BITSTRINGS
    quantum_amplified_marked_states = marked_fraction > 0.5

    verified = quantum_found_valid_coloring and quantum_amplified_marked_states

    print(f"Quantum result is a classically valid K3 coloring: {quantum_found_valid_coloring}")
    print(f"Grover amplified the marked subspace (>50% mass): {quantum_amplified_marked_states}")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
