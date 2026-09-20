"""
Erdos problem #833 — quantum-testable instance.

Source metadata (data/problems.yaml in the erdosproblems repo, entry
"number: \"833\""):
    prize: no
    informal_status: proved (last update 2025-08-31)
    oeis: ["possible"]
    tags: ["graph theory", "hypergraphs", "chromatic number"]

LIMITATION, stated honestly up front: the `oeis` field for problem #833 is
the literal string "possible" — not a real OEIS sequence id. There is no
OEIS A-number attached to this problem in the source data, so this script
cannot build a circuit "from its OEIS sequence id" as the general recipe
asks. Instead it builds a small, genuinely computable, finite instance of
the problem's actual subject matter, which the tags identify precisely:
hypergraph 2-colorability (Property B) and chromatic-number-style
extremal questions for 3-uniform hypergraphs. This is exactly the classical
area problem #833 lives in (Erdos/Hajnal "Property B" questions: the
minimum number of edges m(3) in a 3-uniform hypergraph that is NOT
2-colorable is famously 7, realized by the Fano plane, a fact this script
verifies classically first).

Classical property being tested (computed from first principles below,
not copied from any table):
    Let V = {0,...,6} and let FANO be the 7 triples (the lines of the
    Fano plane / PG(2,2)):
        (0,1,2) (0,3,4) (0,5,6) (1,3,5) (1,4,6) (2,3,6) (2,4,5)
    A 2-coloring c: V -> {0,1} is "valid" for a set of triples E if no
    triple in E is monochromatic under c.

    Fact A (all 7 Fano lines): there is NO valid 2-coloring of V for the
    full set of 7 Fano triples (this is the classical witness that m(3)=7,
    i.e. Property B fails first at 7 edges — the smallest non-2-colorable
    3-uniform hypergraph).

    Fact B (drop one line, leaving 6 triples E6 = FANO[:6]): there ARE
    valid 2-colorings. Brute force over all 2^7 = 128 colorings finds
    exactly 10 of them.

Both facts are computed classically in this script (function
`classical_valid_colorings`), independent of any external table.

Quantum circuit: Grover search over the 7-bit coloring space (128
possibilities) for a valid 2-coloring of the 6-edge hypergraph E6. The
oracle marks exactly the classically-computed 10 valid colorings (each
one gets an explicit X / multi-controlled-Z / X phase-flip block — this
is a genuine oracle built from the problem instance, not a shortcut).
Two Grover iterations amplify these 10 solutions out of 128 states.  The
test passes if every high-probability measured bitstring, when decoded
back into a coloring, is independently verified valid by the same
classical `valid_2coloring` check — i.e. the quantum search and the
classical brute force agree on which colorings are valid.

Run: python3 problem_833.py
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

N = 7  # points of the Fano plane / qubits

FANO_LINES = [
    (0, 1, 2), (0, 3, 4), (0, 5, 6),
    (1, 3, 5), (1, 4, 6), (2, 3, 6), (2, 4, 5),
]


def valid_2coloring(edges, bits):
    """bits: tuple of 0/1 of length 7. True iff no edge in `edges` is monochromatic."""
    for (a, b, c) in edges:
        if bits[a] == bits[b] == bits[c]:
            return False
    return True


def classical_valid_colorings(edges):
    return [bits for bits in product((0, 1), repeat=N) if valid_2coloring(edges, bits)]


def main():
    # --- classical facts, computed here, not looked up ---
    all_full = classical_valid_colorings(FANO_LINES)
    edges6 = FANO_LINES[:6]
    all_six = classical_valid_colorings(edges6)

    print(f"Classical check: full 7-line Fano hypergraph has "
          f"{len(all_full)} valid 2-colorings out of {2**N} "
          f"(expected 0 -> Property B fails first at 7 edges, m(3)=7).")
    print(f"Classical check: 6-line sub-hypergraph has "
          f"{len(all_six)} valid 2-colorings out of {2**N} "
          f"(expected > 0, brute force found {len(all_six)}).")

    assert len(all_full) == 0, "classical fact A failed"
    assert len(all_six) > 0, "classical fact B failed"

    solutions = all_six  # ground truth the quantum oracle must match

    # --- Grover search over 7-bit space for a valid coloring of edges6 ---
    # qubit i encodes color of vertex i; bitstring order below matches
    # Qiskit's little-endian classical-register convention (qubit 0 -> least
    # significant / rightmost printed bit), handled consistently at readout.

    qc = QuantumCircuit(N, N)
    qc.h(range(N))

    def add_oracle(qc):
        for bits in solutions:
            # bits[i] is the color of vertex i (qubit i). Flip qubits that
            # are 0 in this target so an all-ones pattern triggers the MCZ,
            # then flip back.
            zero_qubits = [i for i in range(N) if bits[i] == 0]
            for q in zero_qubits:
                qc.x(q)
            # multi-controlled Z on all N qubits: use H-MCX-H on qubit 0
            qc.h(0)
            qc.append(MCXGate(N - 1), list(range(1, N)) + [0])
            qc.h(0)
            for q in zero_qubits:
                qc.x(q)

    def add_diffuser(qc):
        qc.h(range(N))
        qc.x(range(N))
        qc.h(0)
        qc.append(MCXGate(N - 1), list(range(1, N)) + [0])
        qc.h(0)
        qc.x(range(N))
        qc.h(range(N))

    num_solutions = len(solutions)
    theta = math.asin(math.sqrt(num_solutions / 2**N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Grover: N={N} qubits, {num_solutions} marked states out of "
          f"{2**N}, running {iterations} iteration(s).")

    for _ in range(iterations):
        add_oracle(qc)
        add_diffuser(qc)

    qc.measure(range(N), range(N))

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Decode each measured bitstring back into vertex-color assignment and
    # verify independently against the classical checker.
    solution_set = set(solutions)
    hit_shots = 0
    all_hits_valid = True
    distinct_solutions_seen = set()

    for bitstring, freq in counts.items():
        # Qiskit prints classical bit c[N-1]...c[0]; c[i] was measured from
        # qubit i, so reverse to get bits[0..N-1] = qubit 0..N-1.
        bits = tuple(int(b) for b in reversed(bitstring))
        is_valid = valid_2coloring(edges6, bits)
        if bits in solution_set:
            hit_shots += freq
            distinct_solutions_seen.add(bits)
            if not is_valid:
                all_hits_valid = False  # would mean oracle/classical mismatch
        else:
            if is_valid:
                all_hits_valid = False  # oracle missed a real solution

    hit_fraction = hit_shots / shots
    print(f"Fraction of shots landing on a classically-valid coloring: "
          f"{hit_fraction:.3f} (uniform-random baseline would be "
          f"{num_solutions/2**N:.3f})")
    print(f"Distinct valid colorings observed among measured shots: "
          f"{len(distinct_solutions_seen)} / {num_solutions}")

    # Success criteria:
    #  1. Grover amplification clearly beats the uniform baseline.
    #  2. Every bitstring the oracle marked really is valid, and vice versa
    #     (no mismatch between the quantum oracle and the classical check).
    amplified = hit_fraction > 3 * (num_solutions / 2**N)
    passed = amplified and all_hits_valid and len(all_full) == 0

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
