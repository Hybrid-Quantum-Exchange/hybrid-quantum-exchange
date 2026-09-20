"""
Erdos problem #704 -- quantum-testable-sequence lane.

Source metadata (data/problems.yaml, erdosproblems clone):
    number: "704"
    tags: ["graph theory", "geometry", "chromatic number"]
    oeis: ["N/A"]

LIMITATION (reported honestly, not worked around): problem #704 has no
associated OEIS sequence id in the source data ("N/A"). There is therefore
no OEIS-derived finite computable sequence property to hand to a quantum
circuit for this problem, as the task requires. Fabricating an OEIS id or
inventing a "property" not actually tied to problem #704's mathematical
content would violate the task's instructions, so this script does not do
that.

Best-effort fallback, honestly labelled as not a real test of problem #704:
To still produce a genuine, runnable quantum circuit in this file (rather
than nothing), we build a real Grover search circuit over a small classical
decision problem drawn from the problem's own tags ("graph theory",
"chromatic number"): 3-colorability of the smallest complete graph that is
NOT 3-colorable, K4 (4 vertices, needs exactly 4 colors), restricted to a
witness search for "is there a proper 3-coloring of the 2 non-adjacent
vertex pairs of K4 using 2 bits per vertex, satisfying all 6 edge
inequality constraints simultaneously" -- concretely: Grover search over
all 3-colorings (2 qubits/vertex, 4 vertices = 8 qubits) of K4, marking
proper colorings. K4 is not 3-colorable, so the classically-known correct
answer is: zero proper 3-colorings exist. We verify this both by brute
force classical enumeration and by running Grover's algorithm and checking
that the amplified "marked" subspace has (very close to) zero probability
mass, i.e. Grover with zero solutions leaves the state statistically
indistinguishable from uniform / does not concentrate on any single
"solution" bitstring -- consistent with there being none.

This is NOT a test of any OEIS sequence for problem #704 (none exists) and
is reported as such: verified_against_classical concerns only the K4
3-coloring brute-force fact, used purely to exercise a real circuit tied to
the problem's own tags. ran_ok / verified_against_classical are reported
accurately for this fallback, not as a stand-in success for the missing
OEIS-sequence task.
"""

import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ----------------------------------------------------------------------
# 1. Classical ground truth: proper 3-colorings of K4 (4 vertices, all
#    6 edges present). Each vertex gets a color in {0,1,2} but we encode
#    it in 2 bits (0..3), so color value 3 is an "invalid" encoding that
#    can never be part of a proper coloring either.
# ----------------------------------------------------------------------
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # K4, all pairs


def is_proper_coloring(colors):
    """colors: tuple of 4 ints, each in 0..3 (2-bit encoding)."""
    for (u, v) in EDGES:
        if colors[u] == colors[v]:
            return False
    return True


def classical_proper_colorings():
    """Brute force over all proper 3-colorings (colors restricted to {0,1,2},
    the '11'/3 encoding is deliberately excluded as invalid) of K4."""
    solutions = []
    for colors in itertools.product(range(3), repeat=4):
        if is_proper_coloring(colors):
            solutions.append(colors)
    return solutions


CLASSICAL_SOLUTIONS = classical_proper_colorings()
CLASSICAL_ANSWER_COUNT = len(CLASSICAL_SOLUTIONS)
# K4 needs chromatic number 4; with only 3 colors available (0,1,2; value 3
# unused/invalid) no proper coloring exists.
assert CLASSICAL_ANSWER_COUNT == 0, (
    f"expected 0 proper 3-colorings of K4, brute force found "
    f"{CLASSICAL_ANSWER_COUNT}"
)


# ----------------------------------------------------------------------
# 2. Quantum oracle: 8 qubits (2 per vertex, 4 vertices). Mark a basis
#    state iff it encodes colors all in {0,1,2} (i.e. no vertex uses the
#    invalid encoding '11') AND all 6 edges have differing colors. Since
#    we already know classically this marks nothing, Grover amplification
#    should fail to concentrate probability anywhere -- the state stays
#    close to uniform. We verify that numerically against the ideal
#    simulator.
# ----------------------------------------------------------------------
N_VERTICES = 4
BITS_PER_VERTEX = 2
N_QUBITS = N_VERTICES * BITS_PER_VERTEX  # 8


def vertex_qubits(v):
    return [v * BITS_PER_VERTEX, v * BITS_PER_VERTEX + 1]


def build_oracle_marks():
    """Return the set of 8-bit integers (little-endian, qubit0=LSB) that
    represent a valid-encoding (colors in 0..2) proper coloring of K4.
    We already know classically this set is empty; used to sanity check
    the oracle construction logic is doing the right classical thing."""
    marks = set()
    for colors in itertools.product(range(3), repeat=4):  # valid colors only
        if is_proper_coloring(colors):
            bits = 0
            for v, c in enumerate(colors):
                bits |= (c & 0b11) << (v * BITS_PER_VERTEX)
            marks.add(bits)
    return marks


ORACLE_MARKS = build_oracle_marks()
assert ORACLE_MARKS == set(), "oracle-mark derivation disagrees with brute force"


def run_grover_no_solution_check(shots=4096):
    """Run a small Grover circuit whose oracle marks nothing (since K4 has
    no proper 3-coloring), and confirm the resulting measurement
    distribution stays close to uniform across all 2^8 = 256 basis
    states -- i.e. no amplitude concentrates anywhere, consistent with
    zero marked solutions. This directly exercises the AerSimulator on a
    genuine (if here vacuous) Grover oracle+diffuser circuit."""
    n = N_QUBITS
    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    # Oracle: since ORACLE_MARKS is empty, the oracle is the identity
    # (phase-flips nothing). We still build it generically from the mark
    # set so the circuit genuinely implements "flip phase of marked
    # states" rather than hardcoding a no-op -- for an empty mark set this
    # correctly reduces to doing nothing, which we assert below.
    oracle = QuantumCircuit(n, name="Oracle")
    for mark in sorted(ORACLE_MARKS):
        bits = [(mark >> i) & 1 for i in range(n)]
        flip_qubits = [i for i, b in enumerate(bits) if b == 0]
        for i in flip_qubits:
            oracle.x(i)
        oracle.h(n - 1)
        oracle.mcx(list(range(n - 1)), n - 1)
        oracle.h(n - 1)
        for i in flip_qubits:
            oracle.x(i)
    assert oracle.size() == 0, "oracle should be empty (no marked states)"

    # Diffuser (standard Grover diffusion operator)
    diffuser = QuantumCircuit(n, name="Diffuser")
    diffuser.h(range(n))
    diffuser.x(range(n))
    diffuser.h(n - 1)
    diffuser.mcx(list(range(n - 1)), n - 1)
    diffuser.h(n - 1)
    diffuser.x(range(n))
    diffuser.h(range(n))

    # With zero marked states there is technically no well-defined number
    # of Grover iterations that helps; we still apply a fixed, small
    # number of iterations (as a real Grover circuit would for an unknown
    # solution count) and check the outcome stays statistically uniform.
    iterations = 3
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n), range(n))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    n_outcomes = 2 ** n
    expected_uniform = shots / n_outcomes
    max_count = max(counts.values())
    # Chi-square-ish sanity: no single outcome should dominate the way a
    # true Grover hit would (a real solution concentrates a large fraction
    # of shots on one basis state). We just check nothing exceeds a small
    # multiple of the uniform expectation.
    concentrated = max_count > 8 * expected_uniform
    return counts, concentrated, expected_uniform, max_count


def main():
    print("Erdos problem #704 -- quantum-testable-sequences lane")
    print("Tags:", ["graph theory", "geometry", "chromatic number"])
    print("OEIS id(s) in source data: N/A (no sequence exists for #704)")
    print()
    print("Classical check: brute-force proper 3-colorings of K4 (4 vertices,")
    print(f"all 6 edges) = {CLASSICAL_ANSWER_COUNT} (expected 0, K4 needs 4 colors)")
    print()

    counts, concentrated, expected_uniform, max_count = run_grover_no_solution_check()
    print(f"Grover run: 8 qubits, {sum(counts.values())} shots, "
          f"{len(counts)} distinct outcomes observed")
    print(f"Expected uniform count per outcome: {expected_uniform:.2f}, "
          f"observed max count: {max_count}")

    quantum_matches_classical = (CLASSICAL_ANSWER_COUNT == 0) and (not concentrated)

    print()
    if quantum_matches_classical:
        print("PASS")
    else:
        print("FAIL")

    print()
    print("NOTE: this PASS/FAIL concerns only the fallback K4 3-coloring")
    print("sanity check (tied to problem #704's tags), NOT an OEIS sequence")
    print("test, because problem #704 has no OEIS id in the source data.")

    return 0 if quantum_matches_classical else 1


if __name__ == "__main__":
    sys.exit(main())
