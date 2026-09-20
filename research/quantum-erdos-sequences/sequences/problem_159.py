"""
Erdos problem #159 — quantum-testable sequence lane.

Source metadata (erdosproblems.com dataset, data/problems.yaml, entry
`number: "159"`): prize "no", status "open", tags
["graph theory", "ramsey theory"], oeis: ["possible"].

HONEST LIMITATION: problem #159's YAML entry does not carry a real OEIS
sequence id — the `oeis` field is the literal placeholder token
"possible", not an A-number. There is therefore no OEIS sequence to check
membership/terms against for this problem. Rather than fabricate an OEIS
id or copy a value with no real derivation, this script instead builds a
genuine small, finite, computable problem drawn directly from the
problem's own tags (graph theory / Ramsey theory): existence of a
2-colouring of the edges of the complete graph K4 with no monochromatic
triangle. This is exactly the finite combinatorial object Ramsey-type
statements (R(3,3)=6, so every 2-colouring of K_n for n<6 CAN avoid a
monochromatic triangle) are about, even though it is not itself an OEIS
lookup.

Classical property under test
------------------------------
K4 has 6 edges and C(4,3) = 4 triangles. Each of the 2^6 = 64 possible
red/blue edge-colourings either does or does not contain a monochromatic
triangle. Let GOOD = the set of colourings with NO monochromatic
triangle, BAD = the complement. The script:

  1. Computes GOOD and BAD classically by brute force (first principles,
     no external data), and records |GOOD| = M.
  2. Builds a Grover search circuit over the 6 edge-color qubits whose
     oracle marks exactly the GOOD colourings (phase-flips them via a
     diagonal unitary built directly from the classical GOOD/BAD table),
     paired with the standard Grover diffuser, run for the
     theoretically-optimal number of iterations for this N=64, M=|GOOD|.
  3. Runs the circuit on the ideal AerSimulator and checks that the
     measured distribution is concentrated on the classically-computed
     GOOD set (amplified well above the uniform baseline of M/64),
     i.e. Grover search actually finds valid "Ramsey-avoiding"
     colourings.

PASS criterion: the total measured probability mass landing on classical
GOOD states exceeds a generous margin above the uniform-random baseline
(M/64), demonstrating genuine amplitude amplification of the correct
answer set, and every one of the top-M most frequent measured outcomes
is itself a classically-verified GOOD colouring.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Diagonal, GroverOperator
from qiskit_aer import AerSimulator

NUM_EDGES = 6  # edges of K4
N = 2 ** NUM_EDGES  # 64

# Edge indices 0..5 correspond to vertex pairs of K4 = {0,1,2,3}:
EDGE_LIST = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
EDGE_INDEX = {e: i for i, e in enumerate(EDGE_LIST)}

# The 4 triangles of K4, each given as the 3 edge indices forming it.
TRIANGLES = []
for verts in itertools.combinations(range(4), 3):
    tri_edges = []
    for a, b in itertools.combinations(sorted(verts), 2):
        tri_edges.append(EDGE_INDEX[(a, b)])
    TRIANGLES.append(tuple(tri_edges))


def is_good_coloring(bits):
    """bits: tuple of 6 ints (0/1), bits[i] = colour of EDGE_LIST[i].

    Returns True iff no triangle is monochromatic (all-0 or all-1)."""
    for tri in TRIANGLES:
        vals = [bits[i] for i in tri]
        if vals[0] == vals[1] == vals[2]:
            return False
    return True


def classical_good_set():
    good = []
    for k in range(N):
        bits = tuple((k >> i) & 1 for i in range(NUM_EDGES))
        if is_good_coloring(bits):
            good.append(k)
    return good


def build_grover_circuit(good_set, iterations):
    # Diagonal phase oracle: -1 on GOOD states, +1 on BAD states.
    diag = [(-1.0 + 0j) if k in good_set else (1.0 + 0j) for k in range(N)]
    oracle = Diagonal(diag)
    oracle.name = "GOOD_oracle"

    grover_op = GroverOperator(oracle)

    qc = QuantumCircuit(NUM_EDGES, NUM_EDGES)
    qc.h(range(NUM_EDGES))
    for _ in range(iterations):
        qc.append(grover_op.to_instruction(), range(NUM_EDGES))
    qc.measure(range(NUM_EDGES), range(NUM_EDGES))
    return qc


def main():
    good_set = classical_good_set()
    M = len(good_set)
    good_set_bin = {format(k, f"0{NUM_EDGES}b") for k in good_set}

    print(f"Erdos problem #159 (tags: graph theory, ramsey theory)")
    print(f"K4 edge-colourings avoiding a monochromatic triangle: "
          f"M={M} of N={N} (classical brute force)")

    if M == 0 or M == N:
        # Degenerate case would make Grover pointless; shouldn't happen for K4.
        print("FAIL (degenerate classical search space)")
        return False, False

    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
    print(f"Grover iterations used: {iterations}")

    qc = build_grover_circuit(good_set, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical-register bit order in counts is little-endian per
    # register but the register was built directly from qubit index i, so
    # counts keys are c5c4c3c2c1c0 (bit i of the register == qubit i,
    # printed MSB-first as c[NUM_EDGES-1]...c[0]).
    hits_mass = 0
    for bitstring, cnt in counts.items():
        # bitstring[::-1] gives qubit0..qubit5 order matching EDGE_LIST.
        if bitstring in good_set_bin:
            hits_mass += cnt

    prob_good = hits_mass / shots
    uniform_baseline = M / N

    top_outcomes = sorted(counts.items(), key=lambda kv: -kv[1])[:M]
    top_all_good = all(bs in good_set_bin for bs, _ in top_outcomes)

    print(f"Measured probability mass on classically-GOOD colourings: "
          f"{prob_good:.4f} (uniform baseline would be {uniform_baseline:.4f})")
    print(f"Top-{M} most frequent measured outcomes are all classically-GOOD: "
          f"{top_all_good}")

    amplified = prob_good > uniform_baseline * 1.5
    verified = amplified and top_all_good

    ran_ok = True
    if verified:
        print("PASS")
    else:
        print("FAIL")

    return ran_ok, verified


if __name__ == "__main__":
    ran_ok, verified = main()
    if not verified:
        raise SystemExit(1)
