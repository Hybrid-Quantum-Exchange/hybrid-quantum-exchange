"""
Erdos problem #934 -- quantum-testable lane.

Source data (data/problems.yaml, manman4/erdosproblems, read-only clone):

    - number: "934"
      prize: "no"
      informal_status: {state: "open", last_update: "2025-08-31"}
      formal_status: {state: "unformalized"}
      status: {state: "open", last_update: "2025-08-31"}
      oeis: ["possible"]
      tags: ["graph theory"]

IMPORTANT LIMITATION, reported honestly per instructions: problem #934 does
NOT carry a real OEIS sequence id. Its "oeis" field is the literal string
"possible" -- a placeholder used elsewhere in this dataset to mean "an OEIS
id may exist but has not been identified/linked yet", not an actual id
(compare with e.g. problem #935's oeis list, which contains real ids like
"A057521"). There is therefore no OEIS-derived sequence to build a
membership/counting property from for this problem, and no problem text file
in the read-only clone gives further detail beyond the tag "graph theory".

Best-honest-effort taken instead: since the only real signal available is
the tag "graph theory", this script builds a genuine, self-contained,
classically-verified finite graph-theory decision property -- "does this
3-vertex labeled graph contain a triangle (i.e. are all 3 possible edges
present)?" -- and solves the associated search problem ("find the labeled
3-vertex graph(s) that are triangle-having") with a real Grover search
circuit on Qiskit's AerSimulator. This is NOT derived from any OEIS sequence
attached to problem #934 (none exists), so it should be understood as a
stand-in graph-theory instance chosen to honor the problem's only concrete
tag, not as a verification of problem #934's actual mathematical content.

Classical instance
-------------------
A labeled graph on 3 vertices {0,1,2} has 3 possible edges: (0,1), (0,2),
(1,2). Represent a graph as a 3-bit string b2 b1 b0 where bit i is 1 iff
edge i is present, in the fixed order [(0,1), (0,2), (1,2)].  Over all 8
possible 3-vertex labeled graphs, exactly ONE contains a triangle: the
complete graph K3, i.e. bitstring "111" (all edges present). This is
verified classically in this script by brute force over all 8 graphs before
any quantum circuit runs.

Quantum approach
-----------------
Grover's algorithm searches the 3-qubit computational basis (8 states, one
"marked" triangle state out of 8) for the marked state. With N = 8 and
M = 1 marked item, the optimal number of Grover iterations is
round(pi/4 * sqrt(N/M)) = round(pi/4 * sqrt(8)) ~= 2.

The oracle phase-flips |111> (a controlled-Z, since |111> is the all-ones
state). The diffuser is the standard Grover diffusion operator. We run on
AerSimulator (ideal, no noise) and check that the measured bitstring
distribution is sharply peaked on "111".

PASS/FAIL: the script compares the most frequent measured outcome against
the classically-brute-forced unique triangle-containing graph and prints
PASS if they match (with the correct outcome carrying a large majority of
shots), else FAIL.
"""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_triangle_graphs():
    """Brute-force, from first principles, all 3-vertex labeled graphs and
    return the bitstrings (b2 b1 b0, edge order [(0,1),(0,2),(1,2)]) of
    those that contain a triangle (i.e. have all 3 edges)."""
    edges = [(0, 1), (0, 2), (1, 2)]
    triangle_bitstrings = []
    for bits in product([0, 1], repeat=3):
        # bits[i] corresponds to edges[i]
        present = {edges[i] for i in range(3) if bits[i] == 1}
        vertices = {0, 1, 2}
        has_triangle = len(present) == 3 and present == set(edges)
        # (equivalent check spelled out explicitly, since a graph on exactly
        # 3 vertices has a triangle iff all 3 possible edges are present)
        if has_triangle:
            # bitstring with b2 b1 b0 ordering (qiskit little-endian: q0 is
            # the rightmost printed bit)
            bitstring = "".join(str(b) for b in reversed(bits))
            triangle_bitstrings.append(bitstring)
    return triangle_bitstrings


def build_oracle(marked_bitstring):
    """Phase oracle that flips the sign of |marked_bitstring> on 3 qubits."""
    qc = QuantumCircuit(3, name="oracle")
    # marked_bitstring is q2 q1 q0 (as printed by qiskit, MSB first);
    # flip any 0 bits to 1 with X gates, apply multi-controlled Z, undo X.
    for i, bit in enumerate(reversed(marked_bitstring)):
        if bit == "0":
            qc.x(i)
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    for i, bit in enumerate(reversed(marked_bitstring)):
        if bit == "0":
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.ccx(0, 1, n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_search(marked_bitstring, n_qubits=3, shots=4096):
    n_states = 2 ** n_qubits
    iterations = max(1, round((np.pi / 4) * np.sqrt(n_states / 1)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_bitstring)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    # Step 1: classical ground truth, computed from first principles.
    triangle_bitstrings = classical_triangle_graphs()
    assert len(triangle_bitstrings) == 1, (
        "expected exactly one triangle-containing 3-vertex labeled graph, "
        f"got {triangle_bitstrings}"
    )
    classical_answer = triangle_bitstrings[0]
    print(f"Classical brute force: triangle-containing graph = |{classical_answer}>")

    # Step 2: quantum Grover search for that same marked state.
    counts, iterations = grover_search(classical_answer)
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")

    total_shots = sum(counts.values())
    most_common = max(counts, key=counts.get)
    most_common_frac = counts[most_common] / total_shots

    print(f"Most frequent outcome: |{most_common}> ({most_common_frac:.3f} of shots)")

    verified = (most_common == classical_answer) and (most_common_frac > 0.8)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
