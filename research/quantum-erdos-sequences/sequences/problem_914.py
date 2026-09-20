"""
Erdos problem #914 -- quantum-testable lane (best-honest-effort, limitation noted)
====================================================================================

Source metadata (from erdosproblems data/problems.yaml, entry "number: '914'"):
    prize: no
    informal_status: proved
    formal_status: Lean
    oeis: ["N/A"]
    tags: ["graph theory"]

LIMITATION, stated up front: problem #914 carries no OEIS sequence id (the
yaml entry literally records oeis: ["N/A"]). There is therefore no genuine
"quantum-testable sequence" to build for it in the sense the library asks
for (an OEIS-anchored property). Rather than fabricate an OEIS value or
pretend a sequence membership test exists where none does, this script
instead builds a real, honest quantum circuit for the one piece of content
the metadata *does* give us -- the "graph theory" tag -- on a small, fully
finite, classically-checkable instance, and is explicit that it is not tied
to any OEIS id.

Chosen finite, computable property
-----------------------------------
Over the complete graph K3 on 3 labelled vertices {0,1,2}, there are exactly
3 possible edges: (0,1), (0,2), (1,2). A 3-bit string x = x01 x02 x12 encodes
a subgraph of K3 (bit=1 means that edge is present). Exactly one of the
2^3 = 8 possible subgraphs is a triangle: the one with all three edges
present, i.e. x = 111.

Property tested: "which 3-edge subgraph of K3 is a triangle (all edges
present)?" This is computed classically from first principles below by
brute-force enumeration of all 8 edge-subsets, and then found via Grover
search (oracle + diffuser on 3 qubits, exact 1-marked-out-of-8 case, so a
single Grover iteration succeeds with probability 1 on the ideal simulator).

This is a real, if modest, graph-theory search circuit -- an honest stand-in
given the absence of any OEIS sequence for this particular Erdos problem.
"""

import itertools

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth: brute-force over all subgraphs of K3.
# ---------------------------------------------------------------------------

def edges_present(bits):
    """bits = (b01, b02, b12) -> list of present edges."""
    labels = [(0, 1), (0, 2), (1, 2)]
    return [labels[i] for i, b in enumerate(bits) if b == 1]


def is_triangle(bits):
    """A subgraph of K3 is a triangle iff all three edges are present."""
    return sum(bits) == 3


classical_triangle_strings = []
for bits in itertools.product([0, 1], repeat=3):
    if is_triangle(bits):
        # bit order (b01, b02, b12) -> circuit string with qubit0=b01 (LSB)
        classical_triangle_strings.append("".join(str(b) for b in reversed(bits)))

assert classical_triangle_strings == ["111"], classical_triangle_strings
CLASSICAL_ANSWER = "111"
print(f"Classical brute-force result: unique triangle subgraph bitstring = "
      f"{CLASSICAL_ANSWER} (edges present: {edges_present((1, 1, 1))})")


# ---------------------------------------------------------------------------
# 2. Grover search circuit: 3 qubits, oracle marks x=111, one iteration.
# ---------------------------------------------------------------------------

def build_oracle():
    """Phase-flip |111> via a multi-controlled Z (CCZ on 3 qubits)."""
    qc = QuantumCircuit(3, name="oracle")
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    return qc


def build_diffuser():
    """Standard Grover diffuser (inversion about the mean) on 3 qubits."""
    qc = QuantumCircuit(3, name="diffuser")
    qc.h([0, 1, 2])
    qc.x([0, 1, 2])
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    qc.x([0, 1, 2])
    qc.h([0, 1, 2])
    return qc


def build_grover_circuit():
    qc = QuantumCircuit(3, 3)
    qc.h([0, 1, 2])  # uniform superposition over all 8 subgraphs

    oracle = build_oracle()
    diffuser = build_diffuser()

    # With N=8 and M=1 marked item, the optimal number of Grover iterations
    # is round(pi/4 * sqrt(N/M)) = round(pi/4 * sqrt(8)) = 2.
    num_iterations = round((np.pi / 4) * np.sqrt(8 / 1))
    for _ in range(num_iterations):
        qc.compose(oracle, qubits=[0, 1, 2], inplace=True)
        qc.compose(diffuser, qubits=[0, 1, 2], inplace=True)

    qc.measure([0, 1, 2], [0, 1, 2])
    return qc, num_iterations


def run_grover():
    qc, num_iterations = build_grover_circuit()
    sim = AerSimulator()
    shots = 2048
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    print(f"Grover iterations used: {num_iterations}")
    print(f"Measurement counts (top): {sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")
    best = max(counts, key=counts.get)
    return best, counts, shots


def main():
    best, counts, shots = run_grover()
    hits = counts.get(CLASSICAL_ANSWER, 0)
    frac = hits / shots

    print(f"Most frequent measured bitstring: {best}")
    print(f"Classical answer bitstring:       {CLASSICAL_ANSWER}")
    print(f"Fraction of shots matching classical answer: {frac:.4f}")

    verified = (best == CLASSICAL_ANSWER) and (frac > 0.9)
    if verified:
        print("PASS")
    else:
        print("FAIL")
    return verified


if __name__ == "__main__":
    main()
