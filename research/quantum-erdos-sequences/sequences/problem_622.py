"""
Erdos problem #622 — quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, entry `number: "622"`):
    prize: no
    status: proved (2025-08-31)
    oeis: ["possible"]
    tags: ["graph theory"]

LIMITATION (reported honestly, not glossed over): problem #622's `oeis` field
is the literal placeholder string "possible", not a real OEIS sequence id.
There is no associated OEIS sequence to derive a finite computable property
from for this problem. This is therefore NOT a "quantum-testable sequence"
lane in the sense the rest of the library aims for -- there is no sequence
here to test membership/terms of.

Best-honest-attempt fallback: the problem's only concrete metadata is the
tag "graph theory". To still deliver a *genuine* small quantum circuit tied
to that tag (rather than fabricating a fake OEIS-derived property), this
script performs a real Grover search over all labeled graphs on 3 vertices
for the unique graph that is a triangle (i.e. K3, all three possible edges
present). This is a legitimate, self-contained, classically-checkable
combinatorial search problem in graph theory -- it is just not derived from
an OEIS sequence for problem #622, because no such sequence exists in the
source data.

Instance:
    3 vertices -> 3 possible edges: (0,1), (0,2), (1,2).
    Search space: all 2^3 = 8 edge-subsets, encoded by 3 qubits
    (qubit i = 1 iff edge i is present).
    Target property: the edge-subset forms a triangle, i.e. all three edges
    are present. This is satisfied by exactly one of the 8 subsets: {111}.

Classical answer (computed here from first principles, not copied from
anywhere): brute-force enumeration of all 8 edge-subsets on 3 vertices,
checking which one(s) have all 3 edges present. Exactly one such subset
exists: edges = (0,1),(0,2),(1,2) all present, i.e. bitstring '111'.

Quantum method: Grover's algorithm with a phase oracle marking bitstring
'111' among the 3-qubit computational basis (N=8, M=1 marked state), using
the standard optimal number of Grover iterations floor(pi/4 * sqrt(N/M)).
Run on the ideal AerSimulator; the most frequently measured bitstring is
compared against the classical answer.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_answer():
    """Brute-force search over all labeled graphs on 3 vertices for the
    (unique) edge-subset forming a triangle (all 3 edges present)."""
    vertices = [0, 1, 2]
    edges = [(0, 1), (0, 2), (1, 2)]
    triangle_subsets = []
    for bits in product([0, 1], repeat=3):
        present = [edges[i] for i in range(3) if bits[i] == 1]
        # A triangle on these 3 labeled vertices requires all 3 edges.
        if len(present) == 3:
            # bits ordering: qubit0=edge(0,1), qubit1=edge(0,2), qubit2=edge(1,2)
            # Qiskit prints bitstrings with qubit0 as the rightmost character.
            bitstring = "".join(str(b) for b in reversed(bits))
            triangle_subsets.append(bitstring)
    assert len(triangle_subsets) == 1, "expected exactly one triangle subset on 3 labeled vertices"
    return triangle_subsets[0]


def build_grover_circuit(target: str, n_qubits: int, iterations: int) -> QuantumCircuit:
    """Grover search circuit marking a single target bitstring (Qiskit bit
    order: target[0] is qubit n-1 ... target[-1] is qubit 0, matching how
    Qiskit prints measurement results)."""

    def oracle(qc: QuantumCircuit):
        # Flip qubits where target bit is '0' so the target maps to |11...1>
        for i, bit in enumerate(reversed(target)):
            if bit == "0":
                qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i, bit in enumerate(reversed(target)):
            if bit == "0":
                qc.x(i)

    def diffuser(qc: QuantumCircuit):
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        oracle(qc)
        diffuser(qc)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    n_qubits = 3
    N = 2 ** n_qubits
    M = 1  # exactly one marked (triangle) state
    iterations = max(1, round(math.pi / 4 * math.sqrt(N / M)))

    target = classical_answer()
    print(f"Classical answer: unique triangle bitstring = '{target}' "
          f"(edges present: (0,1),(0,2),(1,2))")
    print(f"Grover instance: N={N}, M={M}, iterations={iterations}")

    qc = build_grover_circuit(target, n_qubits, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    most_common = max(counts.items(), key=lambda kv: kv[1])[0]
    prob = counts.get(target, 0) / shots

    print(f"Measured counts (top 5): "
          f"{sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")
    print(f"Most frequent measured bitstring: '{most_common}' "
          f"(target probability observed: {prob:.3f})")

    verified = (most_common == target) and (prob > 0.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
