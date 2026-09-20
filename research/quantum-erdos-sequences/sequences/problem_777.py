"""
Erdos problem #777 (per manman4/erdosproblems data/problems.yaml, entry
`number: "777"`): tags ["graph theory", "combinatorics"], informal_status
"solved", and crucially `oeis: ["N/A"]` -- the dataset records NO OEIS
sequence id for this problem. There is therefore no "quantum-testable
sequence" that can honestly be derived from problem #777 itself: there is
no integer sequence membership/growth property in the record to check a
quantum circuit against.

LIMITATION (reported honestly, not glossed over): this script does not
verify any property of an Erdos-problem-777 OEIS sequence, because no such
sequence id exists in the source data. Faking one (inventing an OEIS id or
copying an unrelated one) would violate the task's requirement not to
fabricate content.

Best-effort substitute: since problem #777 is tagged "graph theory" /
"combinatorics" and is about a finite combinatorial property (graphs), we
build a genuine, self-contained finite combinatorial search problem in that
spirit and solve it with a real Grover search circuit on the ideal
AerSimulator, verifying the quantum result against a from-scratch classical
computation. Specifically:

  Property tested: "which 2-element subsets {i, j} of {0,1,2,3} (i.e. edges
  of the complete graph K4, encoded by a 2-bit index over the 6 possible
  edges) have both endpoints even?" i.e. edges of K4 all of whose vertices
  lie in {0, 2}. Over the 6 edges of K4, enumerated in a fixed order, we
  compute classically (by brute force over all C(4,2)=6 edges) exactly
  which edges satisfy "both endpoints are even," giving a small marked set
  in a 3-qubit search space (search space size N=8, padded from 6 edges),
  and use Grover's algorithm to amplify and recover that marked set,
  comparing the quantum measurement distribution to the classical answer.

This is offered strictly as a best-honest-effort finite combinatorial
circuit in the spirit of problem #777's graph-theory/combinatorics tags,
NOT as a verification of any Erdos-problem-777 OEIS sequence -- no such
sequence exists in the source data.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCMTGate, ZGate


def classical_marked_edges():
    """Brute-force, from first principles: enumerate all 6 edges of K4
    (vertices 0..3) in a fixed order, and determine which have both
    endpoints even (i.e. both endpoints in {0, 2})."""
    vertices = [0, 1, 2, 3]
    edges = list(itertools.combinations(vertices, 2))  # fixed order, 6 edges
    marked = []
    for idx, (a, b) in enumerate(edges):
        if a % 2 == 0 and b % 2 == 0:
            marked.append(idx)
    return edges, marked


def build_oracle(n_qubits, marked_indices):
    """Phase-flip oracle marking each index in marked_indices (as an
    n_qubits-bit binary string) with a -1 phase, via a multi-controlled Z
    on the appropriate X-conjugated pattern."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")
        # flip qubits where bit == '0' so that a multi-controlled Z fires
        # exactly on this basis state
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
            qc.append(mcz, list(range(n_qubits)))
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
        qc.append(mcz, list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits, marked_indices, shots=4096):
    n_marked = len(marked_indices)
    n_total = 2 ** n_qubits
    if n_marked == 0 or n_marked == n_total:
        raise ValueError("Grover requires 0 < |marked| < N")

    theta = math.asin(math.sqrt(n_marked / n_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(n_qubits, marked_indices)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    qc = qc.decompose().decompose()
    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    edges, marked = classical_marked_edges()
    print("K4 edges (fixed order):", edges)
    print("Classical: edges with both endpoints even (indices):", marked)
    for i in marked:
        print(f"  edge index {i} -> {edges[i]}")

    # search space: 3 qubits covers indices 0..7, enough for 6 edges (0..5)
    n_qubits = 3
    counts, iterations = run_grover(n_qubits, marked)
    print(f"\nGrover iterations used: {iterations}")
    print("Measurement counts:", counts)

    total_shots = sum(counts.values())
    # Qiskit's classical-register bitstrings are printed with qubit n-1
    # first (little-endian), so reverse our MSB-first index encoding to
    # match the simulator's output convention.
    marked_bitstrings = {format(i, f"0{n_qubits}b")[::-1] for i in marked}
    marked_hits = sum(c for bs, c in counts.items() if bs in marked_bitstrings)
    marked_fraction = marked_hits / total_shots

    # classical baseline: uniform random guess would hit a marked state
    # with probability |marked|/2**n_qubits
    baseline = len(marked) / (2 ** n_qubits)

    # most frequently measured bitstring(s) should be exactly the marked set
    max_count = max(counts.values())
    top_bitstrings = {bs for bs, c in counts.items() if c == max_count}

    print(f"\nFraction of shots landing on a marked state: {marked_fraction:.3f}")
    print(f"Uniform-random baseline fraction: {baseline:.3f}")
    print(f"Most frequent measured bitstring(s): {sorted(top_bitstrings)}")
    print(f"Expected marked bitstring(s): {sorted(marked_bitstrings)}")

    success = (
        marked_fraction > 2 * baseline
        and top_bitstrings.issubset(marked_bitstrings)
    )

    if success:
        print("\nPASS: Grover search amplified exactly the classically "
              "computed marked edge set (both endpoints even), well above "
              "the uniform-random baseline.")
    else:
        print("\nFAIL: quantum result did not match the classical answer.")

    return success


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
