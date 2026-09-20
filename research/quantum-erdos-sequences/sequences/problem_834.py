"""
Erdos problem #834 -- quantum-testable lane (best-effort, limitation noted).

Source record: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: \"834\"":
    prize: no
    informal_status: solved (last_update 2026-01-01)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["graph theory", "hypergraphs"]

LIMITATION (read before trusting the "OEIS id(s) used" claim): problem #834's
YAML record carries no OEIS sequence id -- the field is the literal string
"N/A". There is therefore no OEIS sequence to derive a finite computable term
membership/counting property from for this problem, as the assignment
otherwise requires. Rather than fabricate a fake OEIS id or copy an
unrelated one, this script honestly falls back to a small, genuinely
finite, classically-checkable property that is faithful to the problem's
*tags* (graph theory / hypergraphs) instead of to a specific OEIS entry:

    Property tested: among all labeled simple graphs on 3 vertices
    (there are 2^3 = 8, one bit per possible edge {1,2},{1,3},{2,3}),
    exactly one graph, up to the fixed vertex labeling, is both
    (a) triangle-free (does not contain all 3 possible edges) and
    (b) connected (every vertex reachable from every other).
    That graph is the 3-vertex path (exactly 2 of the 3 edges present).
    This is computed from first principles below by brute-force over all
    8 edge-subsets -- no OEIS value is copied.

This is a direct, self-contained combinatorial search problem (find the
unique 3-bit string satisfying a boolean predicate), which is exactly the
shape Grover's algorithm targets, so it is used here as a genuine quantum
search circuit rather than a trivial arithmetic wrapper.

Because problem #834 itself has no OEIS sequence, this script cannot claim
to "quantum-test" a specific OEIS sequence for #834; it demonstrates the
quantum search machinery on a small instance faithful to the problem's
tags, and that limitation is reported honestly rather than papered over.
"""

import itertools
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


N_VERTICES = 3
EDGES = list(itertools.combinations(range(N_VERTICES), 2))  # [(0,1),(0,2),(1,2)]
NUM_EDGES = len(EDGES)  # 3 qubits: one per possible edge


def is_connected(present_edges, n_vertices):
    if n_vertices <= 1:
        return True
    adj = {v: set() for v in range(n_vertices)}
    for (a, b) in present_edges:
        adj[a].add(b)
        adj[b].add(a)
    seen = {0}
    frontier = [0]
    while frontier:
        u = frontier.pop()
        for w in adj[u]:
            if w not in seen:
                seen.add(w)
                frontier.append(w)
    return len(seen) == n_vertices


def classical_marked_values():
    """Brute force over all 2^NUM_EDGES edge-subsets of the 3-vertex graph.

    Marked = triangle-free (not all 3 edges present) AND connected.
    Returns the set of integers (bitmask, bit i = EDGES[i] present) that
    satisfy the property, computed from first principles (no OEIS lookup).
    """
    marked = set()
    for mask in range(2 ** NUM_EDGES):
        present = [EDGES[i] for i in range(NUM_EDGES) if (mask >> i) & 1]
        triangle_free = mask != (2 ** NUM_EDGES - 1)  # not all 3 edges
        connected = is_connected(present, N_VERTICES)
        if triangle_free and connected:
            marked.add(mask)
    return marked


def build_oracle(marked_values, num_qubits):
    """Phase-flip oracle: |x> -> -|x> for x in marked_values, else |x>."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{num_qubits}b")[::-1]  # qubit i <-> bit i
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
            qc.h(num_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def run_grover(marked_values, num_qubits, shots=2048):
    n_marked = len(marked_values)
    n_total = 2 ** num_qubits
    # optimal number of Grover iterations
    theta = np.arcsin(np.sqrt(n_marked / n_total))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5)) if theta > 0 else 0

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    oracle = build_oracle(marked_values, num_qubits)
    diffuser = build_diffuser(num_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(num_qubits))
        qc.append(diffuser.to_gate(), range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))
    qc = qc.decompose().decompose().decompose()

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    classical_marked = classical_marked_values()
    print(f"Classical brute-force marked bitmasks (edge present/absent, "
          f"triangle-free AND connected 3-vertex graphs): {sorted(classical_marked)}")
    for m in sorted(classical_marked):
        present = [EDGES[i] for i in range(NUM_EDGES) if (m >> i) & 1]
        print(f"  mask={m:03b} edges={present}")

    counts, iterations = run_grover(classical_marked, NUM_EDGES, shots=4096)
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")

    # bitstrings from Qiskit are big-endian string of clbits (qubit num_qubits-1 .. 0)
    quantum_marked_hits = {}
    for bitstring, count in counts.items():
        value = int(bitstring[::-1], 2)  # convert back to our little-endian int convention
        quantum_marked_hits[value] = quantum_marked_hits.get(value, 0) + count

    total_shots = sum(counts.values())
    marked_shots = sum(c for v, c in quantum_marked_hits.items() if v in classical_marked)
    marked_fraction = marked_shots / total_shots

    quantum_top_value = max(quantum_marked_hits, key=quantum_marked_hits.get)
    # Uniform-random baseline would land on a marked value with probability
    # len(classical_marked)/2**NUM_EDGES = 3/8 = 0.375. Grover amplification
    # with the optimal single iteration here has a per-shot theoretical
    # success probability of 1.0, achieved only up to finite-shot sampling
    # noise, so a generous-but-still-far-above-baseline threshold is used.
    verified = (quantum_top_value in classical_marked) and (marked_fraction > 0.75)

    print(f"Most frequent measured value: {quantum_top_value:03b} "
          f"(in classical marked set: {quantum_top_value in classical_marked})")
    print(f"Fraction of shots landing on a classically-marked value: {marked_fraction:.4f}")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
