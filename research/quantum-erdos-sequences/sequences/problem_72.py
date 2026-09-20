"""
Erdos problem #72 (erdosproblems.com), quantum-testable lane.

Source metadata (from /home/user/manman4/erdosproblems/data/problems.yaml,
entry "number: \"72\"", verified 2026-09-19):
    prize: $100
    informal_status: proved (2025-08-31)
    oeis: ["N/A"]
    tags: ["graph theory", "cycles"]

LIMITATION, stated honestly up front: problem #72 has no associated OEIS
sequence id in the source data (oeis == ["N/A"]). There is therefore no
"identify a term of OEIS sequence X" property to build a Grover/oracle
circuit around, and this script does not pretend otherwise. Per the task's
fallback instructions, this is the best-effort substitute: a small, finite,
genuinely computable property drawn directly from the problem's own tags
("graph theory", "cycles"), verified classically from first principles in
this script and then checked with a real Grover-search quantum circuit run
on AerSimulator.

Chosen classical property
--------------------------
Take the complete graph K4 on labeled vertices {0,1,2,3}. Consider every
Hamiltonian cycle that starts at vertex 0, i.e. every permutation of
(1,2,3) appended after 0. There are 3! = 6 such permutations, indexed
0..5 in the order produced by itertools.permutations. Each permutation
traces a 4-cycle through all vertices; reversing the direction of a cycle
gives the same *undirected* cycle. So among the 6 directed labelings there
are exactly 3 distinct undirected Hamiltonian cycles in K4, and each has
exactly one representative permutation-index that is numerically smallest
among the (at most two) permutation-indices tracing it.

The property under test: "which permutation-indices (0..5, embeddable in
3 qubits, since 2^3 = 8) are the canonical (numerically-smallest) index of
a distinct undirected Hamiltonian cycle of K4?" This is computed here
classically from first principles (no OEIS lookup, no literature value),
and the answer is a finite, checkable set of marked basis states.

Quantum circuit
----------------
A standard Grover search over 3 qubits (8 basis states, indices 0-7; only
0-5 are ever valid permutation indices, 6 and 7 are never marked). The
oracle phase-flips exactly the classically-computed marked indices (the
canonical representatives found above); the diffuser is the standard
Grover diffusion operator. One Grover iteration is enough for 3 marked
states out of 8 (near the optimal ~1 iteration for this ratio). The
circuit is run on AerSimulator (statevector, no noise), and the script
checks that the measurement distribution is concentrated on the
classically-marked indices, printing PASS/FAIL.
"""

from itertools import permutations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_marked_indices():
    """Classically derive, from first principles, the canonical
    permutation-indices of the distinct undirected Hamiltonian cycles of
    K4 that start at vertex 0."""
    perms = list(permutations([1, 2, 3]))  # 6 permutations, index 0..5

    def cycle_edge_set(perm):
        path = (0,) + perm
        edges = set()
        for i in range(len(path)):
            a, b = path[i], path[(i + 1) % len(path)]
            edges.add(frozenset((a, b)))
        return frozenset(edges)

    seen = {}
    canonical = []
    for idx, perm in enumerate(perms):
        es = cycle_edge_set(perm)
        if es not in seen:
            seen[es] = idx
            canonical.append(idx)
    return sorted(canonical), perms


def build_oracle(n_qubits, marked_indices):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")[::-1]  # little-endian per qubit order
        flip_qubits = [q for q, b in enumerate(bits) if b == "0"]
        for q in flip_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in flip_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits, marked_indices, iterations=1, shots=4096):
    oracle = build_oracle(n_qubits, marked_indices)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    marked_indices, perms = classical_marked_indices()
    print(f"Permutations of (1,2,3) after start vertex 0: {perms}")
    print(f"Classically-derived canonical Hamiltonian-cycle indices of K4: {marked_indices}")
    assert marked_indices == [0, 1, 2], (
        "Sanity check failed: K4 has exactly 3 distinct undirected "
        "Hamiltonian cycles, whose canonical representatives among "
        "itertools.permutations([1,2,3]) should be indices [0,1,2]."
    )

    n_qubits = 3
    counts = run_grover(n_qubits, marked_indices, iterations=1, shots=4096)
    print(f"Measurement counts: {counts}")

    total_shots = sum(counts.values())
    marked_bitstrings = {format(i, f"0{n_qubits}b")[::-1] for i in marked_indices}
    # Qiskit reports bitstrings with qubit 0 as the rightmost character.
    marked_bitstrings_qiskit = {format(i, f"0{n_qubits}b") for i in marked_indices}

    marked_hits = sum(c for b, c in counts.items() if b in marked_bitstrings_qiskit)
    marked_fraction = marked_hits / total_shots

    print(f"Marked bitstrings (Qiskit little-endian order): {sorted(marked_bitstrings_qiskit)}")
    print(f"Fraction of shots landing on a classically-marked index: {marked_fraction:.4f}")

    # With 3 marked states out of 8 and 1 Grover iteration, the theoretical
    # success probability is high (>90%); require a solid majority as the
    # pass threshold to be robust to simulator/shot noise.
    verified = marked_fraction > 0.80

    print("PASS" if verified else "FAIL")


if __name__ == "__main__":
    main()
