"""
Erdos problem #743 -- "tree packing conjecture" (graph theory).

Source metadata (from erdosproblems.com data, problems.yaml, number: "743"):
    prize: no
    status: falsifiable
    oeis: ["N/A"]
    tags: ["graph theory"]
    comments: "tree packing conjecture"

LIMITATION: this problem carries no OEIS sequence id (oeis: "N/A" in the
source data). There is therefore no integer sequence to test membership in,
and the problem itself (the Gyarfas/Ringel-style tree packing conjecture --
"any family of trees T_1, ..., T_{n-1} with T_i having i edges can be packed
edge-disjointly into K_n") is a statement about all n, not a single finite
computable value. What CAN be made finite and genuinely computable is one
concrete small instance of the conjecture: n = 4. This script does exactly
that, honestly, as the best available small quantum-testable instance of the
problem's own combinatorial content -- it is not an OEIS lookup.

Classical property being tested (computed from first principles below, not
copied from anywhere):
    K_4 has 6 edges, labelled 0..5:
        0:(0,1) 1:(0,2) 2:(0,3) 3:(1,2) 4:(1,3) 5:(2,3)
    The tree-packing instance for n=4 asks for edge-disjoint trees T_1 (1
    edge), T_2 (2 edges, a path), T_3 (3 edges, a path on all 4 vertices --
    a Hamiltonian path of K_4) that together use all 6 edges of K_4 exactly
    once (1+2+3=6). Equivalently: does K_4 contain a 3-edge subset that is a
    Hamiltonian path? (Whatever 3 edges remain always split validly into a
    2-edge path plus a single edge for K_4, which this script also verifies
    classically, so the Hamiltonian-path subsets are exactly the valid T_3
    choices and the instance is genuinely satisfiable.)

    The finite computable property: among all C(6,3) = 20 three-edge subsets
    of K_4, which ones form a Hamiltonian path (i.e. a spanning path
    touching all 4 vertices, each edge used once, vertex degrees <= 2,
    connected)? This is computed exactly in `classical_hamiltonian_paths()`
    below by brute force over all 20 subsets.

Quantum approach:
    Grover's algorithm searches the 2^6 = 64 computational basis states
    (each qubit = "is edge i part of the candidate T_3 subset?") for the
    marked "good" states -- the 3-edge subsets that are Hamiltonian paths,
    as computed classically. The oracle is built directly from that
    classical enumeration (multi-controlled-Z per marked bitstring), so the
    circuit is a genuine amplitude-amplification search over the true
    solution set, not a fabricated marking. After running Grover with the
    (classically derived) optimal number of iterations on the ideal
    AerSimulator, the most frequently measured bitstrings must exactly equal
    the classically computed solution set for PASS.
"""

from itertools import combinations
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # edges of K_4
N_EDGES = len(EDGES)  # 6 -> 6 qubits, search space 2^6 = 64


def is_hamiltonian_path(edge_subset):
    """True iff these 3 edges of K_4 form a Hamiltonian path (spanning,
    simple path visiting all 4 vertices, each vertex degree <= 2, single
    connected path)."""
    if len(edge_subset) != 3:
        return False
    verts = set()
    deg = {}
    for (u, v) in edge_subset:
        verts.add(u)
        verts.add(v)
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    if len(verts) != 4:
        return False
    if any(d > 2 for d in deg.values()):
        return False
    # connectivity check via union-find / simple BFS
    adj = {v: [] for v in verts}
    for (u, v) in edge_subset:
        adj[u].append(v)
        adj[v].append(u)
    start = next(iter(verts))
    seen = {start}
    stack = [start]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return len(seen) == 4


def classical_hamiltonian_paths():
    """Brute force over all C(6,3)=20 three-edge subsets of K_4; return the
    bitmasks (over EDGES, bit i set iff EDGES[i] in the subset) of those
    that are Hamiltonian paths."""
    good = []
    for combo in combinations(range(N_EDGES), 3):
        subset = [EDGES[i] for i in combo]
        if is_hamiltonian_path(subset):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            good.append(mask)
    return sorted(good)


def verify_full_instance_is_satisfiable(good_masks):
    """Sanity check the whole tree-packing instance (T_1,T_2,T_3 partition
    all 6 edges of K_4), not just the T_3 Hamiltonian-path subproblem:
    for at least one good T_3 mask, confirm the remaining 3 edges really do
    split into a 2-edge path (T_2) + 1 edge (T_1)."""
    all_mask = (1 << N_EDGES) - 1
    for mask in good_masks:
        rem_mask = all_mask & ~mask
        rem_edges = [EDGES[i] for i in range(N_EDGES) if (rem_mask >> i) & 1]
        # try every way to split the 3 remaining edges into a pair + single
        for pair_idx in combinations(range(3), 2):
            pair = [rem_edges[i] for i in pair_idx]
            single_idx = [i for i in range(3) if i not in pair_idx][0]
            single = rem_edges[single_idx]
            u0, v0 = pair[0]
            u1, v1 = pair[1]
            shares_vertex = len({u0, v0} & {u1, v1}) == 1
            if shares_vertex and pair[0] != single and pair[1] != single:
                return True
    return False


def build_grover_circuit(good_masks, n_qubits, n_iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    def apply_oracle(circuit):
        for mask in good_masks:
            zero_bits = [i for i in range(n_qubits) if not (mask >> i) & 1]
            for b in zero_bits:
                circuit.x(b)
            circuit.h(n_qubits - 1)
            circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            circuit.h(n_qubits - 1)
            for b in zero_bits:
                circuit.x(b)

    def apply_diffuser(circuit):
        circuit.h(range(n_qubits))
        circuit.x(range(n_qubits))
        circuit.h(n_qubits - 1)
        circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        circuit.h(n_qubits - 1)
        circuit.x(range(n_qubits))
        circuit.h(range(n_qubits))

    for _ in range(n_iterations):
        apply_oracle(qc)
        apply_diffuser(qc)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    good_masks = classical_hamiltonian_paths()
    good_set = set(good_masks)
    assert len(good_masks) > 0, "no Hamiltonian path exists in K_4 -- unexpected"

    satisfiable = verify_full_instance_is_satisfiable(good_masks)

    N = 1 << N_EDGES  # 64
    M = len(good_masks)
    n_iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    qc = build_grover_circuit(good_masks, N_EDGES, n_iterations)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # bitstrings from qiskit are little-endian in the classical register
    # string (c[n-1]...c[0]); convert to our integer mask (bit i = edge i).
    def bitstring_to_mask(bitstr):
        # bitstr[0] corresponds to qubit n_qubits-1 ... bitstr[-1] to qubit 0
        rev = bitstr[::-1]
        mask = 0
        for i, ch in enumerate(rev):
            if ch == "1":
                mask |= (1 << i)
        return mask

    mask_counts = {}
    for bitstr, c in counts.items():
        mask_counts[bitstring_to_mask(bitstr)] = mask_counts.get(bitstring_to_mask(bitstr), 0) + c

    # top M measured masks should be exactly the amplified (good) states
    ranked = sorted(mask_counts.items(), key=lambda kv: -kv[1])
    top_measured = set(mask for mask, _ in ranked[:M])

    good_shots = sum(c for mask, c in mask_counts.items() if mask in good_set)
    good_fraction = good_shots / shots

    print(f"K_4 edges: {EDGES}")
    print(f"Classical Hamiltonian-path (T_3) subsets of K_4, as edge-index bitmasks: {good_masks}")
    print(f"Number of solutions M = {M} out of N = {N} states; Grover iterations = {n_iterations}")
    print(f"Full tree-packing instance (T_1,T_2,T_3 partition all 6 edges) satisfiable: {satisfiable}")
    print(f"Top {M} measured states: {sorted(top_measured)}")
    print(f"Fraction of shots landing on a classical solution state: {good_fraction:.4f}")

    # M/N = 12/64 is large enough that Grover's amplification is modest
    # (a single amplitude-amplification round on this instance theoretically
    # lands on a solution roughly 60% of the time); the strong, exact check
    # is that the M most-measured states are precisely the classically
    # computed solution set, backed by amplification clearly above the
    # uniform-random baseline M/N = 0.1875.
    verified = (top_measured == good_set) and satisfiable and (good_fraction > 0.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
