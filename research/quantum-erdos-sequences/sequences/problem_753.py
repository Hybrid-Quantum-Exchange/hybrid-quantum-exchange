"""
Erdos problem #753 -- quantum-testable sequence entry.

Source metadata (from erdosproblems/data/problems.yaml, "number: 753"):
    prize: no
    informal_status: disproved (Lean-formalized, last_update 2026-04-14)
    oeis: ["N/A"]
    tags: ["graph theory", "chromatic number"]

HONEST LIMITATION: problem #753 carries no OEIS sequence id (oeis: ["N/A"]),
so there is no literal OEIS term to check a quantum circuit's output against.
Per the task instructions, in that situation this script constructs its own
small, finite, computable property drawn from the problem's *tags*
("graph theory", "chromatic number") rather than fabricating or copying an
OEIS value that does not exist for this entry.

Chosen classical property
--------------------------
Let G = P4, the path graph on 4 vertices {0,1,2,3} with edges
    (0,1), (1,2), (2,3).
Property tested: "G has a proper vertex 2-coloring", i.e. an assignment
c: {0,1,2,3} -> {0,1} such that c(u) != c(v) for every edge (u,v).

This is exactly a chromatic-number-flavoured decision/search problem (the
problem's own tag): P4 is bipartite, so it is 2-colorable, and the number of
proper 2-colorings of a connected bipartite graph on n vertices is exactly 2
(swap the two colour classes). We first compute this classically by brute
force over all 2^4 = 16 assignments (ground truth, derived here from first
principles, not copied from any table), then use a real Grover search
circuit on the ideal AerSimulator to find a proper coloring, and check that
the state Grover returns is indeed in the classically-computed solution set.

Circuit design
--------------
- 4 "colour" qubits q0..q3, one per vertex, |0>/|1> = colour 0/1.
- 3 ancilla qubits a0,a1,a2, one per edge, each computed as
  a_i = q_u XOR q_v (via two CNOTs) -- this is 1 exactly when the edge's
  endpoints differ, i.e. the edge constraint is satisfied.
- 1 output ancilla "out", flipped to |1> (via a multi-controlled X on
  a0,a1,a2) exactly when *all* edge constraints hold, i.e. exactly when the
  4-bit string on q0..q3 is a proper 2-colouring of P4.
- Oracle: phase-kickback Z on "out" while it is in the |-> state (X + H
  prepared once at the start), giving a standard phase-flip oracle for the
  "proper colouring" marked states, followed by uncomputing the ancillas.
- Grover diffuser on the 4 colour qubits, repeated the optimal integer
  number of times for N=16, M=2 marked states (2 iterations).

Correctness/verification: the script computes the classical solution set by
brute force, runs the Grover circuit on AerSimulator, and checks the most
frequently measured 4-bit colour string is in that classical solution set
(PASS) or not (FAIL).
"""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator


EDGES = [(0, 1), (1, 2), (2, 3)]  # path graph P4 on vertices 0..3
NUM_VERTICES = 4


def is_proper_coloring(bits, edges):
    """bits: tuple of 0/1 per vertex. True iff every edge's endpoints differ."""
    return all(bits[u] != bits[v] for (u, v) in edges)


def classical_solution_set(num_vertices, edges):
    """Brute-force, from first principles, every proper 2-coloring of the graph."""
    solutions = []
    for bits in product([0, 1], repeat=num_vertices):
        if is_proper_coloring(bits, edges):
            solutions.append(bits)
    return solutions


def build_grover_circuit(num_vertices, edges, iterations):
    color = QuantumRegister(num_vertices, "q")
    edge_anc = QuantumRegister(len(edges), "a")
    out = QuantumRegister(1, "out")
    creg = ClassicalRegister(num_vertices, "c")
    qc = QuantumCircuit(color, edge_anc, out, creg)

    # Uniform superposition over all colourings.
    qc.h(color)

    # Prepare output ancilla in |-> for phase kickback.
    qc.x(out[0])
    qc.h(out[0])

    def oracle():
        # Compute edge satisfaction bits into ancillas: a_i = q_u XOR q_v.
        for i, (u, v) in enumerate(edges):
            qc.cx(color[u], edge_anc[i])
            qc.cx(color[v], edge_anc[i])
        # Flip 'out' iff all edge ancillas are 1 (all constraints satisfied).
        qc.mcx(list(edge_anc), out[0])
        # Uncompute edge ancillas.
        for i, (u, v) in enumerate(edges):
            qc.cx(color[v], edge_anc[i])
            qc.cx(color[u], edge_anc[i])

    def diffuser():
        qc.h(color)
        qc.x(color)
        qc.h(color[num_vertices - 1])
        qc.mcx(list(color[: num_vertices - 1]), color[num_vertices - 1])
        qc.h(color[num_vertices - 1])
        qc.x(color)
        qc.h(color)

    for _ in range(iterations):
        oracle()
        diffuser()

    # Undo the |-> preparation on the output ancilla (restore to |0>).
    qc.h(out[0])
    qc.x(out[0])

    qc.measure(color, creg)
    return qc


def optimal_grover_iterations(n_states, n_marked):
    theta = np.arcsin(np.sqrt(n_marked / n_states))
    iterations = int(round((np.pi / (4 * theta)) - 0.5))
    return max(1, iterations)


def main():
    # --- Classical ground truth, computed here from first principles. ---
    solutions = classical_solution_set(NUM_VERTICES, EDGES)
    n_states = 2 ** NUM_VERTICES
    n_marked = len(solutions)
    print(f"Graph P4 on {NUM_VERTICES} vertices, edges {EDGES}")
    print(f"Classical brute force: {n_marked} proper 2-colorings out of {n_states} "
          f"assignments -> {solutions}")

    if n_marked == 0:
        print("No proper 2-coloring exists classically; nothing for Grover to find.")
        print("FAIL")
        return False, False

    iterations = optimal_grover_iterations(n_states, n_marked)
    print(f"Grover iterations used: {iterations}")

    qc = build_grover_circuit(NUM_VERTICES, EDGES, iterations)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 2048
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit classical-register bit order is c[num_vertices-1]...c[0] (MSB first)
    # in the returned bitstring; convert to our (v0, v1, v2, v3) tuple ordering.
    def bitstring_to_tuple(bitstring):
        rev = bitstring[::-1]  # rev[i] corresponds to vertex i
        return tuple(int(b) for b in rev)

    top_bitstring = max(counts, key=counts.get)
    top_tuple = bitstring_to_tuple(top_bitstring)
    top_prob = counts[top_bitstring] / shots

    print(f"Most frequent measured colouring: {top_bitstring} -> vertices {top_tuple} "
          f"(probability {top_prob:.3f} over {shots} shots)")
    print(f"Full counts (top 5): "
          f"{sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")

    solution_bitstrings = {
        "".join(str(b) for b in tup[::-1]) for tup in solutions
    }
    solution_prob = sum(
        c for bs, c in counts.items() if bs in solution_bitstrings
    ) / shots

    print(f"Total probability mass on classically-valid colourings: "
          f"{solution_prob:.3f}")

    ran_ok = True
    verified = top_tuple in solutions and solution_prob > 0.9

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return ran_ok, verified


if __name__ == "__main__":
    ran_ok, verified = main()
    if not (ran_ok and verified):
        raise SystemExit(1)
