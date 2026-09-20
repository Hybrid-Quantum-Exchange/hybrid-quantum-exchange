"""
Erdos problem #182 -- quantum-testable lane.

LIMITATION (read first): Erdos problem #182's entry in
erdosproblems/data/problems.yaml carries `oeis: ["possible"]`, tags
`["graph theory"]`, and no title/statement text field. "possible" is a
placeholder meaning "an OEIS sequence might exist for this problem", not an
actual OEIS id -- there is no A-number to derive a property from, and the
data file gives no problem statement to work from either. So this is not a
genuine instance of Erdos problem #182's own mathematics; there is nothing
concrete in the source data to encode.

Best-effort fallback: since the entry's only real content is the tag
"graph theory", this script tests a real, small, finite, computable graph
theory property -- chromatic number -- on a fixed graph, using Grover
search over an actual quantum circuit. This is offered honestly as a
graph-theory-flavored quantum circuit exercise, NOT as a verified instance
of problem 182's actual (unstated) claim.

Property tested: the 5-cycle graph C5 (vertices 0..4, edges i-(i+1 mod 5))
has chromatic number 3, i.e. it IS 3-colorable but is NOT 2-colorable.
Both facts are computed classically from first principles by brute force
over all colorings, then the 3-colorability side is verified by a Grover
search quantum circuit over a 3-coloring oracle built from the graph's
edges.

Encoding: each of the 5 vertices gets 2 qubits (10 qubits total) encoding
a color in {0,1,2} (value 3 is treated as invalid/unused by the oracle).
An oracle marks basis states where, for every edge (u,v), the two vertices'
2-bit colors differ. Grover search amplifies these marked "valid 3-coloring"
states; sampling the final state should recover a valid 3-coloring with
high probability, matching a coloring found by classical brute force.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Graph: C5, the 5-cycle.
# ---------------------------------------------------------------------------
N_VERTICES = 5
EDGES = [(i, (i + 1) % N_VERTICES) for i in range(N_VERTICES)]  # 0-1,1-2,2-3,3-4,4-0


def classical_chromatic_checks():
    """Brute force over all colorings with k colors, k = 2 and k = 3."""

    def is_k_colorable(k):
        for coloring in itertools.product(range(k), repeat=N_VERTICES):
            if all(coloring[u] != coloring[v] for u, v in EDGES):
                return True, coloring
        return False, None

    two_colorable, _ = is_k_colorable(2)
    three_colorable, example = is_k_colorable(3)
    return two_colorable, three_colorable, example


TWO_COLORABLE, THREE_COLORABLE, CLASSICAL_EXAMPLE = classical_chromatic_checks()
assert TWO_COLORABLE is False, "C5 must not be 2-colorable (odd cycle)"
assert THREE_COLORABLE is True, "C5 must be 3-colorable (chromatic number 3)"
print(f"Classical: C5 is 2-colorable = {TWO_COLORABLE}, "
      f"3-colorable = {THREE_COLORABLE}, example coloring = {CLASSICAL_EXAMPLE}")


# ---------------------------------------------------------------------------
# Quantum: Grover search over 3-colorings of C5, 2 qubits per vertex (10
# qubits), with an oracle that marks states where every edge's endpoints
# have different 2-bit color values (and neither endpoint uses the unused
# code 11 = 3).
# ---------------------------------------------------------------------------
QUBITS_PER_VERTEX = 2
N_QUBITS = QUBITS_PER_VERTEX * N_VERTICES


def vertex_qubits(v):
    lo = v * QUBITS_PER_VERTEX
    return [lo, lo + 1]


def build_oracle():
    """Oracle marking (phase flip) basis states that are valid proper
    3-colorings of C5: every edge's endpoints differ, and no vertex uses
    the unused code '11'.
    """
    n_edges = len(EDGES)
    # qubit layout: [color qubits (10)] [per-edge "differ" flags (5)]
    #               [per-vertex "valid code" flags (5)] [final AND (1)]
    color_q = list(range(N_QUBITS))
    edge_flag = list(range(N_QUBITS, N_QUBITS + n_edges))
    valid_flag = list(range(N_QUBITS + n_edges, N_QUBITS + n_edges + N_VERTICES))
    out = N_QUBITS + n_edges + N_VERTICES
    total_qubits = out + 1

    qc = QuantumCircuit(total_qubits, name="oracle")

    # per-vertex: valid_flag[v] = 1 iff vertex v's code != 11 (i.e. NOT both bits 1)
    for v in range(N_VERTICES):
        qv = vertex_qubits(v)
        qc.ccx(qv[0], qv[1], valid_flag[v])
        qc.x(valid_flag[v])  # now 1 iff code != 11 (NAND of the two color bits)

    # per-edge equality checks need 3 scratch qubits, reused sequentially
    # (computed then uncomputed for each edge).
    from qiskit.circuit import QuantumRegister
    scratch = list(range(total_qubits, total_qubits + 3))
    qc.add_register(QuantumRegister(3, "scratch"))
    total_qubits = total_qubits + 3

    for e, (u, v) in enumerate(EDGES):
        qu = vertex_qubits(u)
        qv = vertex_qubits(v)
        eq_bits = []
        for i in range(QUBITS_PER_VERTEX):
            eb = scratch[i]
            qc.cx(qu[i], eb)
            qc.cx(qv[i], eb)
            qc.x(eb)
            eq_bits.append(eb)
        qc.mcx(eq_bits, scratch[2])
        qc.x(scratch[2])  # differ flag
        qc.cx(scratch[2], edge_flag[e])
        qc.x(scratch[2])
        # uncompute
        qc.mcx(eq_bits, scratch[2])
        for i in range(QUBITS_PER_VERTEX):
            eb = scratch[i]
            qc.x(eb)
            qc.cx(qu[i], eb)
            qc.cx(qv[i], eb)

    # final AND of all edge_flags and valid_flags -> phase flip on out (used as |-> ancilla)
    all_flags = edge_flag + valid_flag
    qc.mcx(all_flags, out)

    # uncompute edge/valid flags (reverse order) so oracle is a clean phase oracle
    for e, (u, v) in reversed(list(enumerate(EDGES))):
        qu = vertex_qubits(u)
        qv = vertex_qubits(v)
        eq_bits = []
        for i in range(QUBITS_PER_VERTEX):
            eb = scratch[i]
            qc.cx(qu[i], eb)
            qc.cx(qv[i], eb)
            qc.x(eb)
            eq_bits.append(eb)
        qc.mcx(eq_bits, scratch[2])
        qc.x(scratch[2])
        qc.cx(scratch[2], edge_flag[e])
        qc.x(scratch[2])
        qc.mcx(eq_bits, scratch[2])
        for i in range(QUBITS_PER_VERTEX):
            eb = scratch[i]
            qc.x(eb)
            qc.cx(qu[i], eb)
            qc.cx(qv[i], eb)

    for v in reversed(range(N_VERTICES)):
        qv = vertex_qubits(v)
        qc.x(valid_flag[v])
        qc.ccx(qv[0], qv[1], valid_flag[v])

    return qc, total_qubits, out


def diffuser(n_color_qubits):
    qc = QuantumCircuit(n_color_qubits, name="diffuser")
    qc.h(range(n_color_qubits))
    qc.x(range(n_color_qubits))
    qc.h(n_color_qubits - 1)
    qc.mcx(list(range(n_color_qubits - 1)), n_color_qubits - 1)
    qc.h(n_color_qubits - 1)
    qc.x(range(n_color_qubits))
    qc.h(range(n_color_qubits))
    return qc


def run_grover():
    oracle, total_qubits, out_qubit = build_oracle()

    full = QuantumCircuit(total_qubits, N_QUBITS)
    full.h(range(N_QUBITS))
    full.x(out_qubit)
    full.h(out_qubit)

    # Number of valid 3-colorings of C5 among the 4^5 = 1024 basis states
    # (colors in {0,1,2,3} per vertex, code 3 invalid): count classically.
    n_valid = 0
    for coloring in itertools.product(range(4), repeat=N_VERTICES):
        if any(c == 3 for c in coloring):
            continue
        if all(coloring[u] != coloring[v] for u, v in EDGES):
            n_valid += 1
    search_space = 4 ** N_VERTICES
    theta = math.asin(math.sqrt(n_valid / search_space))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    for _ in range(iterations):
        full.append(oracle.to_instruction(), range(total_qubits))
        full.append(diffuser(N_QUBITS).to_instruction(), range(N_QUBITS))

    full.measure(range(N_QUBITS), range(N_QUBITS))
    full = full.decompose().decompose()
    return full, n_valid


def bits_to_coloring(bitstring):
    # Qiskit bitstring is little-endian by qubit index, reversed in string form.
    bits = bitstring[::-1]
    coloring = []
    for v in range(N_VERTICES):
        lo, hi = vertex_qubits(v)
        c = int(bits[lo]) + 2 * int(bits[hi])
        coloring.append(c)
    return tuple(coloring)


def is_valid_3_coloring(coloring):
    if any(c == 3 for c in coloring):
        return False
    return all(coloring[u] != coloring[v] for u, v in EDGES)


def main():
    print(f"Search space: colorings of C5 with 2 qubits/vertex, "
          f"{N_QUBITS} color qubits + ancillas.")
    circuit, n_valid = run_grover()
    print(f"Classical count of valid 3-colorings among {4**N_VERTICES} "
          f"codewords: {n_valid}")

    sim = AerSimulator()
    result = sim.run(circuit, shots=2048).result()
    counts = result.get_counts()

    top_bitstring = max(counts, key=counts.get)
    top_coloring = bits_to_coloring(top_bitstring)
    quantum_found_valid = is_valid_3_coloring(top_coloring)

    # Aggregate: fraction of shots landing on a valid coloring.
    valid_shots = sum(
        c for bs, c in counts.items() if is_valid_3_coloring(bits_to_coloring(bs))
    )
    total_shots = sum(counts.values())
    valid_fraction = valid_shots / total_shots

    print(f"Most frequent measured coloring: {top_coloring} "
          f"(valid 3-coloring: {quantum_found_valid})")
    print(f"Fraction of shots landing on a valid 3-coloring: {valid_fraction:.3f}")

    verified = (
        THREE_COLORABLE
        and quantum_found_valid
        and valid_fraction > 0.5  # Grover should strongly amplify valid states
    )

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
