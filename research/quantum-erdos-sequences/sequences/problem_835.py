"""
Erdos problem #835 (from https://github.com/manman4/erdosproblems, data/problems.yaml)
tags: ["graph theory", "hypergraphs"]; oeis: ["N/A"]

LIMITATION, stated up front: problem #835's metadata carries no OEIS sequence
id ("N/A"). There is therefore no "quantum-testable sequence" in the sense
the rest of this library uses (an OEIS id whose membership/term property is
checked by a circuit). Rather than fabricate a fake OEIS-backed property, or
copy a literal sequence value with no derivation, this script instead builds
a genuine, small, computable property that sits squarely inside the same
mathematical territory as the problem's tags (hypergraph coloring / Property
B, a classic Erdos topic): 2-colorability of a small 3-uniform hypergraph
(does a red/blue vertex-coloring exist with no monochromatic hyperedge?).

Exact property tested (for the fixed instance below):
    Vertices V = {0,1,2,3}. Hyperedges (3-uniform, i.e. every hyperedge has
    exactly 3 vertices) E = [(0,1,2), (0,1,3), (1,2,3)].
    A coloring c: V -> {0,1} is "good" iff no hyperedge is monochromatic,
    i.e. for every edge (a,b,c) in E: NOT (c(a)==c(b)==c(c)).
    We ask: which of the 2^4 = 16 colorings are good, and is the hypergraph
    2-colorable (does at least one good coloring exist)?

Classical answer (computed here by brute force over all 16 colorings, from
first principles -- see classical_good_colorings()):
    The hypergraph IS 2-colorable. The set of good colorings, as 4-bit
    integers c0 + 2*c1 + 4*c2 + 8*c3 (bit i = color of vertex i), is computed
    below and printed. (By hand: any coloring that is not the 2
    monochromatic colorings 0000/1111 and does not make {0,1,2}, {0,1,3} or
    {1,2,3} monochromatic works; brute force nails the exact set.)

Quantum approach: Grover's algorithm.
    4 "vertex-color" qubits q0..q3 encode a candidate coloring. A quantum
    oracle built from the SAME Boolean logic as the classical check
    (an edge is bad iff its 3 color-bits are all equal, i.e. all 0 or all 1)
    phase-flips exactly the good colorings, using ancilla qubits to detect
    "all-equal" per edge (an edge is monochromatic iff bit_a XOR bit_b == 0
    AND bit_a XOR bit_c == 0). Grover diffusion amplifies those computational
    basis states. Enough Grover iterations are run (computed from the true
    count of good colorings, exactly as the standard Grover optimal-iteration
    formula prescribes) that measurement should return a good coloring with
    high probability.

PASS/FAIL: the script measures the circuit on the ideal AerSimulator, takes
the most frequent 4-bit outcome, and checks it against the classically
computed set of good colorings. It also cross-checks the full measured
distribution: the total probability mass landing on good colorings must
exceed a comfortable threshold (0.5), independently confirming amplification
actually happened rather than just getting lucky on the top count.
"""

import itertools
import sys

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

N_VERTICES = 4
EDGES = [(0, 1, 2), (0, 1, 3), (1, 2, 3)]  # 3-uniform hypergraph on 4 vertices


def is_edge_monochromatic(coloring, edge):
    a, b, c = edge
    return coloring[a] == coloring[b] == coloring[c]


def is_good_coloring(coloring):
    return not any(is_edge_monochromatic(coloring, e) for e in EDGES)


def classical_good_colorings():
    """Brute force over all 2^N_VERTICES colorings; returns the sorted list
    of integers (bit i = color of vertex i) that are 'good' (no
    monochromatic hyperedge)."""
    good = []
    for bits in itertools.product([0, 1], repeat=N_VERTICES):
        if is_good_coloring(bits):
            # bit i (vertex i) -> integer with bit i = bits[i]
            value = sum(b << i for i, b in enumerate(bits))
            good.append(value)
    return sorted(set(good))


def build_grover_circuit(good_colorings, n_iterations):
    """Build a Grover search circuit over N_VERTICES qubits whose oracle
    marks exactly the states in `good_colorings` (as computed classically),
    built from the same per-edge "all three colors equal" logic, expressed
    directly as quantum gates rather than as a black-box lookup table."""

    color = QuantumRegister(N_VERTICES, "color")
    # one ancilla per edge to flag "this edge is monochromatic"
    edge_anc = AncillaRegister(len(EDGES), "edge_bad")
    out = AncillaRegister(1, "oracle_out")
    creg = ClassicalRegister(N_VERTICES, "meas")

    qc = QuantumCircuit(color, edge_anc, out, creg)

    # uniform superposition over all colorings
    qc.h(color)

    # phase kickback ancilla for the oracle: |-> state so a controlled-X
    # from it flips phase of marked states
    qc.x(out[0])
    qc.h(out[0])

    def mark_oracle():
        # For each edge (a,b,c): edge is monochromatic iff a==b and b==c,
        # i.e. NOT(a XOR b) AND NOT(b XOR c). Compute that into edge_anc[i].
        for i, (a, b, c) in enumerate(EDGES):
            # a XOR b XOR c into a temp using CNOTs onto color[a] would
            # destroy input, so instead detect equality with X-CNOT-X trick
            # using a dedicated ancilla per edge:
            # edge_anc[i] = NOT(a XOR b) AND NOT(b XOR c)
            # Step 1: put (a XOR b) into edge_anc via CNOTs (reversible, we
            # uncompute after), then negate; similarly need a second temp
            # for (b XOR c). Use edge_anc[i] to accumulate via a small
            # temporary computed with Toffoli after two CNOT-based XOR
            # checks stored on ancilla qubits borrowed from 'out' register
            # is not available (only 1 ancilla there), so instead do this
            # directly with an MCX conditioned on both parity checks using
            # ancilla-free equality: for 3 same-value bits, a,b,c all equal
            # 0 or all equal 1. Equivalent condition:
            #   (a AND b AND c) OR (NOT a AND NOT b AND NOT c)
            # which we build with two Toffolis and a shared ancilla.
            qc.x(color[a])
            qc.x(color[b])
            qc.x(color[c])
            qc.mcx([color[a], color[b], color[c]], edge_anc[i])  # all-zero case
            qc.x(color[a])
            qc.x(color[b])
            qc.x(color[c])
            qc.mcx([color[a], color[b], color[c]], edge_anc[i])  # all-one case (OR via second mcx)

        # good coloring iff ALL edge_anc are 0 (no edge monochromatic)
        # flip out[0] iff all edge_anc bits are 0 -> use X on each ancilla,
        # MCX controlled on all being 1 (i.e. originally 0), then X back
        for i in range(len(EDGES)):
            qc.x(edge_anc[i])
        qc.mcx(list(edge_anc), out[0])
        for i in range(len(EDGES)):
            qc.x(edge_anc[i])

        # uncompute edge_anc (undo the two mcx's above, same self-inverse ops)
        for i, (a, b, c) in enumerate(EDGES):
            qc.x(color[a])
            qc.x(color[b])
            qc.x(color[c])
            qc.mcx([color[a], color[b], color[c]], edge_anc[i])
            qc.x(color[a])
            qc.x(color[b])
            qc.x(color[c])
            qc.mcx([color[a], color[b], color[c]], edge_anc[i])

    def diffuser():
        qc.h(color)
        qc.x(color)
        qc.h(color[-1])
        qc.mcx(list(color[:-1]), color[-1])
        qc.h(color[-1])
        qc.x(color)
        qc.h(color)

    for _ in range(n_iterations):
        mark_oracle()
        diffuser()

    # undo phase-kickback ancilla prep
    qc.h(out[0])
    qc.x(out[0])

    qc.measure(color, creg)
    return qc


def main():
    good = classical_good_colorings()
    total = 2 ** N_VERTICES
    print(f"Hypergraph: vertices={list(range(N_VERTICES))}, edges={EDGES}")
    print(f"Classical brute force: {len(good)} / {total} colorings are good (no monochromatic edge).")
    print(f"Good colorings (as ints, bit i = color of vertex i): {good}")

    if not good:
        print("Hypergraph is NOT 2-colorable for this instance -- no target for Grover. FAIL")
        sys.exit(1)

    # standard optimal Grover iteration count for M solutions out of N=2^n
    M = len(good)
    N = total
    theta = np.arcsin(np.sqrt(M / N))
    n_iterations = max(1, round((np.pi / (4 * theta)) - 0.5))
    print(f"Grover: N={N} states, M={M} marked (good) states -> {n_iterations} iteration(s).")

    qc = build_grover_circuit(good, n_iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=4096).result()
    counts = result.get_counts()

    # qiskit bit order: classical register string is c3 c2 c1 c0 (q3 first char)
    def bitstring_to_int(bs):
        bits = bs[::-1]  # reverse so index i corresponds to qubit i
        return int(bits, 2)

    dist = {}
    for bitstring, n in counts.items():
        val = bitstring_to_int(bitstring)
        dist[val] = dist.get(val, 0) + n

    total_shots = sum(dist.values())
    top_value, top_count = max(dist.items(), key=lambda kv: kv[1])
    good_mass = sum(c for v, c in dist.items() if v in good) / total_shots

    print(f"Most frequent measured coloring: {top_value} ({top_count}/{total_shots} shots)")
    print(f"Fraction of shots landing on a classically-good coloring: {good_mass:.3f}")

    top_is_good = top_value in good
    amplified = good_mass > 0.5

    ok = top_is_good and amplified
    print(f"top_value in classical good set: {top_is_good}; amplification (>0.5): {amplified}")

    if ok:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
