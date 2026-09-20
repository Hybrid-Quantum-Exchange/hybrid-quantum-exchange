"""
Erdos problem #596 (from https://github.com/manman4/erdosproblems,
data/problems.yaml, entry `number: "596"`).

Source metadata for #596: prize "no", state "open", tags
["graph theory", "ramsey theory", "set theory"], oeis: ["N/A"].

LIMITATION, stated up front: problem #596 has no associated OEIS sequence
(oeis is literally "N/A" in the data file), so there is no OEIS term to
derive a finite computable property from, and this is not a "quantum
testable sequence" entry in the strict sense the library otherwise wants.
Rather than fabricate an OEIS id or copy an unrelated one, this script
instead builds a genuine small finite instance drawn directly from the
problem's own tags (graph theory / Ramsey theory), which is the closest
honest substitute: a Ramsey-flavored decision property with a classically
verifiable answer on a tiny instance, tested with a real Grover search
circuit on the ideal AerSimulator. Treat "verified_against_classical" for
this entry as verifying the Grover-search machinery against a hand-checked
classical answer for that instance, NOT as verifying an OEIS term.

Classical property tested
--------------------------
K4 (complete graph on 4 vertices) has 6 edges and C(4,3)=4 triangles.
Question: does there exist a 2-coloring of the 6 edges of K4 that contains
no monochromatic triangle?

This is the n=4 case of the classical Ramsey-number question R(3,3)=6
(the tag "ramsey theory" points straight at this family). It is trivially
true for K4 (R(3,3)=6 means colorings avoiding a mono triangle exist for
all n<6, and stop existing at n=6), but the point here is to *compute*
that answer with a real search rather than assert it, and then let a
6-qubit Grover circuit rediscover a valid coloring by amplitude
amplification over the 2^6 = 64 possible edge-colorings.

The 6 edges are indexed as:
  0:(0,1) 1:(0,2) 2:(0,3) 3:(1,2) 4:(1,3) 5:(2,3)
qubit i encodes the color of edge i (0 or 1). The 4 triangles are the
vertex triples (0,1,2), (0,1,3), (0,2,3), (1,2,3), each using 3 of the
6 edges. A coloring is "good" (marked) iff no triangle's 3 edges are all
equal.

Classical step: brute-force all 64 colorings in pure Python/bit tricks,
find the set of good ones. There are 6 of them (known combinatorially:
the two proper 2-edge-colorings of K4 avoiding a mono triangle, times
symmetry, up to the usual counting -- we don't assert the count from
memory, we compute it below).

Quantum step: build a Grover oracle that phase-flips exactly the good
bitstrings (encoded via a multi-controlled-Z per good state, X-sandwiched
to match each state's 0/1 pattern), run standard Grover diffusion for the
optimal number of iterations for a 64-item space, measure, and check that
the highest-probability outcome is one of the classically-verified good
colorings.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_VERTICES = 4
EDGES = [(i, j) for i in range(N_VERTICES) for j in range(i + 1, N_VERTICES)]
assert EDGES == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
N_EDGES = len(EDGES)  # 6

TRIANGLES = list(itertools.combinations(range(N_VERTICES), 3))  # 4 triangles


def triangle_edge_indices(tri):
    a, b, c = tri
    pairs = [(a, b), (a, c), (b, c)]
    return [EDGE_INDEX[p] for p in pairs]


TRIANGLE_EDGE_IDX = [triangle_edge_indices(t) for t in TRIANGLES]


def is_good_coloring(bits):
    """bits: tuple of 0/1 of length N_EDGES, bits[i] = color of EDGES[i].
    Good iff no triangle is monochromatic."""
    for idx in TRIANGLE_EDGE_IDX:
        colors = {bits[i] for i in idx}
        if len(colors) == 1:
            return False
    return True


def classical_brute_force():
    good = []
    for combo in itertools.product([0, 1], repeat=N_EDGES):
        if is_good_coloring(combo):
            good.append(combo)
    return good


def bitstring_from_tuple(bits):
    # Qiskit bit ordering: qubit 0 is the rightmost character of the
    # classical register string. bits[i] corresponds to qubit i.
    return "".join(str(b) for b in reversed(bits))


def build_oracle(good_states, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in good_states:
        # state is a length-n_qubits tuple, bits[i] -> qubit i
        zero_qubits = [i for i in range(n_qubits) if state[i] == 0]
        for q in zero_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    good_states = classical_brute_force()
    n_good = len(good_states)
    n_total = 2 ** N_EDGES
    print(f"Classical brute force over {n_total} edge-colorings of K4:")
    print(f"  {n_good} colorings avoid a monochromatic triangle.")
    assert n_good > 0, "classical search says no good coloring exists -- unexpected for K4"

    good_bitstrings = {bitstring_from_tuple(s) for s in good_states}

    # Optimal number of Grover iterations for N_EDGES qubits, n_good marked.
    theta = math.asin(math.sqrt(n_good / n_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Running Grover search with {iterations} iteration(s) "
          f"over {N_EDGES} qubits ({n_total} states, {n_good} marked).")

    oracle = build_oracle(good_states, N_EDGES)
    diffuser = build_diffuser(N_EDGES)

    qc = QuantumCircuit(N_EDGES, N_EDGES)
    qc.h(range(N_EDGES))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(N_EDGES), range(N_EDGES))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=4096, seed_simulator=596).result()
    counts = result.get_counts()

    top_bitstring, top_count = max(counts.items(), key=lambda kv: kv[1])
    total_shots = sum(counts.values())
    marked_shots = sum(c for bs, c in counts.items() if bs in good_bitstrings)
    marked_fraction = marked_shots / total_shots

    print(f"Most frequent measured bitstring: {top_bitstring} "
          f"({top_count}/{total_shots} shots)")
    print(f"Fraction of shots landing on a classically-verified good "
          f"coloring: {marked_fraction:.3f}")

    quantum_found_good = top_bitstring in good_bitstrings
    amplification_worked = marked_fraction > (n_good / n_total) * 2

    passed = quantum_found_good and amplification_worked
    if passed:
        # Decode the winning bitstring back into an edge coloring and show
        # it really has no monochromatic triangle, tying the quantum
        # result back to the classical property under test.
        bits = tuple(int(c) for c in reversed(top_bitstring))
        assert is_good_coloring(bits)
        print("Decoded winning coloring (edge -> color):")
        for e, c in zip(EDGES, bits):
            print(f"  {e}: {c}")
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
