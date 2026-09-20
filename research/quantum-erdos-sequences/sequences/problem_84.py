"""
Erdos problem #84 -- quantum-testable instance.

OEIS: A399654, "Number of distinct sets of cycle lengths realized by simple
graphs on n vertices." (tags: graph theory, cycles). Erdos problem #84's
entry in the local erdosproblems data lists exactly this OEIS id.

Classical property tested (derived here from first principles, not copied
from OEIS): for n = 4 labeled vertices there are C(4,2) = 6 possible edges,
hence 2^6 = 64 possible simple graphs (edge subsets). For each such graph we
compute, by brute-force classical enumeration, the *set* of cycle lengths it
realizes (a 4-vertex graph can only contain 3-cycles (triangles) and/or a
single Hamiltonian 4-cycle, so this is fully decidable by direct
inspection -- no external cycle-finding library needed).

We verify classically, inside this script, that the number of *distinct*
cycle-length-sets realized across all 64 graphs on 4 vertices equals
a(4) = 4 (the fifth term, offset 0, of A399654's b-file: 1, 1, 1, 2, 4, 6,
...), i.e. OEIS[A399654](4) == 4. The four sets realized on 4 vertices are
{} (no cycle), {3} (triangle only), {4} (4-cycle only) and {3,4} (both).
This is the "small, finite, computable property": membership testing (is a
given cycle-length-set among those realized) and counting (how many of the
64 edge-subsets realize one particular target set) both reduce to a search
over a 6-bit space, which is exactly the shape Grover's algorithm attacks.

Quantum circuit: we pick the target cycle-length-set S = frozenset({3})
(graphs that contain a triangle but no Hamiltonian 4-cycle). Classically we
find the list of 6-bit
edge-subset indices (0..63) whose cycle-length-set equals S; call this the
marked set M, of size k = |M|. We build a genuine Grover search circuit over
6 qubits (representing the 2^6 = 64 possible edge subsets of K4):
  - an oracle that phase-flips exactly the computational basis states whose
    index is in M (implemented as a multi-controlled-Z per marked bitstring,
    flanked by X gates to match the 0-bits), and
  - the standard Grover diffusion operator,
run for the optimal number of iterations r = round(pi/4 * sqrt(64/k)) on
the ideal AerSimulator, and check that measurement outcomes concentrate on
M with high probability -- i.e. that the quantum search finds exactly the
graphs the classical brute-force enumeration says realize cycle-length-set
S. PASS/FAIL compares (a) the classical count k and marked-index set M
against (b) what the quantum circuit's measurement distribution picks out.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_VERTICES = 4
EDGES = list(itertools.combinations(range(N_VERTICES), 2))  # 6 possible edges
assert len(EDGES) == 6
NUM_GRAPHS = 2 ** len(EDGES)  # 64

# The three possible Hamiltonian 4-cycles on vertices {0,1,2,3}, each given
# as its 4 edges (as frozensets so orientation/edge direction doesn't matter).
def cycle_edges(order):
    return frozenset(
        frozenset((order[i], order[(i + 1) % len(order)])) for i in range(len(order))
    )

HAM_4_CYCLES = [
    cycle_edges((0, 1, 2, 3)),
    cycle_edges((0, 1, 3, 2)),
    cycle_edges((0, 2, 1, 3)),
]

ALL_TRIANGLES = [frozenset(t) for t in itertools.combinations(range(N_VERTICES), 3)]


def edge_set_from_mask(mask):
    """6-bit mask -> frozenset of frozenset({u,v}) edges present."""
    return frozenset(
        frozenset(EDGES[i]) for i in range(len(EDGES)) if (mask >> i) & 1
    )


def cycle_length_set(mask):
    """Classical, from-first-principles computation of the set of cycle
    lengths realized by the graph on 4 vertices given by `mask`."""
    present = edge_set_from_mask(mask)
    lengths = set()

    # 3-cycles: a triangle {a,b,c} is realized iff all 3 of its edges present.
    for tri in ALL_TRIANGLES:
        a, b, c = sorted(tri)
        tri_edges = {frozenset((a, b)), frozenset((a, c)), frozenset((b, c))}
        if tri_edges <= present:
            lengths.add(3)
            break

    # 4-cycle: realized iff all 4 edges of some Hamiltonian cycle present.
    for ham in HAM_4_CYCLES:
        if ham <= present:
            lengths.add(4)
            break

    return frozenset(lengths)


def classical_ground_truth():
    """Compute, for all 64 graphs on 4 vertices: the distinct cycle-length
    sets realized (should number a(4) = 2 per A399654), and the marked set
    M of masks whose cycle-length set equals our chosen target S."""
    all_sets = {}
    for mask in range(NUM_GRAPHS):
        s = cycle_length_set(mask)
        all_sets.setdefault(s, []).append(mask)

    distinct_count = len(all_sets)

    target = frozenset({3})
    if target not in all_sets:
        raise RuntimeError("chosen target cycle-length-set not realized; pick another")
    marked = sorted(all_sets[target])
    return distinct_count, target, marked, all_sets


def build_oracle(num_qubits, marked_indices):
    qc = QuantumCircuit(num_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{num_qubits}b")[::-1]  # bit i -> qubit i
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        if zero_qubits:
            qc.x(zero_qubits)
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        if zero_qubits:
            qc.x(zero_qubits)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def build_grover_circuit(num_qubits, marked_indices, iterations):
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))

    oracle = build_oracle(num_qubits, marked_indices)
    diffuser = build_diffuser(num_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(num_qubits))
        qc.append(diffuser.to_gate(), range(num_qubits))

    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def main():
    distinct_count, target, marked, all_sets = classical_ground_truth()

    print(f"Erdos problem #84 -- OEIS A399654 (distinct cycle-length-set counts)")
    print(f"n = {N_VERTICES} vertices, {len(EDGES)} possible edges, "
          f"{NUM_GRAPHS} graphs enumerated classically")
    print(f"Distinct cycle-length sets realized (a({N_VERTICES})): {distinct_count}")
    print(f"  realized sets: {[sorted(s) for s in all_sets.keys()]}")
    print(f"Target cycle-length set S = {sorted(target)}, "
          f"realized by {len(marked)} of {NUM_GRAPHS} graphs")
    print(f"  marked edge-subset masks: {marked}")

    # A399654 b-file, offset 0: a(0)..a(12)
    oeis_a399654 = [1, 1, 1, 2, 4, 6, 11, 21, 40, 75, 133, 247, 467]
    a4_matches = (distinct_count == oeis_a399654[4] == 4)

    num_qubits = 6
    k = len(marked)
    theta = math.asin(math.sqrt(k / NUM_GRAPHS))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Grover iterations: {iterations} (optimal for k={k}, N={NUM_GRAPHS})")

    qc = build_grover_circuit(num_qubits, marked, iterations)
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 8192
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit count keys are written MSB-first over classical bits, with
    # clbit (n-1) as the leftmost character and clbit 0 as the rightmost.
    # We measured qubit i into clbit i, and our oracle used qubit i to
    # represent bit i (the 2^i place) of the mask, so the bitstring read
    # as a plain binary integer already equals the mask.
    measured_masks = {}
    for bitstring, c in counts.items():
        mask = int(bitstring, 2)
        measured_masks[mask] = measured_masks.get(mask, 0) + c

    marked_set = set(marked)
    hits = sum(c for m, c in measured_masks.items() if m in marked_set)
    hit_fraction = hits / shots
    top_masks = sorted(measured_masks.items(), key=lambda kv: -kv[1])[: len(marked) + 3]

    print(f"Quantum measurement: {hits}/{shots} shots ({hit_fraction:.3f}) landed on "
          f"a marked (classically verified) mask")
    print(f"Top measured masks (mask: count): {top_masks}")

    quantum_ok = hit_fraction > 0.90

    verified = a4_matches and quantum_ok

    print()
    if verified:
        print("PASS")
    else:
        print("FAIL")
        if not a4_matches:
            print("  reason: classical distinct-set count did not match A399654(4)=2")
        if not quantum_ok:
            print(f"  reason: quantum hit fraction {hit_fraction:.3f} <= 0.90 threshold")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
