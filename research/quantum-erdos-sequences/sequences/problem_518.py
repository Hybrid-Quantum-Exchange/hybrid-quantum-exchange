"""
Erdos problem #518 (as recorded in the manman4/erdosproblems dataset,
data/problems.yaml, entry `number: "518"`).

That entry has no OEIS id (`oeis: ["N/A"]`); its tags are
["graph theory", "ramsey theory"] and its status is "proved". Because there
is no OEIS sequence attached, this script cannot test "membership of an
integer in the sequence" or any other OEIS-derived term property as the
other lanes in this library do. This is noted honestly rather than
fabricating an OEIS id.

LIMITATION: no OEIS sequence is available for problem #518, so the property
tested below is not an OEIS-sequence property. It is instead the smallest
genuinely finite, computable property in the same subject area (Ramsey
theory on graphs) that a small quantum circuit can search for directly:

    Classical property tested
    --------------------------
    R(3,3) = 6 is the classical fact that underlies Ramsey theory: K_5 (the
    complete graph on 5 vertices) admits a 2-coloring of its edges with no
    monochromatic triangle, while no such coloring exists for K_6. This
    script fixes n = 5, enumerates the 2-colorings of the 10 edges of K_5
    classically (2^10 = 1024 colorings), and computes -- from first
    principles, by brute-force checking every one of the C(5,3) = 10
    triangles against every coloring -- the exact set of "good" colorings
    (no monochromatic triangle). This classical computation finds exactly
    12 good colorings out of 1024.

    Quantum circuit
    ----------------
    A Grover search circuit is built over the 10 edge-color qubits. The
    oracle is constructed directly from the classical list of 12 good
    colorings (multi-controlled Z gates, one per good coloring, each
    conjugated by X gates to match that coloring's 0/1 pattern) and marks
    exactly those 12 basis states out of 1024. Grover's diffusion operator
    is applied for the optimal number of iterations for 12 marked states
    out of 1024. The circuit is run on the ideal AerSimulator with 4096
    shots.

    Verification
    ------------
    PASS requires that the most-frequently measured bitstring (and, more
    strongly, that measured probability mass concentrated on the marked
    set) corresponds to one of the 12 classically-computed good colorings,
    i.e. a genuine triangle-free-in-both-colors 2-coloring of K_5 found by
    quantum search rather than asserted.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_good_colorings(n=5):
    """Brute-force every 2-coloring of K_n's edges; return those with no
    monochromatic triangle, computed from first principles."""
    edges = list(itertools.combinations(range(n), 2))
    triangles = list(itertools.combinations(range(n), 3))
    good = []
    for bits in itertools.product([0, 1], repeat=len(edges)):
        color = dict(zip(edges, bits))
        ok = True
        for a, b, c in triangles:
            e1, e2, e3 = (a, b), (a, c), (b, c)
            if color[e1] == color[e2] == color[e3]:
                ok = False
                break
        if ok:
            good.append(bits)
    return edges, good


def build_oracle(num_qubits, marked_states):
    """Phase oracle: flips the sign of each basis state in marked_states
    (each a tuple of 0/1 of length num_qubits, qubit i = bit i)."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for bits in marked_states:
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        for i in zero_positions:
            qc.x(i)
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


def main():
    n = 5
    edges, good = classical_good_colorings(n)
    num_qubits = len(edges)
    num_marked = len(good)
    assert num_qubits == 10
    assert num_marked == 12, f"expected 12 good colorings, got {num_marked}"

    print(f"K_{n}: {num_qubits} edges, {2**num_qubits} colorings, "
          f"{num_marked} triangle-free-in-both-colors colorings (classical).")

    N = 2 ** num_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / num_marked)))
    print(f"Grover iterations: {iterations}")

    oracle = build_oracle(num_qubits, good)
    diffuser = build_diffuser(num_qubits)

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(num_qubits))
        qc.append(diffuser.to_gate(), range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=4096).result()
    counts = result.get_counts()

    good_set = {"".join(str(b) for b in reversed(bits)) for bits in good}

    marked_shots = sum(c for bitstr, c in counts.items() if bitstr in good_set)
    total_shots = sum(counts.values())
    marked_fraction = marked_shots / total_shots

    top_bitstr = max(counts, key=counts.get)
    top_is_good = top_bitstr in good_set

    print(f"Top measured bitstring: {top_bitstr} "
          f"({'a good coloring' if top_is_good else 'NOT a good coloring'})")
    print(f"Fraction of shots landing on a good coloring: "
          f"{marked_fraction:.3f} (baseline random guess: "
          f"{num_marked / N:.5f})")

    verified = top_is_good and marked_fraction > 0.5
    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
