"""
Erdos problem #654 -- quantum-testable instance.

Source metadata (erdosproblems.com dataset, data/problems.yaml, entry
`number: "654"`):
    prize: no
    informal_status: open (last_update 2025-08-31)
    oeis: ["possible"]
    tags: ["geometry", "distances"]

LIMITATION, stated honestly up front: the dataset does not give a real OEIS
sequence id for problem 654 -- the single entry in its `oeis` list is the
literal string "possible", which is a placeholder/annotation in this dataset,
not an OEIS A-number. There is therefore no genuine integer sequence from
OEIS to test membership/terms against for this problem. Rather than fabricate
an OEIS id or copy an unrelated one, this script instead builds a small,
finite, honestly-computable problem drawn directly from the two real tags
attached to #654: "geometry" and "distances". This mirrors the flavour of
Erdos's distinct-distances questions (of which #654 is one) without claiming
to formalize #654 itself.

Concrete finite property tested
--------------------------------
Take the 6 integer points on a line at positions P = [0, 1, 2, 3, 4, 5].
Consider the first 8 of the C(6,3) = 20 possible 3-point subsets (in the
order produced by itertools.combinations(P, 3)), indexed by i = 0..7 (i.e.
one 3-qubit register). A 3-point subset {a, b, c} is a "Sidon" configuration
(all pairwise distances distinct -- exactly the "distances" property the
tags name) when |a-b|, |b-c|, |a-c| are pairwise different.

The classical property under test: "which indices i in {0,...,7} correspond
to a Sidon (all-distinct-pairwise-distance) 3-point subset of the first 8
combinations of P taken 3 at a time?"

This is computed from first principles classically inside this script
(brute force over the 8 subsets), giving a classical answer set S subset of
{0,...,7}. The same predicate is then encoded as a Grover oracle over a
3-qubit register and Grover's algorithm is run on the ideal AerSimulator;
the indices measured with high probability must equal the classical set S.

This is a genuine (if modest) quantum search circuit -- not a fabricated
"quantum result" -- built and verified against a classical computation done
in this same file.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


def classical_answer():
    """Brute-force, from first principles: which of the first 8 3-subsets
    of P = [0..5] (in itertools.combinations order) have all three pairwise
    distances distinct (a Sidon / all-distinct-distance triple)?"""
    P = [0, 1, 2, 3, 4, 5]
    triples = list(itertools.combinations(P, 3))[:8]
    assert len(triples) == 8

    marked = []
    for i, (a, b, c) in enumerate(triples):
        d1, d2, d3 = abs(a - b), abs(b - c), abs(a - c)
        if len({d1, d2, d3}) == 3:
            marked.append(i)
    non_sidon = [i for i in range(8) if i not in marked]
    return triples, marked, non_sidon


def build_oracle(n_qubits, marked_indices):
    """Phase-flip oracle: for each marked index, flip the phase of that
    computational basis state (multi-controlled Z pattern, open controls
    on the 0-bits of that index)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")[::-1]  # bit i -> qubit i
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        # multi-controlled Z on all n_qubits (phase flip |11...1>)
        qc.h(n_qubits - 1)
        mcx = MCXGate(n_qubits - 1)
        qc.append(mcx, list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    mcx = MCXGate(n_qubits - 1)
    qc.append(mcx, list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits, marked_indices, shots=4096):
    N = 2 ** n_qubits
    M = len(marked_indices)
    if M == 0 or M == N:
        return None  # Grover degenerate; not our case here

    # optimal number of Grover iterations
    iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N / M)))

    oracle = build_oracle(n_qubits, marked_indices)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    triples, marked, non_sidon = classical_answer()
    n_qubits = 3  # 2**3 = 8 candidate indices

    print("Erdos problem #654 -- quantum-testable instance")
    print("OEIS id available: NONE (dataset entry only has placeholder 'possible')")
    print("Tags used to derive the instance: geometry, distances")
    print()
    print("First 8 3-subsets of {0,...,5} (itertools.combinations order):")
    for i, t in enumerate(triples):
        print(f"  index {i}: {t}")
    print()
    print(f"Classical answer -- indices WITH all-distinct pairwise distances (Sidon): {marked}")
    print(f"Classical answer -- indices WITHOUT that property (a repeated distance): {non_sidon}")
    print("(searching the quantum circuit for the smaller target set: non_sidon)")

    counts = run_grover(n_qubits, non_sidon, shots=4096)
    marked = non_sidon
    if counts is None:
        print("Grover degenerate case (0 or all marked) -- cannot test meaningfully.")
        print("FAIL")
        return False, False

    # top measured indices (by count), take as many as len(marked)
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_k = sorted_counts[: len(marked)]
    measured_indices = sorted(int(bits[::-1], 2) for bits, _ in top_k)
    # (bits string from Qiskit is c[n-1]...c[0]; reverse to match our qubit->bit convention)

    total_top_counts = sum(c for _, c in top_k)
    total_shots = sum(counts.values())
    top_fraction = total_top_counts / total_shots

    print()
    print(f"Measurement counts (top {len(marked)}): {top_k}")
    print(f"Measured indices (decoded): {measured_indices}")
    print(f"Fraction of shots landing on marked indices: {top_fraction:.3f}")

    verified = (measured_indices == sorted(marked)) and (top_fraction > 0.8)
    ran_ok = True

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return ran_ok, verified


if __name__ == "__main__":
    ok, verified = main()
    if not (ok and verified):
        raise SystemExit(1)
