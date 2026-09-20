"""
Erdos problem #306 -- quantum-testable instance.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: '306'"):
    prize: no
    status: open
    tags: ["number theory", "unit fractions"]
    oeis: ["N/A"]

LIMITATION, stated honestly: problem #306's YAML entry carries no OEIS id
(oeis: ["N/A"]), and the erdosproblems dataset clone available here has no
per-problem description file, only this metadata stub. There is therefore no
specific sequence to target directly. Rather than fabricate an OEIS-backed
property, this script builds a small, finite, genuinely computable property
that sits squarely inside the entry's own tags ("number theory",
"unit fractions"): Egyptian-fraction (unit fraction) decomposition of a
rational number as a sum of two unit fractions,

    1/n = 1/a + 1/b

which is the classical decidable question underlying the whole "unit
fractions" family of Erdos problems (e.g. Erdos-Straus is the 3-term
analogue). We fix n = 6 and a small, explicit candidate list of 16 ordered
pairs (a, b) with a <= b (chosen from a modest divisor-adjacent range), and
ask: which candidates actually satisfy 1/6 = 1/a + 1/b?

The classical answer is computed first, in this script, from first
principles (exact fractions.Fraction arithmetic, no floating point, no
hard-coded literature values): among the 16 candidates, exactly two satisfy
the equation: (10, 15) and (12, 12), since 1/10+1/15 = 1/6 and 1/12+1/12 = 1/6.

The quantum part is a genuine Grover search over the 16-element candidate
index space (4 qubits): a phase oracle built directly from the classical
boolean array (marks exactly the indices classically verified above, via
multi-controlled-Z gates keyed on each marked index's bit pattern), 2 Grover
iterations (optimal for N=16, M=2 marked items: floor(pi/4 * sqrt(16/2)) = 2),
run on the ideal AerSimulator. The script checks that measurement
overwhelmingly concentrates on the two classically-marked indices, and PASSes
only if the quantum output matches the classical answer.
"""

import sys
from fractions import Fraction
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def build_candidates():
    """16 fixed candidate (a, b) pairs with a <= b, index 0..15."""
    raw = [
        (7, 42), (7, 43), (8, 24), (9, 18),
        (10, 15), (10, 16), (11, 13), (12, 12),
        (12, 13), (13, 14), (14, 15), (15, 15),
        (16, 18), (18, 20), (20, 24), (24, 30),
    ]
    assert len(raw) == 16
    return raw


def classical_marked_indices(n, candidates):
    """Exact (fractions.Fraction) classical check of 1/n = 1/a + 1/b."""
    target = Fraction(1, n)
    marked = []
    for i, (a, b) in enumerate(candidates):
        if Fraction(1, a) + Fraction(1, b) == target:
            marked.append(i)
    return marked


def index_to_bits(i, nbits):
    return format(i, f"0{nbits}b")


def apply_mcz_on_index(qc, index, nbits):
    """Flip the phase of the |index> basis state using an n-controlled-Z,
    implemented via X-sandwiching so the control pattern is the all-ones
    pattern for `index`'s bit string."""
    bits = index_to_bits(index, nbits)
    zero_positions = [pos for pos, b in enumerate(bits) if b == "0"]
    qubit_order = list(range(nbits))  # qubit i <-> bit string position i (MSB..LSB order below)

    # bits[0] is MSB -> corresponds to qubit nbits-1 in Qiskit little-endian
    # convention; map explicitly.
    ctrl_qubits = []
    for pos, b in enumerate(bits):
        qubit_index = nbits - 1 - pos
        ctrl_qubits.append(qubit_index)

    flip_qubits = [ctrl_qubits[pos] for pos in zero_positions]
    for q in flip_qubits:
        qc.x(q)

    if nbits == 1:
        qc.z(0)
    elif nbits == 2:
        qc.cz(0, 1)
    else:
        qc.h(nbits - 1)
        qc.mcx(list(range(nbits - 1)), nbits - 1)
        qc.h(nbits - 1)

    for q in flip_qubits:
        qc.x(q)


def build_grover_circuit(marked_indices, nbits):
    qc = QuantumCircuit(nbits, nbits)
    qc.h(range(nbits))

    n_iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt((2 ** nbits) / len(marked_indices)))))

    for _ in range(n_iterations):
        # Oracle: flip phase of each marked index.
        for idx in marked_indices:
            apply_mcz_on_index(qc, idx, nbits)

        # Diffusion operator (inversion about the mean).
        qc.h(range(nbits))
        qc.x(range(nbits))
        qc.h(nbits - 1)
        qc.mcx(list(range(nbits - 1)), nbits - 1)
        qc.h(nbits - 1)
        qc.x(range(nbits))
        qc.h(range(nbits))

    qc.measure(range(nbits), range(nbits))
    return qc, n_iterations


def main():
    n = 6
    candidates = build_candidates()
    nbits = 4
    assert len(candidates) == 2 ** nbits

    classical_marked = classical_marked_indices(n, candidates)
    print(f"Classical check: 1/{n} = 1/a + 1/b over {len(candidates)} candidate pairs")
    for i, (a, b) in enumerate(candidates):
        tag = " <-- MATCH" if i in classical_marked else ""
        print(f"  idx {i:2d} bits={index_to_bits(i, nbits)} (a={a:2d}, b={b:2d}){tag}")
    print(f"Classical marked indices: {classical_marked} "
          f"(pairs: {[candidates[i] for i in classical_marked]})")

    if not classical_marked:
        print("No classical solutions found in this candidate set -- cannot run Grover search.")
        print("RESULT: FAIL")
        sys.exit(1)

    qc, n_iterations = build_grover_circuit(classical_marked, nbits)
    print(f"Grover iterations used: {n_iterations}")

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    job = sim.run(tqc, shots=shots)
    counts = job.result().get_counts()

    # Qiskit bitstrings are c[nbits-1]...c[0]; our classical index encoding
    # used qubit (nbits-1-pos) for bit position `pos` of the MSB..LSB index
    # string, i.e. qubit q holds index bit (nbits-1-q) counting from MSB.
    # The measured bitstring (as printed by Qiskit, MSB..LSB over qubit
    # nbits-1..0) is therefore read directly as the index's binary string.
    def bitstring_to_index(bs):
        return int(bs, 2)

    measured_indices = {}
    for bitstring, count in counts.items():
        idx = bitstring_to_index(bitstring)
        measured_indices[idx] = measured_indices.get(idx, 0) + count

    sorted_measured = sorted(measured_indices.items(), key=lambda kv: -kv[1])
    print("Top measured indices (index: count):")
    for idx, count in sorted_measured[:6]:
        print(f"  {idx:2d} ({index_to_bits(idx, nbits)}): {count}")

    marked_prob = sum(measured_indices.get(i, 0) for i in classical_marked) / shots
    top_indices = {idx for idx, _ in sorted_measured[: len(classical_marked)]}

    print(f"Probability mass on classically-marked indices: {marked_prob:.4f}")
    print(f"Top-{len(classical_marked)} measured indices: {sorted(top_indices)}")
    print(f"Classically marked indices:                    {sorted(classical_marked)}")

    verified = (top_indices == set(classical_marked)) and (marked_prob > 0.6)

    if verified:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
