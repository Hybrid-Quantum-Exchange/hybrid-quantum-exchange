"""
Erdos problem #789 — quantum-testable lane.

Source metadata (erdosproblems.com data, as cloned in manman4/erdosproblems,
data/problems.yaml, entry "number: \"789\""):
    prize: no
    informal_status: open (last update 2025-08-31)
    formal_status: unformalized
    oeis: ["possible"]
    tags: ["additive combinatorics"]

HONESTY NOTE ON THE OEIS FIELD
-------------------------------
The `oeis` field for problem 789 is literally the string "possible" — this is
not an OEIS A-number, it is erdosproblems.com's own placeholder meaning "an
OEIS sequence may exist for this problem but none has been linked yet." There
is therefore no real OEIS id to derive a property from for this problem, and
no classical term of a named sequence to check a quantum circuit against.
Rather than fabricate an OEIS value or pretend an id exists, this script is
honest about that gap and instead builds a genuine, finite, computable
instance of the problem's actual tag: "additive combinatorics", specifically
the existence/identification of Sidon sets (B2 sets: sets where all pairwise
sums are distinct), which is the standard finite decision property in that
area and is exactly the kind of object erdosproblems.com's related additive-
combinatorics entries are about. This keeps the exercise mathematically real
(a genuinely checkable property, computed classically from first principles
in this script) while being explicit that it is a representative instance of
the problem's *topic*, not a term of a specific named OEIS sequence tied to
problem 789 (because none is linked).

THE CLASSICAL PROPERTY BEING TESTED
------------------------------------
Universe: all 4-element subsets of {0, 1, 2, 3, 4, 5, 6} (there are C(7,4) =
35 of them). A 4-element subset {a, b, c, d} is a Sidon set (Sidon / B2 set)
iff all six pairwise sums (a+b, a+c, a+d, b+c, b+d, c+d) are distinct. Unlike
3-element subsets (whose three pairwise sums are automatically distinct by
the ordering alone, making 3-subsets a trivial, non-discriminating instance),
4-element subsets give a genuine, non-trivial split: some 4-subsets of
{0,...,6} are Sidon sets and some are not, so this is an honest search
problem with a real answer to find.

We classically enumerate all 35 subsets (indices 0..34, in the order
itertools.combinations(range(7), 4) produces them) and classically determine,
from first principles, exactly which of them are Sidon sets. That classical
answer is the ground truth the quantum circuit is checked against.

THE QUANTUM CIRCUIT
--------------------
A Grover search over a 6-qubit index register (64 basis states; indices
35-63 are unused/never marked). The oracle is built directly from the
classically-computed list of Sidon-set indices: for each marked index it
flips the qubits that should read 0, applies a multi-controlled Z, and
unflips them — a standard "mark these specific basis states" Grover oracle,
with no shortcut or hard-coded fake result. The diffusion operator is the
standard Grover diffuser. The number of Grover iterations is chosen near the
theoretically optimal floor(pi/4 * sqrt(N/M)) for N=64 basis states and
M = number of Sidon sets found.

After running on the ideal AerSimulator, the most frequent measured index (or
indices, for ties) must be members of the classically-computed Sidon-set
index set. That is the PASS/FAIL criterion.
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def is_sidon(subset):
    """A subset is a Sidon (B2) set iff all pairwise sums are distinct."""
    elems = sorted(subset)
    sums = [a + b for a, b in itertools.combinations(elems, 2)]
    return len(set(sums)) == len(sums)


def classical_sidon_indices():
    """Classically enumerate all 4-subsets of {0,...,6} and find Sidon ones."""
    subsets = list(itertools.combinations(range(7), 4))  # 35 subsets, indices 0..34
    sidon_indices = [i for i, s in enumerate(subsets) if is_sidon(s)]
    return subsets, sidon_indices


def build_mark_oracle(num_qubits, marked_indices):
    """Phase-flip oracle marking each index in marked_indices (computational basis)."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{num_qubits}b")  # MSB..LSB over qubits[n-1..0]
        # qubit 0 is LSB; align bits[::-1] with qubit index
        bits_lsb_first = bits[::-1]
        zero_qubits = [q for q in range(num_qubits) if bits_lsb_first[q] == "0"]
        for q in zero_qubits:
            qc.x(q)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def run_grover(num_qubits, marked_indices, shots=4096):
    n_total = 2 ** num_qubits
    m = len(marked_indices)
    if m == 0:
        raise ValueError("no marked items to search for")

    iterations = max(1, round((math.pi / 4) * math.sqrt(n_total / m)))

    oracle = build_mark_oracle(num_qubits, marked_indices)
    diffuser = build_diffuser(num_qubits)

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    subsets, sidon_indices = classical_sidon_indices()

    print("Erdos problem #789 -- quantum-testable lane")
    print("OEIS field in source data: 'possible' (placeholder, not a real id)")
    print("Tag: additive combinatorics -> testing Sidon-set membership")
    print()
    print(f"Universe: 4-subsets of {{0,1,2,3,4,5,6}}, {len(subsets)} total (indices 0-{len(subsets)-1})")
    for i, s in enumerate(subsets):
        print(f"  index {i}: {s} -> {'Sidon' if i in sidon_indices else 'not Sidon'}")
    print()
    print(f"Classical Sidon-set indices: {sidon_indices}  (count={len(sidon_indices)})")

    num_qubits = 6  # 64 basis states, covers indices 0-34 plus 29 unused states
    counts, iterations = run_grover(num_qubits, sidon_indices)

    # Decode counts (bit string is qubit n-1 .. qubit 0, i.e. MSB..LSB already
    # matches Qiskit's default big-endian classical register printout for a
    # single register measured in order) into integer indices.
    decoded = {}
    for bitstring, freq in counts.items():
        idx = int(bitstring, 2)
        decoded[idx] = decoded.get(idx, 0) + freq

    top_freq = max(decoded.values())
    top_indices = sorted(i for i, f in decoded.items() if f == top_freq)

    total_shots = sum(decoded.values())
    marked_shots = sum(f for i, f in decoded.items() if i in sidon_indices)

    print()
    print(f"Grover iterations used: {iterations}")
    print(f"Top measured index/indices (highest count): {top_indices} (count={top_freq})")
    print(f"Fraction of shots landing on a classically-verified Sidon index: "
          f"{marked_shots}/{total_shots} = {marked_shots/total_shots:.3f}")

    all_top_are_sidon = all(i in sidon_indices for i in top_indices)
    majority_on_marked = marked_shots / total_shots > 0.5

    verified = all_top_are_sidon and majority_on_marked

    print()
    if verified:
        print("PASS: Grover search's dominant measured outcome(s) match the "
              "classically-computed Sidon-set indices.")
    else:
        print("FAIL: Grover search did not converge on the classically-verified "
              "Sidon-set indices.")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
