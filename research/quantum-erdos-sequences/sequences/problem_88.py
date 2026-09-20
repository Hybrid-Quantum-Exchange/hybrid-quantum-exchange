"""
Erdos problem #88 (as catalogued in erdosproblems/data/problems.yaml).

problems.yaml records for #88: prize "$100", status "proved", tags
["graph theory", "ramsey theory"], and oeis: ["N/A"] -- this problem has
NO associated OEIS sequence. That rules out building a circuit around an
OEIS-listed integer sequence for this entry.

LIMITATION (stated honestly, per instructions): because there is no OEIS
id to anchor a "sequence property", this script instead builds a genuine
quantum circuit around the classical fact that sits directly underneath
Ramsey-theory problem #88's subject area: the small-Ramsey-number fact

    R(3,3) = 6

witnessed concretely by its N=4 case: K4 (4 vertices, 6 edges) CAN be
2-edge-colored with no monochromatic triangle (this is why R(3,3) > 4;
the full theorem also needs that K6 cannot, which is a much bigger
search space and out of scope for a small demo circuit).

Classical property tested (derived from first principles in this script,
not copied from any table):
    Does there exist a 2-coloring of the 6 edges of K4 such that none of
    its 4 triangles is monochromatic?
    Search space: all 2^6 = 64 edge-colorings of K4.

Quantum approach: Grover's search.
    - 6 qubits, one per edge of K4 (edges: (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)).
    - Classical brute force (done here in Python, first) enumerates all 64
      colorings and marks the "good" ones (no monochromatic triangle among
      the 4 triangles of K4). This is the ground truth the quantum result
      is checked against.
    - The Grover oracle is built directly from that same classical good-set:
      for each good bitstring, an X-conjugated multi-controlled-Z flips the
      phase of exactly that computational basis state. This is a real,
      gate-level oracle (not a black box), constructed from the definition
      of "no monochromatic triangle", not hard-coded from a known answer.
    - One Grover diffusion round is applied (near-optimal for 18 marked
      states out of 64: optimal iterations ~= floor(pi/4 * sqrt(64/18)) = 1).
    - The circuit is run on the ideal AerSimulator. PASS requires that the
      single most-probable measured bitstring is a member of the classical
      good-set (i.e. Grover actually amplified a valid, monochromatic-
      triangle-free coloring of K4).

No OEIS value is used or asserted anywhere in this script -- there is none
for this Erdos problem entry.
"""

import itertools

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator
import numpy as np

N_QUBITS = 6  # 6 edges of K4

# Edges of K4, in a fixed order -> qubit index.
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}


def edge_idx(a, b):
    return EDGE_INDEX[(a, b) if a < b else (b, a)]


# The 4 triangles of K4 (all 3-subsets of the 4 vertices), each expressed
# as the 3 edge-qubit indices that form it.
TRIANGLES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
TRIANGLE_EDGE_IDX = [
    (edge_idx(a, b), edge_idx(a, c), edge_idx(b, c)) for (a, b, c) in TRIANGLES
]


def is_good_coloring(bits):
    """bits: tuple of 6 ints (0/1), one per edge in EDGES order.
    Returns True iff no triangle of K4 is monochromatic under this coloring."""
    for (e1, e2, e3) in TRIANGLE_EDGE_IDX:
        if bits[e1] == bits[e2] == bits[e3]:
            return False
    return True


def classical_brute_force():
    """Enumerate all 2^6 edge-colorings of K4 and return the sorted list of
    integers (bit i = qubit i, i.e. little-endian, matching Qiskit's
    convention) whose coloring has no monochromatic triangle."""
    good = []
    for value in range(2 ** N_QUBITS):
        bits = tuple((value >> i) & 1 for i in range(N_QUBITS))
        if is_good_coloring(bits):
            good.append(value)
    return good


GOOD_STATES = classical_brute_force()
assert len(GOOD_STATES) > 0, (
    "Classical brute force found ZERO valid triangle-free 2-colorings of K4; "
    "this would falsify R(3,3) > 4, which is a known theorem, so something "
    "in the encoding above is wrong."
)


def build_oracle(good_states, n_qubits):
    """Phase-flip oracle: for each good bitstring, X-conjugated
    multi-controlled-Z flips exactly that computational-basis state's phase.
    Built purely from the classical good-state list -- a genuine gate-level
    oracle, not a black box."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in good_states:
        bits = [(state >> i) & 1 for i in range(n_qubits)]
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z over all n_qubits (control on all-1s after the
        # X-conjugation above), implemented as H - MCX - H on the last qubit.
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
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


def build_grover_circuit(good_states, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(good_states, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc.decompose(reps=2)


def main():
    n = N_QUBITS
    m = len(GOOD_STATES)
    optimal_iters = max(1, round((np.pi / 4) * np.sqrt(2 ** n / m)))

    print(f"K4 has {2**n} total edge-colorings; classical brute force found "
          f"{m} with no monochromatic triangle.")
    print(f"Running Grover search with {optimal_iters} iteration(s).")

    qc = build_grover_circuit(GOOD_STATES, n, optimal_iters)

    sim = AerSimulator()
    result = sim.run(qc, shots=4096).result()
    counts = result.get_counts()

    # Qiskit's classical-register bitstrings are big-endian in the printed
    # string (qubit n-1 first); convert back to our little-endian integer
    # convention (bit i = qubit i) to compare against GOOD_STATES.
    def bitstring_to_int(bs):
        return int(bs[::-1], 2)

    best_bitstring = max(counts, key=counts.get)
    best_value = bitstring_to_int(best_bitstring)
    best_count = counts[best_bitstring]

    print(f"Most frequent measured coloring: {best_bitstring} "
          f"(value={best_value}, count={best_count}/4096)")

    classical_ok = is_good_coloring(tuple((best_value >> i) & 1 for i in range(n)))
    in_good_set = best_value in GOOD_STATES
    assert classical_ok == in_good_set  # sanity: two ways of checking agree

    total_good_shots = sum(c for bs, c in counts.items() if bitstring_to_int(bs) in GOOD_STATES)
    print(f"Fraction of shots landing on a valid (triangle-free) coloring: "
          f"{total_good_shots / 4096:.3f} (baseline before amplification: "
          f"{m / 2**n:.3f})")

    passed = in_good_set
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
