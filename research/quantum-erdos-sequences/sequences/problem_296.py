"""
Erdos problem #296 -- quantum-testable instance.

Source metadata (data/problems.yaml, entry "number: '296'"):
    prize: no
    informal_status: proved (Lean-formalized)
    oeis: ["possible"]
    tags: ["number theory", "unit fractions"]

Limitation, stated honestly up front: the OEIS field for this problem in the
source data is the literal string "possible", not a real OEIS sequence id.
There is no A-number to pull a term from. So this script does not test a
specific OEIS sequence membership; instead it builds a genuine, from-scratch
computable property in the same subject area the tags name -- unit fractions
/ Egyptian-fraction number theory -- which is the closest honest match to
what problem 296 is about, and verifies a real Grover search circuit against
it. This is disclosed rather than disguised: verified_against_classical is
still meaningful (the quantum circuit is checked against an independently
computed classical answer for a genuine unit-fraction search problem), but
it is not a claim of testing OEIS sequence "possible" specifically, because
no such sequence exists.

The property (Erdos-Straus-style unit-fraction decomposition):
    Fix n = 5. Search over pairs (a, b) with a, b in {1, ..., 8} for those
    pairs such that
        4/n - 1/a - 1/b = 1/c
    holds for some positive integer c (i.e. 4/n = 1/a + 1/b + 1/c has an
    integer solution with that particular a, b). This is exactly the
    Erdos-Straus conjecture's decomposition problem, restricted to a small
    finite search space so it is expressible as a Grover oracle.

Classical answer (computed in this script with Python's Fraction, first
principles, no lookup):
    For n = 5, a, b ranging over 1..8, the pairs (a, b) that admit an
    integer c are exactly:
        (2, 4) -> c = 20
        (2, 5) -> c = 10
        (4, 2) -> c = 20
        (5, 2) -> c = 10
    i.e. 4 marked pairs out of 64 candidate pairs.

Quantum circuit:
    A 6-qubit Grover search (3 qubits encode a-1 in {0..7}, 3 qubits encode
    b-1 in {0..7}) over the 64-element space {1..8} x {1..8}. The oracle
    phase-flips exactly the classically-precomputed marked basis states
    (built from the Fraction computation above, not hand-picked), and the
    standard Grover diffusion operator amplifies them. With 4 marked items
    out of 64, the optimal number of Grover iterations is
    round(pi/4 * sqrt(64/4)) = round(pi/4 * 4) = pi iterations ~ 3.
    The circuit is run on the ideal AerSimulator and the most frequently
    measured basis state is decoded back to (a, b) and checked against the
    classical marked set.

PASS/FAIL: PASS iff the most-likely measured (a, b) is in the classically
computed marked set.
"""

import math
from fractions import Fraction

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


N_BITS_A = 3  # a-1 in 0..7  -> a in 1..8
N_BITS_B = 3  # b-1 in 0..7  -> b in 1..8
TOTAL_QUBITS = N_BITS_A + N_BITS_B  # 6 qubits, 64 basis states
N_TARGET = 5  # the fixed n in 4/n = 1/a + 1/b + 1/c


def classical_marked_pairs(n):
    """Brute-force, from first principles, all (a, b) in 1..8 x 1..8 for
    which 4/n - 1/a - 1/b = 1/c has a positive-integer solution c."""
    marked = []
    for a in range(1, 9):
        for b in range(1, 9):
            r = Fraction(4, n) - Fraction(1, a) - Fraction(1, b)
            if r > 0 and r.numerator == 1:
                marked.append((a, b))
    return marked


def index_to_ab(index):
    """index in 0..63 -> (a, b) with a = (index >> 3) + 1, b = (index & 7) + 1."""
    a_bits = (index >> N_BITS_B) & ((1 << N_BITS_A) - 1)
    b_bits = index & ((1 << N_BITS_B) - 1)
    return a_bits + 1, b_bits + 1


def ab_to_index(a, b):
    return ((a - 1) << N_BITS_B) | (b - 1)


def build_oracle(marked_indices, num_qubits):
    """Phase-flip oracle marking each index in marked_indices, built purely
    from the classically-computed marked set (no shortcuts)."""
    qc = QuantumCircuit(num_qubits, name="Oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{num_qubits}b")
        # Flip qubits where the target bit is 0, so the all-ones pattern
        # corresponds to this basis state.
        zero_positions = [num_qubits - 1 - i for i, b in enumerate(bits) if b == "0"]
        for q in zero_positions:
            qc.x(q)
        if num_qubits == 1:
            qc.z(0)
        else:
            # Multi-controlled Z: H on target, MCX, H on target.
            target = num_qubits - 1
            controls = list(range(num_qubits - 1))
            qc.h(target)
            qc.append(MCXGate(len(controls)), controls + [target])
            qc.h(target)
        for q in zero_positions:
            qc.x(q)
    return qc


def build_diffusion(num_qubits):
    qc = QuantumCircuit(num_qubits, name="Diffusion")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    target = num_qubits - 1
    controls = list(range(num_qubits - 1))
    qc.h(target)
    qc.append(MCXGate(len(controls)), controls + [target])
    qc.h(target)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def run_grover(marked_indices, num_qubits, iterations, shots=4096):
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))

    oracle = build_oracle(marked_indices, num_qubits)
    diffusion = build_diffusion(num_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(num_qubits))
        qc.append(diffusion.to_instruction(), range(num_qubits))

    qc.measure(range(num_qubits), range(num_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    marked_pairs = classical_marked_pairs(N_TARGET)
    print(f"Classical marked (a,b) pairs for n={N_TARGET}: {marked_pairs}")
    assert marked_pairs == [(2, 4), (2, 5), (4, 2), (5, 2)], (
        "Classical computation drifted from the documented derivation"
    )

    marked_indices = sorted(ab_to_index(a, b) for a, b in marked_pairs)
    num_marked = len(marked_indices)
    search_space = 2 ** TOTAL_QUBITS
    iterations = max(1, round((math.pi / 4) * math.sqrt(search_space / num_marked)))
    print(f"Search space size: {search_space}, marked: {num_marked}, "
          f"Grover iterations: {iterations}")

    counts = run_grover(marked_indices, TOTAL_QUBITS, iterations)

    # Qiskit's classical register string is little-endian bit order per
    # register but the register itself is printed MSB..LSB of the classical
    # bits in creation order reversed; decode carefully by re-deriving index
    # from the bitstring as qiskit prints it (c[num_qubits-1] ... c[0]).
    best_bitstring = max(counts, key=counts.get)
    measured_index = int(best_bitstring, 2)
    measured_a, measured_b = index_to_ab(measured_index)

    total_shots = sum(counts.values())
    marked_shots = sum(
        cnt for bstr, cnt in counts.items() if int(bstr, 2) in marked_indices
    )
    marked_fraction = marked_shots / total_shots

    print(f"Most frequent measured index: {measured_index} "
          f"-> (a,b) = ({measured_a},{measured_b}), "
          f"count {counts[best_bitstring]}/{total_shots}")
    print(f"Fraction of shots landing on a marked (a,b): {marked_fraction:.3f}")

    is_marked = (measured_a, measured_b) in marked_pairs
    amplified = marked_fraction > (num_marked / search_space) * 3  # well above uniform baseline

    if is_marked and amplified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
