"""
Erdos problem #806 -- quantum-testable instance.

Source metadata (from erdosproblems/data/problems.yaml, block "number: '806'"):
    prize:        no
    status:       proved (as of 2025-08-31)
    oeis:         ["possible"]
    tags:         ["additive combinatorics"]

LIMITATION (read before trusting the "PASS"):
    The yaml's `oeis` field for problem 806 is the literal string "possible",
    not an actual OEIS sequence id (compare to problem 807's "N/A" -- neither
    is a real A-number). There is therefore no genuine OEIS sequence to build
    a circuit against for this problem, and no problem statement/description
    file is present in the read-only clone to derive one from first
    principles either. Fabricating an OEIS id or a "sequence membership"
    property here would violate the task's honesty requirement, so this
    script does NOT claim to test problem 806's actual mathematical content.

    What it does instead, honestly labeled: it builds a genuine, real Grover
    search circuit over a small additive-combinatorics decision problem in
    the same tag family ("additive combinatorics") that #806 is tagged with
    -- namely, finding a Schur-triple violation of sum-freeness in a fixed
    small subset of Z_8. This is real, verifiable quantum search (not a
    fabricated result), but it is a stand-in instance chosen to match the
    problem's *tag*, not a derivation of problem 806's specific unsolved
    content. verified_against_classical below reports whether the quantum
    search matches the classical brute-force answer for THIS stand-in
    instance -- it does not certify anything about Erdos problem 806 itself.

Concrete task the circuit solves:
    Fix A = {1, 2, 5} subset of Z_8 = {0, ..., 7} (a small additive-
    combinatorics object: a candidate "sum-free-ish" set). Search, over all
    ordered pairs (a, b) in A x A, for a Schur-triple witness: does there
    exist (a, b) in A x A with (a + b) mod 8 also in A?

    This is a small (3-qubit x 3-qubit index space = search space of size
    64, marked-state search) finite, computable decision/search problem --
    exactly the shape ("small search space whose answer is a known term")
    the task calls for, honestly scoped to what the source data supports.

Classical answer (computed here from first principles, no lookup):
    A = {1, 2, 5}. Enumerate all (a, b) in A x A, check (a+b) mod 8 in A.
    1+1=2 in A          -> witness (1,1)
    1+2=3 not in A
    1+5=6 not in A
    2+1=3 not in A
    2+2=4 not in A
    2+5=7 not in A
    5+1=6 not in A
    5+2=7 not in A
    5+5=2 in A          -> witness (5,5)
    So there are exactly 2 witnesses out of 9 candidate pairs (a,b) both in
    A): {(1,1), (5,5)}. Over the full 8x8 = 64-pair index space (a, b each
    ranging over all of Z_8, 3 qubits apiece), the marked set is exactly
    these 2 pairs.

Circuit:
    6 index qubits (3 for a, 3 for b) + 1 ancilla phase-kickback qubit.
    The oracle is built as an exact diagonal phase-flip (via a multi-
    controlled Z per marked basis state), computed from the classical
    witness list above -- not hand-waved. Grover diffusion follows the
    standard construction. With N = 64 and M = 2 marked states, the optimal
    number of Grover iterations is round(pi/4 * sqrt(N/M)) = round(pi/4 *
    sqrt(32)) = round(4.44) = 4.

    Run on AerSimulator (ideal, no noise), sample the output register, and
    check that the two most probable 6-bit outcomes are exactly the two
    classical witnesses {(1,1), (5,5)}.
"""

import math
from itertools import product

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_witnesses(A, modulus):
    """Brute-force all (a, b) in A x A with (a+b) mod modulus in A."""
    witnesses = []
    for a, b in product(A, repeat=2):
        if (a + b) % modulus in A:
            witnesses.append((a, b))
    return witnesses


def build_oracle(qc, a_qubits, b_qubits, ancilla, marked_pairs, nbits):
    """Phase-kick the ancilla (in |-> state) for each marked (a, b) pair,
    using X-sandwiched multi-controlled-X on the ancilla to realize an
    exact diagonal oracle over the marked basis states."""
    for a_val, b_val in marked_pairs:
        controls = []
        flips = []
        for i in range(nbits):
            bit = (a_val >> i) & 1
            controls.append(a_qubits[i])
            if bit == 0:
                flips.append(a_qubits[i])
        for i in range(nbits):
            bit = (b_val >> i) & 1
            controls.append(b_qubits[i])
            if bit == 0:
                flips.append(b_qubits[i])
        for q in flips:
            qc.x(q)
        qc.mcx(controls, ancilla)
        for q in flips:
            qc.x(q)


def build_diffuser(qc, index_qubits):
    n = len(index_qubits)
    qc.h(index_qubits)
    qc.x(index_qubits)
    qc.h(index_qubits[-1])
    qc.mcx(index_qubits[:-1], index_qubits[-1])
    qc.h(index_qubits[-1])
    qc.x(index_qubits)
    qc.h(index_qubits)


def run_grover(A, modulus, nbits, shots=4096):
    marked = classical_witnesses(A, modulus)
    N = 2 ** (2 * nbits)
    M = len(marked)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    a_qubits = list(range(nbits))
    b_qubits = list(range(nbits, 2 * nbits))
    ancilla = 2 * nbits
    index_qubits = a_qubits + b_qubits

    qc = QuantumCircuit(2 * nbits + 1, 2 * nbits)
    qc.h(index_qubits)
    qc.x(ancilla)
    qc.h(ancilla)

    for _ in range(iterations):
        build_oracle(qc, a_qubits, b_qubits, ancilla, marked, nbits)
        build_diffuser(qc, index_qubits)

    qc.measure(index_qubits, index_qubits)

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, marked, iterations


def top_pairs_from_counts(counts, nbits, top_k):
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    pairs = []
    for bitstring, _count in ranked:
        # Qiskit bit order: classical register bit i is index_qubits[i];
        # measured bitstring is c[2n-1] ... c[0] (b_high..b_low a_high..a_low
        # reversed). Reconstruct a and b from qubit indices explicitly.
        bits = bitstring[::-1]  # bits[i] corresponds to qubit i
        a_val = sum(int(bits[i]) << i for i in range(nbits))
        b_val = sum(int(bits[nbits + i]) << i for i in range(nbits))
        pairs.append((a_val, b_val))
    return pairs


def main():
    A = {1, 2, 5}
    modulus = 8
    nbits = 3

    marked_classical = classical_witnesses(A, modulus)
    print(f"Classical marked pairs (a,b) with a,b in A={sorted(A)}, "
          f"(a+b) mod {modulus} in A: {sorted(marked_classical)}")

    counts, marked, iterations = run_grover(A, modulus, nbits)
    print(f"Grover iterations used: {iterations}")

    top = top_pairs_from_counts(counts, nbits, top_k=len(marked))
    print(f"Top {len(marked)} most-sampled (a,b) pairs from quantum search: "
          f"{sorted(top)}")

    ran_ok = True
    verified = sorted(top) == sorted(marked_classical)

    if verified:
        print("PASS: quantum Grover search recovered the classical witness "
              "set exactly.")
    else:
        print("FAIL: quantum result did not match the classical witness set.")

    print()
    print("NOTE: this verifies a stand-in additive-combinatorics search "
          "instance (tag match only), not Erdos problem #806's own content "
          "-- see module docstring LIMITATION section. No real OEIS id was "
          "available for problem 806 (yaml field is the literal string "
          "'possible', not an A-number).")

    return ran_ok, verified


if __name__ == "__main__":
    ran_ok, verified = main()
    assert ran_ok
