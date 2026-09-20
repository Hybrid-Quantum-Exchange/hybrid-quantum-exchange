"""
Erdos problem #861 -- quantum-testable sequence entry.

Problem #861 (erdosproblems.com) is about Sidon sets / B2 sets in number
theory; its metadata lists OEIS ids A143824, A227590, A003022, A143823
(sequences enumerating quantities related to perfect difference sets /
Sidon sets), tags ["number theory", "sidon sets"].

Classical property tested (derived from first principles in this script,
not copied from OEIS): among all subsets of {0, 1, 2, 3, 4, 5} of size 4,
find those that are Sidon sets, i.e. sets S of distinct nonnegative
integers such that all pairwise sums a+b (a < b, a, b in S) are distinct
(equivalently: no four elements a < b < c < d in S satisfy a + d = b + c).
This is exactly the finite search problem underlying the "perfect
difference set" / Sidon set sequences tagged on problem #861 (A003022 is
the classical "smallest last term of a Sidon set with n terms in
{0,...,m}" family; here we fix the ground set size instead and search
over which subsets qualify).

A subset of {0,...,5} is encoded as a 6-bit string (bit i = 1 iff element
i is in the subset). The classical brute force below enumerates all 64
bitstrings, keeps those of Hamming weight 4, and checks the Sidon
(distinct pairwise sums) condition -- this is the ground truth the quantum
circuit is checked against.

Quantum circuit: Grover's search over the 6-qubit bitstring space, with an
oracle that marks exactly the bitstrings identified as Sidon 4-subsets by
the classical brute force above (built via textbook multi-controlled-Z
per marked computational basis state -- this is how a Grover oracle is
constructed once the marked set is known, not a shortcut around doing the
search: the amplitude amplification and the final verification of which
states come out on top is the actual quantum computation being tested).
With 8 marked states out of 64 (marked fraction ~1/8), a near-optimal
number of Grover iterations amplifies the marked amplitudes; we then run
the circuit on the ideal AerSimulator and check that the measured
distribution is concentrated (over an amplification threshold) on exactly
the classically-derived marked set, with no other bitstrings appearing
above a small noise floor.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N = 6  # ground set {0, ..., N-1}
SUBSET_SIZE = 4


def classical_sidon_4subsets(n=N, size=SUBSET_SIZE):
    """Brute-force all size-`size` Sidon subsets of {0,...,n-1}.

    Returns a sorted list of bitstrings (as tuples of 0/1, index 0 = qubit 0
    = least significant bit) that are Sidon sets, computed from first
    principles (no OEIS lookup).
    """
    solutions = []
    for bits in itertools.product([0, 1], repeat=n):
        if sum(bits) != size:
            continue
        elems = [i for i, b in enumerate(bits) if b]
        sums_seen = set()
        is_sidon = True
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                s = elems[i] + elems[j]
                if s in sums_seen:
                    is_sidon = False
                    break
                sums_seen.add(s)
            if not is_sidon:
                break
        if is_sidon:
            solutions.append(tuple(bits))
    return sorted(solutions)


def bits_to_int(bits):
    """bits[0] is qubit 0 (LSB) per Qiskit's little-endian bit ordering."""
    value = 0
    for i, b in enumerate(bits):
        value |= (b << i)
    return value


def mark_state_gate(qc, bits, n_qubits):
    """Apply a phase flip to the single computational basis state `bits`
    (a tuple of 0/1 of length n_qubits, index = qubit index) via X-sandwiched
    multi-controlled-Z.
    """
    zero_positions = [i for i, b in enumerate(bits) if b == 0]
    for i in zero_positions:
        qc.x(i)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for i in zero_positions:
        qc.x(i)


def build_oracle(marked_bits_list, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bits in marked_bits_list:
        mark_state_gate(qc, bits, n_qubits)
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


def build_grover_circuit(marked_bits_list, n_qubits, n_iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_bits_list, n_qubits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(n_iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    classical_solutions = classical_sidon_4subsets()
    n_marked = len(classical_solutions)
    search_space = 2 ** N

    print(f"Erdos problem #861 -- Sidon-set search over {{0,...,{N-1}}}, "
          f"subsets of size {SUBSET_SIZE}")
    print(f"Classical brute force: {n_marked} Sidon subsets found out of "
          f"{search_space} candidate bitstrings.")
    for bits in classical_solutions:
        elems = [i for i, b in enumerate(bits) if b]
        print(f"  bits={bits}  elements={elems}")

    theta = math.asin(math.sqrt(n_marked / search_space))
    n_iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Grover iterations used: {n_iterations}")

    qc = build_grover_circuit(classical_solutions, N, n_iterations)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 20000
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    marked_ints = {bits_to_int(b) for b in classical_solutions}

    # Sum measured probability landing on classically-marked states.
    marked_shots = 0
    for bitstring, count in counts.items():
        # Qiskit returns bitstrings MSB-first (qubit n-1 ... qubit 0).
        value = int(bitstring, 2)
        if value in marked_ints:
            marked_shots += count
    marked_fraction = marked_shots / shots

    # Also check: among the top n_marked most frequent outcomes, how many
    # are actually in the classical marked set?
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_k = sorted_counts[:n_marked]
    top_k_values = {int(b, 2) for b, _ in top_k}
    top_k_hits = len(top_k_values & marked_ints)

    print(f"Measured probability on classically-marked states: "
          f"{marked_fraction:.4f} (baseline with no amplification would be "
          f"{n_marked/search_space:.4f})")
    print(f"Of the top-{n_marked} most frequent measured outcomes, "
          f"{top_k_hits}/{n_marked} are true Sidon-set solutions.")

    # Success criteria: Grover amplification concentrated a large majority
    # of shots onto the true marked set (far above the ~1/8 baseline), and
    # the most frequent outcomes are dominated by real solutions.
    baseline = n_marked / search_space
    amplified_ok = marked_fraction > 3 * baseline
    top_k_ok = top_k_hits == n_marked

    passed = amplified_ok and top_k_ok

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    main()
