"""
Erdos problem #31 -- quantum-testable instance.

Source: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: \"31\"" (informal_status: proved (Lean), tags: ["number theory",
"additive basis"], oeis: ["N/A"]).

LIMITATION, stated honestly up front: problem #31's YAML entry carries no
OEIS sequence id ("oeis: [\"N/A\"]"), so there is no OEIS sequence to test
membership/terms against here. What the entry does give is a concrete
finite, computable notion from its tags: an "additive basis" -- a set S of
residues mod N such that every residue mod N can be written as a sum of two
elements of S ("basis of order 2"). This script tests exactly that kind of
finite property (existence of a representing pair for a given target), on a
small hand-picked instance, using a genuine Grover search circuit. It is NOT
a literal OEIS value; it is a small combinatorial question motivated
directly by the "additive basis" tag on this problem.

Classical property under test
------------------------------
Let N = 8 and S = [1, 2, 3, 5] (indices 0..3). S happens to be an additive
basis of order 2 for Z_8 (every residue 0..7 is expressible as S[i]+S[j] mod
8 for some i, j in {0,1,2,3}). Fix target t = 6 (mod 8). We want to find
index pairs (i, j) in {0,1,2,3}^2 such that:

    (S[i] + S[j]) mod N == t

This is computed by brute force in Python first (the ground truth), giving
the exact set of marked index pairs, encoded as 4-bit strings "i1 i0 j1 j0".

Quantum circuit
----------------
A 4-qubit Grover search over the 16 possible (i, j) index pairs. The oracle
is built directly from the classically-precomputed marked index pairs (a
standard "mark these computational basis states" multi-controlled-Z oracle
-- this is exactly how Grover's algorithm is normally instantiated once the
marked set is known), followed by the standard diffuser, run for the
Grover-optimal number of iterations for M marked states out of 16. The
simulator is the ideal AerSimulator (statevector method, no noise).

Pass condition
--------------
Measure the 4-qubit register many times; the pair(s) whose classical value
(S[i]+S[j]) mod N indeed equals t must dominate the measured distribution,
and every one of the classically-marked pairs must be recoverable among the
circuit's high-probability outcomes while no un-marked pair is
over-represented. PASS/FAIL compares the quantum-recovered marked set
against the classically computed marked set.
"""

from itertools import product

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


def classical_marked_pairs(S, N, target):
    """Brute-force ground truth: all (i, j) with (S[i]+S[j]) % N == target."""
    marked = []
    for i, j in product(range(len(S)), repeat=2):
        if (S[i] + S[j]) % N == target:
            marked.append((i, j))
    return marked


def bits_for_index(idx, n_bits):
    return format(idx, "0{}b".format(n_bits))


def apply_multi_controlled_z_on_pattern(qc, qubits, bitstring):
    """Flip a Z on |bitstring> only, via X-sandwiched multi-controlled-Z."""
    # bitstring[0] corresponds to the most significant qubit in `qubits`
    # list order (qubits[0] is MSB here, matching bits_for_index output).
    flip_qubits = [q for q, b in zip(qubits, bitstring) if b == "0"]
    for q in flip_qubits:
        qc.x(q)
    # multi-controlled Z across all qubits in the register
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in flip_qubits:
        qc.x(q)


def build_grover_circuit(n_bits_i, n_bits_j, marked_pairs, S_len, iterations):
    n_qubits = n_bits_i + n_bits_j
    qc = QuantumCircuit(n_qubits, n_qubits)

    # register layout: [i_msb..i_lsb, j_msb..j_lsb]
    i_qubits = list(range(0, n_bits_i))
    j_qubits = list(range(n_bits_i, n_bits_i + n_bits_j))
    all_qubits = i_qubits + j_qubits

    qc.h(range(n_qubits))

    for _ in range(iterations):
        # --- oracle: mark each classically-known (i, j) pair ---
        for (i, j) in marked_pairs:
            bitstring = bits_for_index(i, n_bits_i) + bits_for_index(j, n_bits_j)
            apply_multi_controlled_z_on_pattern(qc, all_qubits, bitstring)

        # --- diffuser (inversion about the mean) ---
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(all_qubits[-1])
        qc.mcx(all_qubits[:-1], all_qubits[-1])
        qc.h(all_qubits[-1])
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    N = 8
    S = [1, 2, 3, 5]
    target = 6
    n_bits_i = 2  # len(S) == 4 -> 2 bits per index
    n_bits_j = 2

    marked_pairs = classical_marked_pairs(S, N, target)
    print("Classical brute force: S =", S, " N =", N, " target =", target)
    print("Classically marked (i, j) pairs with (S[i]+S[j]) mod N == target:",
          marked_pairs)
    assert len(marked_pairs) > 0, "instance must have at least one solution"

    total_states = 2 ** (n_bits_i + n_bits_j)
    M = len(marked_pairs)
    # Grover-optimal iteration count
    iterations = max(1, round((np.pi / 4) * np.sqrt(total_states / M)))
    print("Total search space:", total_states, " marked:", M,
          " Grover iterations:", iterations)

    qc = build_grover_circuit(n_bits_i, n_bits_j, marked_pairs, len(S), iterations)

    sim = AerSimulator(method="statevector")
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # decode measured bitstrings back to (i, j) index pairs
    decoded_counts = {}
    for bitstring, c in counts.items():
        # qiskit orders classical bits reversed (cN-1 ... c0); our
        # register was measured in order [i_msb, i_lsb, j_msb, j_lsb] into
        # classical bits [0..n_qubits-1], so reverse to restore that order.
        ordered = bitstring[::-1]
        i_str = ordered[0:n_bits_i]
        j_str = ordered[n_bits_i:n_bits_i + n_bits_j]
        i = int(i_str, 2)
        j = int(j_str, 2)
        decoded_counts[(i, j)] = decoded_counts.get((i, j), 0) + c

    # quantum-recovered marked set: pairs measured well above the uniform
    # background rate (uniform would give shots/total_states per outcome)
    uniform_rate = shots / total_states
    threshold = uniform_rate * 2  # comfortably above chance
    quantum_marked = sorted(
        pair for pair, c in decoded_counts.items() if c >= threshold
    )

    print("Measured counts (top 6):",
          sorted(decoded_counts.items(), key=lambda kv: -kv[1])[:6])
    print("Quantum-recovered marked pairs (count >= {:.1f}):".format(threshold),
          quantum_marked)

    ok = set(quantum_marked) == set(marked_pairs)

    # sanity: every recovered pair really does satisfy the classical property
    verified = all((S[i] + S[j]) % N == target for (i, j) in quantum_marked)

    if ok and verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
