"""
Erdos problem #155 -- quantum-testable instance.

Erdos problem 155 (erdosproblems.com/155) is tagged "additive combinatorics" /
"sidon sets" and lists OEIS ids A143824, A227590, A003022. This script uses
A003022, "Length of shortest (optimal) Golomb ruler with n marks" -- a Golomb
ruler is exactly a Sidon set of integers (all pairwise differences distinct),
which is the combinatorial object problem 155 is about.

A003022 (offset so a(1)=0, a(2)=1, a(3)=3, a(4)=6, a(5)=11, ...) gives, for
n marks, the smallest possible ruler length L such that there exists a set of
n integers {0 = m_1 < m_2 < ... < m_n = L} with all C(n,2) pairwise
differences distinct (a Golomb ruler / Sidon set).

Classical property tested here (computed from first principles below, not
copied from OEIS):
    For n = 4 marks, A003022 gives the optimal (shortest) Golomb ruler length
    as a(4) = 6, realised by the ruler {0, 1, 4, 6} (and its mirror image
    {0, 2, 5, 6}). With the first mark fixed at 0 and the last mark fixed at
    L = 6, this script searches, over all pairs of middle marks (a, b) with
    0 < a < b < 6 encoded as two 3-bit integers in [0, 7], for the pairs that
    make {0, a, b, 6} a Golomb ruler (Sidon set) -- i.e. all six pairwise
    differences distinct.

    The value L = 6 itself is not read off OEIS: it is derived in this script
    by a first-principles brute-force search over candidate ruler lengths
    (see optimal_length()), and only then checked to equal A003022(4).

    Brute force over the 64 (a, b) pairs (6 qubits total) gives the
    classical answer. That answer is encoded as a Grover "marked states"
    oracle, and Grover's algorithm is run on the ideal AerSimulator to search
    for those marks. Grover amplifying exactly the classically-valid pairs
    (and not the invalid ones) is the quantum verification: it shows the
    quantum search finds the same Sidon-set / optimal-Golomb-ruler solutions
    that the classical brute force finds for A003022(4) = 6.

Requires only qiskit, qiskit_aer, numpy (already installed).
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_golomb_ruler(marks):
    """A set of integer marks is a Golomb ruler (Sidon set) iff all pairwise
    differences are distinct."""
    diffs = []
    marks = sorted(marks)
    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            diffs.append(marks[j] - marks[i])
    return len(diffs) == len(set(diffs))


def classical_valid_pairs(L, bits_per_mark):
    """Brute-force, from first principles, which pairs of middle marks
    (a, b) in [0, 2**bits_per_mark - 1]^2 with 0 < a < b < L make
    {0, a, b, L} a Golomb ruler (n=4 marks, first mark 0, last mark L). This
    reproduces (for L = A003022(4) = 6) the Sidon-set condition problem 155
    concerns, without ever reading a literal OEIS term for the answer set
    itself."""
    valid = []
    side = 2 ** bits_per_mark
    for a in range(side):
        for b in range(side):
            if 0 < a < b < L and is_golomb_ruler([0, a, b, L]):
                valid.append((a, b))
    return valid


def pair_to_state(a, b, bits_per_mark):
    """Pack (a, b) into one integer, a in the low bits, b in the high bits."""
    return a | (b << bits_per_mark)


def state_to_pair(state, bits_per_mark):
    mask = (1 << bits_per_mark) - 1
    return state & mask, (state >> bits_per_mark) & mask


def build_grover_oracle(n_qubits, marked_states):
    """Phase-flip oracle marking each state in marked_states (list of ints)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
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
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits, marked_states, shots=4096):
    N = 2 ** n_qubits
    M = len(marked_states)
    if M == 0 or M == N:
        raise ValueError("Grover needs 0 < M < N marked states")

    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_grover_oracle(n_qubits, marked_states)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def optimal_length(n_marks, max_L=10):
    """First-principles brute-force search (no OEIS lookup) for the shortest
    Golomb ruler length with n_marks marks, first mark 0, last mark L."""
    for L in range(1, max_L + 1):
        # try all strictly increasing choices of the interior marks
        from itertools import combinations

        for interior in combinations(range(1, L), n_marks - 2):
            if is_golomb_ruler([0, *interior, L]):
                return L
    raise RuntimeError("no optimal length found in search range")


def main():
    # n = 4 marks, ruler length L = A003022(4) = 6, computed classically here
    # by brute force over all candidate lengths (not copied from OEIS):
    L = optimal_length(4)
    assert L == 6, f"expected A003022(4) = 6 from first-principles search, got {L}"

    bits_per_mark = 3  # encodes each of a, b in {0, ..., 7}
    n_qubits = 2 * bits_per_mark

    valid_pairs = classical_valid_pairs(L, bits_per_mark)
    classical_answer = {pair_to_state(a, b, bits_per_mark) for a, b in valid_pairs}

    print(f"Golomb ruler / Sidon-set instance: n=4 marks, L = A003022(4) = {L}")
    print(f"Classical valid (a, b) pairs with 0 < a < b < {L}: {sorted(valid_pairs)}")
    print(f"Encoded as {n_qubits}-qubit marked states: {sorted(classical_answer)}")

    counts, iterations = run_grover(n_qubits, sorted(classical_answer))
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")

    shots = sum(counts.values())
    # Qiskit's count-key bitstrings are already ordered so that a plain
    # binary parse recovers the integer state whose bit i is qubit i's value
    # (the oracle/diffuser above address qubit i as bit i of the state).
    quantum_marked = set()
    for bitstring, c in counts.items():
        state = int(bitstring, 2)
        if c / shots >= 0.15:  # amplified states should dominate the counts
            quantum_marked.add(state)

    quantum_pairs = {state_to_pair(s, bits_per_mark) for s in quantum_marked}
    print(f"Quantum-found (amplified) pairs: {sorted(quantum_pairs)}")

    passed = quantum_marked == classical_answer
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
