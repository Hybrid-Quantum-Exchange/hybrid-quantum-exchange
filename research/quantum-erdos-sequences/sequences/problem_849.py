"""
Erdos problem #849 -- Singmaster's conjecture.

Problem #849 (see erdosproblems.com/849, "Singmaster's conjecture") asks
whether there is a uniform bound on the number of times an integer > 1 can
occur as a binomial coefficient C(n, k) in Pascal's triangle. The associated
OEIS sequences (per data/problems.yaml) include A003015 ("Numbers that occur
6 or more times in Pascal's triangle") and A003016 ("Least number that
occurs exactly n times in Pascal's triangle").

Classical property tested here (finite, computable, and checked from first
principles in this script, not copied from OEIS):

    For the fixed value TARGET = 6 and the finite grid of binomial
    coefficients C(n, k) with 0 <= k <= n < ROWS (ROWS = 8), find every
    index pair (n, k) with C(n, k) == TARGET.

    Classically this is: (n,k) in {(6,1), (4,2), (6,5)}, i.e. 6 occurs
    exactly 3 times in the first 8 rows of Pascal's triangle -- which is
    exactly why 6 is a member of A003015 (it occurs >= 6... no: A003015 is
    about occurring >=6 times over the *whole* triangle, not this bounded
    grid; here we only claim and verify the bounded-grid count of 3, which
    the script computes itself with math.comb, not by asserting OEIS values).

Quantum approach: encode (n, k) as a 6-qubit basis state (3 bits for n in
[0,8), 3 bits for k in [0,8), 64 states total). Build a genuine Grover
search circuit whose oracle marks exactly the basis states (n,k) satisfying
k <= n and C(n,k) == TARGET (the oracle is constructed from the classically
precomputed marked-state list -- this is the standard "database search"
formulation of Grover's algorithm: the oracle recognizes solutions by their
index, exactly as in unstructured search over N=64 items with M known
solutions). Run the ideal AerSimulator, take the optimal number of Grover
iterations for M=3 marked items out of N=64, and check that the top-M
measured basis states are exactly the classically computed marked (n,k)
pairs, and that they dominate the measurement distribution.

PASS/FAIL is decided by comparing the quantum measurement result to the
classical enumeration performed in this same script.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

ROWS = 8          # n ranges over 0..ROWS-1 (3 bits)
TARGET = 6        # value whose occurrences we count
N_BITS = 3         # bits per coordinate (n and k each in [0,8))
TOTAL_QUBITS = 2 * N_BITS  # 6 qubits, 64 basis states


def classical_marked_states():
    """Enumerate all (n, k) with 0 <= k <= n < ROWS and C(n,k) == TARGET."""
    marked = []
    for n in range(ROWS):
        for k in range(0, n + 1):
            if math.comb(n, k) == TARGET:
                marked.append((n, k))
    return marked


def state_to_index(n, k):
    """Pack (n, k) into a single 6-bit integer: high 3 bits = n, low 3 = k."""
    return (n << N_BITS) | k


def index_to_bits(idx, width):
    return [(idx >> i) & 1 for i in range(width)]


def build_oracle(marked_indices, n_qubits):
    """Phase oracle marking exactly the given basis-state indices."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_indices:
        bits = index_to_bits(idx, n_qubits)
        # Flip qubits that should be 0 so the target pattern becomes all-1s.
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            # Multi-controlled Z on qubit n_qubits-1 controlled by the rest.
            controls = list(range(n_qubits - 1))
            target = n_qubits - 1
            qc.h(target)
            qc.append(MCXGate(len(controls)), controls + [target])
            qc.h(target)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    target = n_qubits - 1
    controls = list(range(n_qubits - 1))
    qc.h(target)
    qc.append(MCXGate(len(controls)), controls + [target])
    qc.h(target)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_indices, n_qubits, shots=4096):
    N = 2 ** n_qubits
    M = len(marked_indices)
    if M == 0:
        raise ValueError("no marked states -- Grover search is undefined")

    # Optimal number of Grover iterations for N items, M solutions.
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(marked_indices, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    marked_pairs = classical_marked_states()
    marked_indices = sorted(state_to_index(n, k) for (n, k) in marked_pairs)
    classical_count = len(marked_pairs)

    print(f"Classical: (n,k) pairs in first {ROWS} rows with C(n,k) == {TARGET}:")
    for n, k in marked_pairs:
        print(f"  C({n},{k}) = {math.comb(n, k)}")
    print(f"Classical occurrence count: {classical_count}")

    counts, iterations = run_grover(marked_indices, TOTAL_QUBITS, shots=4096)
    print(f"Grover iterations used: {iterations}")

    # Qiskit bitstrings are c[n_qubits-1]...c[0]; our circuit qubit i holds
    # bit i of the packed index (LSB-first), and Qiskit prints MSB-first,
    # so bitstring[::-1] gives qubit order, from which we recover the index.
    def bitstring_to_index(bs):
        qubit_bits = bs[::-1]  # qubit_bits[i] = value of qubit i
        idx = 0
        for i, ch in enumerate(qubit_bits):
            if ch == "1":
                idx |= (1 << i)
        return idx

    index_counts = {}
    for bitstring, cnt in counts.items():
        idx = bitstring_to_index(bitstring)
        index_counts[idx] = index_counts.get(idx, 0) + cnt

    total_shots = sum(index_counts.values())
    ranked = sorted(index_counts.items(), key=lambda kv: -kv[1])
    top_m = ranked[:classical_count]
    top_m_indices = set(idx for idx, _ in top_m)
    marked_set = set(marked_indices)

    top_m_mass = sum(cnt for idx, cnt in top_m if idx in marked_set)
    marked_mass = sum(cnt for idx, cnt in index_counts.items() if idx in marked_set)

    print("Top measured basis states (index -> (n,k), shots):")
    for idx, cnt in top_m:
        n, k = idx >> N_BITS, idx & ((1 << N_BITS) - 1)
        print(f"  idx={idx:2d} (n={n},k={k}) shots={cnt} in_range={n < ROWS and k <= n}")

    recovered_correctly = top_m_indices == marked_set
    dominant = marked_mass / total_shots > 0.5  # marked states should dominate

    print(f"Recovered marked set matches classical set: {recovered_correctly}")
    print(f"Fraction of shots landing on a marked state: {marked_mass/total_shots:.3f}")

    verified = recovered_correctly and dominant

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
