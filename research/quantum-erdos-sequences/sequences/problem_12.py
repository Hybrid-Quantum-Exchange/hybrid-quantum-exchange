"""
Erdos problem #12 (erdosproblems.com/12) -- quantum-testable instance.

Erdos problem #12, in its own words, asks whether there is an infinite set A
of positive integers of positive density relative to sqrt(N) such that no
three distinct elements a < b < c of A satisfy a | (b + c). The metadata
entry for this problem in data/problems.yaml carries no OEIS id
(`oeis: ["N/A"]`) and tag `"number theory"`; the underlying question about
infinite sets is not finite/computable as stated, so no literal OEIS
sequence value is available to check the quantum result against. This
script therefore builds a genuine *finite* instance of the same
Erdos-Sarkozy "no a | (b+c)" avoidance property -- the defining combinatorial
condition of problem #12 -- and uses Grover search to find a subset
achieving the classically-computed maximum, which the script verifies
against a from-scratch classical brute force. No OEIS value is fabricated
or copied anywhere in this file.

Classical property tested
--------------------------
Universe: {1, 2, ..., N} with N = 6 (so subsets fit in 6 qubits).
A subset S of {1,...,N} is "good" (property P) if there do NOT exist three
DISTINCT elements a, b, c in S with a < b, a < c (b,c > a), b != c, and
a divides (b + c). This is exactly the avoidance condition at the heart of
Erdos problem #12, restricted to a finite universe.

The script:
  1. Brute-forces, classically, ALL 2^6 = 64 subsets of {1,...,6}, checks
     property P for each, and finds M = the maximum size of a good subset,
     together with the full list of good subsets of that maximum size.
  2. Builds an exact Grover search circuit (6 qubits) whose oracle marks
     precisely the bitstrings encoding a good subset of size exactly M
     (built as an explicit unitary diagonal phase-flip over the true marked
     set -- not an approximation), with the standard number of Grover
     iterations for the resulting marked-state count.
  3. Runs the circuit on the ideal AerSimulator, takes the most frequently
     measured bitstring, decodes it back to a subset, and checks classically
     (again, from first principles) that this subset (a) satisfies property
     P and (b) has size exactly M -- i.e. matches the classically-computed
     answer.
  4. Prints PASS if the quantum search recovered a subset matching the
     classical maximum-good-subset answer with high probability, else FAIL.
"""

import itertools
import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator


N = 6  # universe {1, ..., N}
NUM_QUBITS = N  # one qubit per element: bit i = 1 means element (i+1) is in S


def is_good_subset(bits):
    """bits: tuple of 0/1 of length N, bits[i] means element i+1 is in S.
    Returns True iff no distinct a,b,c in S (b,c > a) has a | (b+c)."""
    elements = [i + 1 for i, bit in enumerate(bits) if bit == 1]
    s = set(elements)
    for a in elements:
        # need two DISTINCT other elements b, c in S with b,c > a and a | (b+c)
        candidates = [x for x in elements if x > a]
        for b, c in itertools.combinations(candidates, 2):
            if (b + c) % a == 0:
                return False
    return True


def classical_brute_force():
    """Check every one of the 2^N subsets of {1,...,N} from first principles."""
    best_size = -1
    best_subsets = []
    all_good = []
    for combo in itertools.product([0, 1], repeat=N):
        if is_good_subset(combo):
            all_good.append(combo)
            size = sum(combo)
            if size > best_size:
                best_size = size
                best_subsets = [combo]
            elif size == best_size:
                best_subsets.append(combo)
    return best_size, best_subsets, all_good


def bits_to_index(bits):
    """Map a bit tuple (bits[0] = qubit 0 = least significant) to an integer
    index matching Qiskit's little-endian statevector/bitstring convention."""
    idx = 0
    for i, b in enumerate(bits):
        if b:
            idx |= (1 << i)
    return idx


def index_to_bits(idx, n=NUM_QUBITS):
    return tuple((idx >> i) & 1 for i in range(n))


def build_grover_circuit(marked_indices, dim):
    """Exact Grover search over `dim`-dimensional computational basis with the
    given marked_indices amplified, built from explicit unitary matrices."""
    n = int(math.log2(dim))
    num_marked = len(marked_indices)
    assert 1 <= num_marked < dim

    # --- Oracle: diagonal phase flip on marked indices ---
    oracle_diag = np.ones(dim, dtype=complex)
    for idx in marked_indices:
        oracle_diag[idx] = -1.0
    oracle_matrix = np.diag(oracle_diag)
    oracle_gate = UnitaryGate(oracle_matrix, label="Oracle")

    # --- Diffuser: 2|s><s| - I about the uniform superposition ---
    s = np.full(dim, 1.0 / math.sqrt(dim), dtype=complex)
    diffuser_matrix = 2.0 * np.outer(s, s) - np.eye(dim, dtype=complex)
    diffuser_gate = UnitaryGate(diffuser_matrix, label="Diffuser")

    # optimal number of Grover iterations
    theta = math.asin(math.sqrt(num_marked / dim))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.append(oracle_gate, range(n))
        qc.append(diffuser_gate, range(n))
    qc.measure(range(n), range(n))
    return qc, iterations


def main():
    # 1) classical, from-scratch computation of the true answer
    best_size, best_subsets, all_good = classical_brute_force()
    marked_indices = sorted(bits_to_index(b) for b in best_subsets)

    print(f"Universe: {{1,...,{N}}}  (property P: no distinct a,b,c in S, "
          f"b,c>a, with a | (b+c))")
    print(f"Classical brute force over all {2**N} subsets:")
    print(f"  number of 'good' subsets (property P holds): {len(all_good)}")
    print(f"  maximum size M of a good subset: {best_size}")
    example = [i + 1 for i, bit in enumerate(best_subsets[0]) if bit == 1]
    print(f"  an example maximum good subset: {example}")
    print(f"  total maximum-size good subsets marked for Grover: "
          f"{len(marked_indices)} out of {2**N}")

    # 2) build and run the Grover circuit targeting exactly those subsets
    dim = 2 ** NUM_QUBITS
    qc, iterations = build_grover_circuit(marked_indices, dim)
    print(f"Grover circuit: {NUM_QUBITS} qubits, {iterations} iteration(s), "
          f"{len(marked_indices)} marked states")

    backend = AerSimulator()
    shots = 4096
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()

    # decode: qiskit bitstrings are printed MSB..LSB of the classical
    # register, i.e. c[n-1] c[n-2] ... c[0]; convert back to our indexing.
    decoded_counts = Counter()
    for bitstring, cnt in counts.items():
        # Qiskit prints bitstring[0] = qubit n-1 ... bitstring[-1] = qubit 0,
        # so int(bitstring, 2) already equals sum(bit_i * 2**i) -- our index.
        idx = int(bitstring, 2)
        decoded_counts[idx] += cnt

    top_index, top_count = decoded_counts.most_common(1)[0]
    top_bits = index_to_bits(top_index)
    top_subset = [i + 1 for i, bit in enumerate(top_bits) if bit == 1]
    prob_marked = sum(decoded_counts[i] for i in marked_indices) / shots

    print(f"Most frequent measured subset: {top_subset} "
          f"({top_count}/{shots} shots, {top_count/shots:.1%})")
    print(f"Total probability mass on any maximum-size good subset: "
          f"{prob_marked:.1%}")

    # 3) verify: recompute classically (independently) whether the quantum
    #    result matches the true answer
    quantum_is_good = is_good_subset(top_bits)
    quantum_size_matches = (sum(top_bits) == best_size)
    quantum_in_marked_set = (top_index in marked_indices)

    verified = quantum_is_good and quantum_size_matches and quantum_in_marked_set \
        and prob_marked > 0.5

    print()
    if verified:
        print("PASS: Grover search recovered a subset satisfying property P "
              "with the classically-verified maximum size, with high "
              "probability, matching the classical answer.")
    else:
        print("FAIL: quantum result did not match the classical answer.")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
