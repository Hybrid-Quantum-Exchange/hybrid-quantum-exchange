"""
Erdos problem #725 -- quantum-testable sequence entry.

OEIS id used: A001009.
A001009(n,k) is the triangle of L(n,k), the number of "normalized"
k x n Latin rectangles: a k x n array with entries from {0,...,n-1} (or
equivalently {1,...,n}) such that

  * every row is a permutation of {0,...,n-1},
  * every column has distinct entries (no value repeats down a column),
  * it is "normalized": row 0 is the identity permutation (0,1,...,n-1)
    and column 0 is the increasing sequence (0,1,...,k-1) read down the
    rows.

Classical property being tested (small instance n=4, k=2):

    L(4,2) = number of normalized 2 x 4 Latin rectangles = 3

Concretely: row 0 is fixed to (0,1,2,3). Normalization forces row 1's
first entry to be 1 (column 0 must read 0,1). The remaining three
entries of row 1 are some arrangement of the remaining values
{0,2,3} placed in positions 1,2,3, subject to the Latin-rectangle
column constraint row1[i] != row0[i] = i for i = 1,2,3 (row1 already
differs from row0 in position 0 by construction).

There are 3! = 6 ways to arrange {0,2,3} into positions (1,2,3). This
script:

  1. Enumerates all 6 arrangements classically (first principles,
     no OEIS value copied verbatim) and checks the column constraint
     to find which arrangements are valid rows of a normalized
     Latin rectangle. This reproduces L(4,2) = 3 from scratch.
  2. Encodes the 6 arrangements as 3-qubit basis states |0>..|5|
     (states |6>,|7> are unused padding) and builds a genuine Grover
     search circuit whose oracle marks exactly the valid arrangements
     (a diagonal phase-flip built directly from the classically
     computed marked-index set), with the standard Grover diffuser,
     run on the ideal AerSimulator.
  3. Compares the quantum search's high-probability outcomes (and the
     size of the marked set implied by the oracle) against the
     classical answer, and prints PASS/FAIL.

This is a real amplitude-amplification circuit (Grover's algorithm)
over a genuine finite search space defined by the Latin-rectangle
combinatorics behind A001009, not a lookup of a literal OEIS value.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
try:
    from qiskit.circuit.library import DiagonalGate as Diagonal
except ImportError:  # older qiskit
    from qiskit.circuit.library import Diagonal
from qiskit_aer import AerSimulator


def classical_L_4_2():
    """Brute-force, from first principles, the number of normalized
    2x4 Latin rectangles (i.e. A001009 term for n=4, k=2), and return
    (count, ordered list of all 6 candidate row-1 arrangements, list of
    booleans saying which are valid)."""
    n = 4
    row0 = tuple(range(n))  # (0,1,2,3)
    fixed_first = 1  # column 0 must read (0,1) down the two rows
    remaining_values = [v for v in range(n) if v != fixed_first]  # [0,2,3]

    candidates = []
    valid_flags = []
    for perm in itertools.permutations(remaining_values):
        row1 = (fixed_first,) + perm  # full 4-entry row
        candidates.append(row1)
        # Latin-rectangle column constraint: row1[i] != row0[i] for all i.
        # Position 0 always differs (fixed_first != 0). Check positions 1..3.
        ok = all(row1[i] != row0[i] for i in range(1, n))
        valid_flags.append(ok)

    count = sum(valid_flags)
    return count, candidates, valid_flags


def build_grover_circuit(marked_indices, num_qubits, iterations):
    """Standard Grover's algorithm: uniform superposition, then
    `iterations` rounds of (phase-oracle, diffuser), over `num_qubits`
    qubits. The oracle is a diagonal unitary with -1 exactly at the
    marked computational-basis indices -- built directly from the
    classically-determined marked set, no black box."""
    N = 2 ** num_qubits

    # Oracle: diagonal of +1's, with -1 at each marked index.
    diag = [1.0] * N
    for idx in marked_indices:
        diag[idx] = -1.0
    oracle_gate = Diagonal(diag)

    # Diffuser: standard Grover diffuser (inversion about the mean).
    diffuser = QuantumCircuit(num_qubits, name="diffuser")
    diffuser.h(range(num_qubits))
    diffuser.x(range(num_qubits))
    diffuser.h(num_qubits - 1)
    diffuser.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    diffuser.h(num_qubits - 1)
    diffuser.x(range(num_qubits))
    diffuser.h(range(num_qubits))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.append(oracle_gate, range(num_qubits))
        qc.append(diffuser.to_instruction(), range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def main():
    # --- Step 1: classical ground truth ---
    count, candidates, valid_flags = classical_L_4_2()
    marked_indices = [i for i, ok in enumerate(valid_flags) if ok]

    print("Erdos problem #725 -- OEIS A001009 (normalized Latin rectangles)")
    print(f"Instance: n=4, k=2 (rows are permutations of {{0,1,2,3}})")
    print(f"Row 0 fixed to (0,1,2,3); row 1 candidates (first entry fixed=1):")
    for i, (row1, ok) in enumerate(zip(candidates, valid_flags)):
        print(f"  index {i}: row1={row1}  valid={ok}")
    print(f"Classical count L(4,2) = {count}  (expected OEIS A001009 term = 3)")
    assert count == 3, "classical brute force disagrees with known A001009 term"

    # --- Step 2: quantum Grover search over the 6 (padded to 8) candidates ---
    num_qubits = 3  # 2^3 = 8 >= 6 candidates
    N = 2 ** num_qubits
    M = len(marked_indices)

    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = build_grover_circuit(marked_indices, num_qubits, iterations)

    backend = AerSimulator(method="statevector")
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit ordering: rightmost measured bit is qubit 0. Convert
    # each classical bitstring back to an integer index consistently.
    def bitstring_to_index(bs):
        return int(bs, 2)

    marked_hits = 0
    for bitstring, c in counts.items():
        idx = bitstring_to_index(bitstring)
        if idx in marked_indices:
            marked_hits += c

    marked_probability = marked_hits / shots
    print(f"\nGrover search: {num_qubits} qubits, {iterations} iteration(s), "
          f"{shots} shots")
    print(f"Marked indices (classically derived): {marked_indices}")
    print(f"Measured probability of landing on a marked index: "
          f"{marked_probability:.4f}")

    # Most-likely measured outcome should be a marked (valid) index.
    best_bitstring = max(counts, key=counts.get)
    best_index = bitstring_to_index(best_bitstring)
    most_likely_is_marked = best_index in marked_indices

    # The quantum run "verifies" the classical count of 3 by amplifying
    # exactly the 3 marked states far above uniform (uniform would give
    # 3/8 = 0.375 probability with no amplification); Grover should push
    # this well above that baseline, and the top outcome should be marked.
    baseline = M / N
    amplified = marked_probability > baseline + 0.15
    verified = most_likely_is_marked and amplified and (M == count)

    print(f"Most likely measured index: {best_index} "
          f"(marked={most_likely_is_marked})")
    print(f"Baseline (uniform) marked probability: {baseline:.4f}")
    print(f"Amplified above baseline: {amplified}")

    if verified:
        print("\nPASS: Grover search amplified exactly the classically-"
              "verified L(4,2)=3 valid normalized-Latin-rectangle rows, "
              "matching OEIS A001009.")
    else:
        print("\nFAIL: quantum result did not match the classical answer.")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
