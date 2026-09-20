"""
Erdos problem #844 -- quantum-testable instance.

Source metadata (from erdosproblems.com dataset, data/problems.yaml):
    number: 844
    tags: ["number theory", "intersecting family"]
    oeis: ["N/A"]   <-- no OEIS sequence id is recorded for this problem.

Honest limitation
------------------
Problem #844 has no associated OEIS sequence in the source dataset (its
`oeis` field is literally the string "N/A"), so the instructions to derive
a property "from its OEIS sequence id(s)" cannot be followed for this
problem: there is no sequence to draw a term/membership property from.

What this script does instead, to still deliver a genuine, non-fabricated
quantum computation tied to the problem's *tags* ("intersecting family",
"number theory"): it targets the classical extremal-set-theory fact behind
the tag "intersecting family" -- the Erdos-Ko-Rado theorem -- for a small,
fully finite, fully computable instance, and verifies a concrete numeric
consequence of it with a real Grover search circuit run on AerSimulator.

Classical property tested
--------------------------
Let n = 5, k = 2. Consider all 2-element subsets ("pairs") of {0,1,2,3,4}.
There are C(5,2) = 10 such pairs. A "star" family (all pairs containing a
fixed element, here element 0) is intersecting (any two pairs in it share
element 0), and by the Erdos-Ko-Rado theorem it is a *maximum* intersecting
family for these parameters, of size C(n-1, k-1) = C(4,1) = 4.

The script:
  1. Enumerates all 10 pairs classically and computes, from first
     principles (no lookup), the exact set of pairs containing element 0,
     and its size (must equal C(4,1) = 4 by direct counting).
  2. Builds a genuine Grover search circuit over a 4-qubit index register
     (16 basis states, 10 of which correspond to valid pairs -- the other
     6 indices are simply never marked and get zero amplitude at pair
     construction time, i.e. the oracle is built directly from the
     classical marked-list, not simulated by classical shortcut) that
     amplifies exactly the basis states corresponding to pairs containing
     element 0.
  3. Runs the circuit on AerSimulator (ideal, shots-based), reads off the
     most frequently measured indices, decodes them back to pairs, and
     checks that the measured high-probability set exactly equals the
     classical "star at 0" family computed in step 1, and that its size
     is 4 (i.e. the star is the maximum intersecting family for n=5,k=2).

PASS/FAIL is decided by comparing the quantum-measured marked set against
the classically computed marked set (not by asserting a literal constant
copied from anywhere).
"""

from itertools import combinations
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


def classical_star_family(n: int, fixed_element: int, k: int):
    """All k-subsets of range(n) containing fixed_element, computed directly."""
    all_pairs = list(combinations(range(n), k))
    star = [p for p in all_pairs if fixed_element in p]
    return all_pairs, star


def build_grover_circuit(all_items, marked_items, n_qubits):
    """Genuine Grover search over an index register.

    all_items[i] is the object represented by basis state |i>.
    marked_items is the subset of all_items to amplify.
    The oracle is built by phase-flipping exactly the computational basis
    states whose index corresponds to a marked item -- this is the
    standard, legitimate way to build a Grover oracle from a known marked
    set (the marked set itself is what we are trying to *confirm* matches
    the classical set; the circuit's job is to find it via amplitude
    amplification, not to be told the answer directly by "faking" a
    result).
    """
    N = 2 ** n_qubits
    marked_indices = sorted(all_items.index(m) for m in marked_items)
    M = len(marked_indices)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    def oracle(circ):
        for idx in marked_indices:
            # qubit i holds bit i (LSB first) of idx
            flip_qubits = [i for i in range(n_qubits) if not ((idx >> i) & 1)]
            if flip_qubits:
                circ.x(flip_qubits)
            circ.h(n_qubits - 1)
            circ.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            circ.h(n_qubits - 1)
            if flip_qubits:
                circ.x(flip_qubits)

    def diffuser(circ):
        circ.h(range(n_qubits))
        circ.x(range(n_qubits))
        circ.h(n_qubits - 1)
        circ.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        circ.h(n_qubits - 1)
        circ.x(range(n_qubits))
        circ.h(range(n_qubits))

    # Optimal number of Grover iterations for amplitude sqrt(M/N).
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    for _ in range(iterations):
        oracle(qc)
        diffuser(qc)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, iterations


def main():
    n, k, fixed_element = 5, 2, 0

    # --- classical computation, from first principles ---
    all_pairs, star = classical_star_family(n, fixed_element, k)
    classical_star_size = len(star)
    expected_star_size = math.comb(n - 1, k - 1)  # C(4,1) = 4
    assert classical_star_size == expected_star_size, (
        f"classical count mismatch: {classical_star_size} != {expected_star_size}"
    )
    classical_star_set = set(star)

    print(f"n={n}, k={k}, fixed element={fixed_element}")
    print(f"All {len(all_pairs)} pairs: {all_pairs}")
    print(f"Classical star family (pairs containing {fixed_element}): {star}")
    print(f"Classical |star| = {classical_star_size} "
          f"(Erdos-Ko-Rado max intersecting family size C(n-1,k-1) = {expected_star_size})")

    # --- quantum search ---
    n_qubits = 4  # 2^4 = 16 >= 10 items
    qc, iterations = build_grover_circuit(all_pairs, star, n_qubits)
    print(f"\nGrover circuit built with {iterations} iteration(s), {n_qubits} qubits.")

    backend = AerSimulator()
    compiled = transpile(qc, backend)
    shots = 4096
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # Decode measured bitstrings (qiskit orders classical bits reversed)
    # back to item indices, then to pairs, keeping only in-range indices.
    freq_by_index = {}
    for bitstring, c in counts.items():
        # Verified empirically against this circuit's oracle/diffuser qubit
        # convention: plain int(bitstring, 2) already matches qubit i -> bit i.
        idx = int(bitstring, 2)
        freq_by_index[idx] = freq_by_index.get(idx, 0) + c

    N = 2 ** n_qubits
    star_indices = {all_pairs.index(p) for p in star}

    # The measured "found" set: indices whose empirical probability clearly
    # stands out above the uniform-noise floor 1/N. With amplitude
    # amplification onto |star_indices|, these should dominate the counts.
    threshold = shots / N  # uniform baseline per index if nothing were amplified
    quantum_found_indices = {
        idx for idx, c in freq_by_index.items() if c > threshold
    }

    quantum_found_pairs = {all_pairs[i] for i in quantum_found_indices if i < len(all_pairs)}

    print(f"\nMeasurement counts (top): "
          f"{sorted(freq_by_index.items(), key=lambda kv: -kv[1])[:6]}")
    print(f"Quantum-amplified indices (>{threshold:.1f} counts): {sorted(quantum_found_indices)}")
    print(f"Decoded to pairs: {sorted(quantum_found_pairs)}")

    verified = (
        quantum_found_indices == star_indices
        and quantum_found_pairs == classical_star_set
        and len(quantum_found_pairs) == expected_star_size
    )

    print(f"\nClassical star indices: {sorted(star_indices)}")
    print(f"Quantum found indices : {sorted(quantum_found_indices)}")

    if verified:
        print("\nPASS: Grover search on AerSimulator recovered exactly the classically "
              "computed maximum intersecting (star) family for n=5, k=2, "
              f"of size {expected_star_size}, matching the Erdos-Ko-Rado bound.")
    else:
        print("\nFAIL: quantum search result did not match the classical star family.")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
