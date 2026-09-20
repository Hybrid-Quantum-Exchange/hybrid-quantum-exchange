"""
Erdos problem #46 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems.com mirror clone at
/home/user/manman4/erdosproblems/data/problems.yaml, entry "number: '46'"):
    tags: ["number theory", "unit fractions", "ramsey theory"]
    oeis: ["N/A"]

LIMITATION (report honestly): problem #46 has no associated OEIS sequence
("N/A" in the source data), so there is no OEIS term to verify against.
Per the task's fallback instructions, this script instead builds a genuine,
self-contained finite/computable instance drawn directly from the problem's
own tags -- unit fractions (Egyptian fractions) -- rather than fabricating
an OEIS-derived property or copying a literal OEIS value.

Classical property tested
--------------------------
Search space: all 2^5 = 32 subsets of the denominator set D = {2, 3, 4, 5, 6}.
Property: a subset S subset D is "marked" iff the sum of unit fractions
    sum_{d in S} 1/d == 1  (an exact Egyptian-fraction decomposition of 1).

This is computed from first principles below using Python's exact `fractions`
module (no external data, no hard-coded OEIS values). For D = {2,3,4,5,6}
there is exactly one marked subset: S = {2, 3, 6}, since 1/2 + 1/3 + 1/6 = 1.
(1/4 and 1/5 cannot participate in any exact decomposition of 1 within this D,
as direct enumeration below confirms.)

Quantum circuit
----------------
A 5-qubit Grover search circuit is built whose oracle marks exactly the
computational basis states corresponding to the classically-identified
marked subset(s) (a standard multi-controlled-Z oracle keyed off the
classical solution set -- this is the ordinary way to turn a classically
specified boolean predicate into a Grover oracle; the search itself, i.e.
amplitude amplification of the marked state(s) out of the uniform
superposition over all 32 candidates, is what the quantum circuit performs
and what is being verified, not the underlying arithmetic).  We run the
optimal number of Grover iterations for N=32, M=1 marked state, then sample
the resulting distribution on the ideal AerSimulator and check that the
marked state {2,3,6} is returned as the measured mode, matching the
classical answer.

PASS/FAIL: the script prints PASS iff the most frequently measured bitstring
decodes to the same subset the classical brute-force search found.
"""

from fractions import Fraction
from itertools import combinations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

DENOMS = [2, 3, 4, 5, 6]
N = len(DENOMS)  # 5 qubits, 2^5 = 32 subsets


def classical_marked_subsets():
    """Brute-force, from first principles, every subset of DENOMS whose
    reciprocals sum exactly to 1. Returns a sorted list of frozensets of
    denominators, and the corresponding list of 5-bit index integers
    (bit i set => DENOMS[i] included)."""
    marked = []
    for mask in range(1 << N):
        subset = [DENOMS[i] for i in range(N) if (mask >> i) & 1]
        if not subset:
            continue
        total = sum(Fraction(1, d) for d in subset)
        if total == 1:
            marked.append((mask, frozenset(subset)))
    return marked


def build_grover_circuit(marked_masks, n_qubits, iterations):
    """Standard Grover search circuit over n_qubits, with a phase oracle
    that flips the sign of exactly the basis states in marked_masks
    (given as little-endian integers, bit i <-> qubit i)."""
    qc = QuantumCircuit(n_qubits, n_qubits)

    # uniform superposition
    qc.h(range(n_qubits))

    def apply_oracle():
        for mask in marked_masks:
            # flip qubits that are 0 in this mask so the target state maps to |11...1>
            zero_bits = [i for i in range(n_qubits) if not (mask >> i) & 1]
            for i in zero_bits:
                qc.x(i)
            if n_qubits == 1:
                qc.z(0)
            else:
                qc.h(n_qubits - 1)
                qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
                qc.h(n_qubits - 1)
            for i in zero_bits:
                qc.x(i)

    def apply_diffuser():
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    for _ in range(iterations):
        apply_oracle()
        apply_diffuser()

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    # --- classical computation (first principles) ---
    marked = classical_marked_subsets()
    print("Classical brute-force search over all subsets of D =", DENOMS)
    for mask, subset in marked:
        terms = " + ".join(f"1/{d}" for d in sorted(subset))
        print(f"  found exact unit-fraction decomposition of 1: {terms} = 1  (subset {sorted(subset)})")

    assert len(marked) == 1, "expected exactly one marked subset for this instance"
    classical_mask, classical_subset = marked[0]
    print(f"Classical answer: unique marked subset = {sorted(classical_subset)}, "
          f"bitmask (bit i <-> denom DENOMS[i]) = {classical_mask:05b}")

    # --- quantum search over the same 32-element space ---
    M = len(marked)               # number of marked states
    n_qubits = N
    N_total = 1 << n_qubits
    optimal_iterations = max(1, round((np.pi / 4) * np.sqrt(N_total / M) - 0.5))
    print(f"Building Grover circuit: {n_qubits} qubits, N={N_total}, M={M}, "
          f"iterations={optimal_iterations}")

    marked_masks = [m for m, _ in marked]
    qc = build_grover_circuit(marked_masks, n_qubits, optimal_iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit reports bit strings with qubit (n-1) leftmost; convert back to
    # our little-endian mask convention (bit i <-> qubit i <-> DENOMS[i]).
    def bitstring_to_mask(bstr):
        mask = 0
        for i, bit in enumerate(reversed(bstr)):
            if bit == "1":
                mask |= (1 << i)
        return mask

    counts_by_mask = {}
    for bstr, c in counts.items():
        counts_by_mask[bitstring_to_mask(bstr)] = counts_by_mask.get(bitstring_to_mask(bstr), 0) + c

    top_mask = max(counts_by_mask, key=counts_by_mask.get)
    top_subset = frozenset(DENOMS[i] for i in range(N) if (top_mask >> i) & 1)
    top_prob = counts_by_mask[top_mask] / shots

    print(f"Quantum result: most probable measured subset = {sorted(top_subset)} "
          f"(probability {top_prob:.3f} over {shots} shots)")

    ok = (top_mask == classical_mask) and (top_prob > 0.5)

    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
