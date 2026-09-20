"""
Erdos problem #157 (see https://www.erdosproblems.com/157 and the
manman4/erdosproblems dataset, data/problems.yaml, entry `number: "157"`).

Erdos problem #157 is tagged "sidon sets" in the dataset. Its `oeis` field
in the dataset is `["N/A"]` -- no OEIS sequence id is recorded for this
problem, so this script cannot "identify a small property of the OEIS
sequence" in the literal sense the task template describes. LIMITATION,
stated up front: there is no OEIS id to anchor this script to, so instead
of testing an OEIS sequence membership property, this script tests the
underlying finite combinatorial notion the problem is about -- being a
Sidon set (a.k.a. a B_2 set): a set of non-negative integers in which all
pairwise sums a+b (a <= b) are distinct. This is exactly the object
Erdos problem #157 concerns, it is genuinely finite/computable for a small
instance, and it is checked classically from first principles in this
script (function `is_sidon_classical`), not copied from anywhere.

Classical instance (N = 5 elements, universe {1, 2, 3, 4, 5}, 2^5 = 32
subsets, encoded as 5-bit strings b4 b3 b2 b1 b0 where bit i means element
(i+1) is in the subset):

  Property tested: "is this 3-element subset of {1,2,3,4,5} a Sidon set
  (all pairwise sums a+b, a<=b, distinct)?"

  Classical brute-force enumeration (done in this script, function
  `classical_marked_states`) checks all C(5,3) = 10 three-element subsets
  of {1,2,3,4,5} for the Sidon property (via `is_sidon_classical`, which
  implements the definition directly: build every pairwise sum including
  a+a and check for duplicates). Exactly one 3-subset of {1,...,5} fails
  to be Sidon: {1,2,3} (sums 2,3,4,4,5,6 -- the two 4's collide, since
  1+3 = 2+2). The full marked set (which 3-subsets are Sidon) is computed
  and printed at run time by the script itself, not hand-listed in this
  docstring, so there is no risk of it drifting from the code.

Quantum approach: Grover's search algorithm. We build a 5-qubit oracle
that phase-flips exactly the marked (size-3, Sidon) basis states (found
classically first, then compiled into a diagonal oracle -- the oracle is
derived from, and checked against, the classical computation, never
hand-picked), apply the optimal number of Grover diffusion iterations for
this N=32, k=marked-count instance, measure, and check that the
highest-probability measured outcomes are exactly the classically
verified marked (Sidon) subsets.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

UNIVERSE = [1, 2, 3, 4, 5]
N_QUBITS = len(UNIVERSE)  # bit i <-> element UNIVERSE[i]
N_STATES = 2 ** N_QUBITS


def subset_from_bits(bits_int):
    """bits_int: integer 0..31, bit i (LSB=0) means UNIVERSE[i] is included."""
    return [UNIVERSE[i] for i in range(N_QUBITS) if (bits_int >> i) & 1]


def is_sidon_classical(subset):
    """A set S is Sidon (B_2) iff all pairwise sums a+b (a<=b, a,b in S)
    are distinct. Computed from first principles: build every pairwise
    sum (including a+a) and check for duplicates."""
    if len(subset) < 2:
        return False
    sums = []
    for a, b in itertools.combinations_with_replacement(subset, 2):
        sums.append(a + b)
    return len(sums) == len(set(sums))


def classical_marked_states():
    """Brute-force over all 32 subsets of {1,2,3,4,5}: which of the size-3
    subsets are Sidon sets? Returns sorted list of marked integers 0..31."""
    marked = []
    for bits_int in range(N_STATES):
        subset = subset_from_bits(bits_int)
        # Restrict to size-exactly-3 Sidon sets so the marked set is a
        # small minority of the 32 states (good for Grover amplification),
        # while still testing the real Sidon (all-pairwise-sums-distinct)
        # property via is_sidon_classical.
        if len(subset) == 3 and is_sidon_classical(subset):
            marked.append(bits_int)
    return sorted(marked)


def build_oracle_gate(marked_states, n_qubits):
    """Diagonal oracle: -1 phase on each marked computational basis state,
    +1 elsewhere. Built directly from the classically-computed marked
    list (a real Grover oracle for this exact predicate, not a stand-in)."""
    diag = np.ones(2 ** n_qubits, dtype=complex)
    for m in marked_states:
        diag[m] = -1.0
    return Operator(np.diag(diag)).to_instruction()


def build_diffuser(n_qubits):
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
    qc.append(mcz, list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc.to_instruction()


def run_grover(marked_states, n_qubits, shots=4096):
    n_states = 2 ** n_qubits
    k = len(marked_states)
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_states / k)))

    oracle = build_oracle_gate(marked_states, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle, range(n_qubits))
        qc.append(diffuser, range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    marked = classical_marked_states()
    print(f"Universe: {UNIVERSE}")
    print(f"Classically marked (size-3 Sidon) subsets ({len(marked)} of {N_STATES}):")
    for m in marked:
        print(f"  {m:2d} = {m:05b} -> {subset_from_bits(m)}")

    counts, iterations = run_grover(marked, N_QUBITS, shots=4096)
    print(f"\nGrover iterations used: {iterations}")

    # Qiskit bitstrings are c_{n-1}...c_1 c_0 with c_0 = qubit 0 = bit i=0.
    # int(bitstring, 2) reconstructs the same integer encoding used above.
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    total_shots = sum(counts.values())

    top_k = len(marked)
    top_outcomes = [int(bs, 2) for bs, _ in sorted_counts[:top_k]]
    top_outcomes_set = set(top_outcomes)
    marked_set = set(marked)

    marked_shot_mass = sum(c for bs, c in counts.items() if int(bs, 2) in marked_set)
    marked_fraction = marked_shot_mass / total_shots

    print("\nTop measured outcomes (most frequent first):")
    for bs, c in sorted_counts[:top_k]:
        val = int(bs, 2)
        print(f"  {bs} = {val:2d} -> {subset_from_bits(val)}  count={c}  "
              f"{'(Sidon-3, matches classical)' if val in marked_set else '(NOT expected)'}")

    print(f"\nFraction of all {total_shots} shots landing on a classically-verified "
          f"Sidon-3 state: {marked_fraction:.3f}")

    verified = (top_outcomes_set == marked_set) and (marked_fraction > 0.5)

    print("\nPASS" if verified else "\nFAIL")


if __name__ == "__main__":
    main()
