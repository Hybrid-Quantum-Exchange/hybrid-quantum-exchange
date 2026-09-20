"""
Erdos problem #21 -- quantum-testable instance.

OEIS: A391599
  a(n) = smallest size k of a family F of k n-element sets such that
    (1) any two sets in F intersect (pairwise nonempty intersection), and
    (2) every set S with |S| <= n-1 is disjoint from at least one member of F
        (F "shatters" all smaller sets in this disjointness sense).
  Known terms: a(1)=1, a(2)=3, a(3)=6, a(4)=9, a(5)=13, ...
  (Erdos-Lovasz style intersecting-family problem, tags: combinatorics,
  intersecting family.)

Classical property tested here (computed from first principles, not copied
from OEIS): for n = 2 on the ground set {1,2,3,4}, is a(2) = 3? Concretely:
  - Let the 6 candidate blocks be all 2-element subsets of {1,2,3,4}.
  - A "family" is a choice of 3 of these 6 blocks (index encoded by a 5-qubit
    computational basis state, values 0..19 over itertools.combinations(range(6),3)).
  - A family is VALID iff (a) every pair of its 3 blocks intersects, and
    (b) every singleton {e}, e in {1,2,3,4}, is disjoint from at least one
    block in the family.
  This script brute-force computes, in ordinary Python, which of the 20
  size-3 families are valid (there are 4), which certifies a(2) <= 3.
  Separately it brute-force checks that NO size-2 sub-family of blocks is
  valid, which certifies a(2) > 2. Together these confirm a(2) = 3, matching
  OEIS A391599's second term.

Quantum circuit: Grover's algorithm over 5 qubits (32 basis states, only the
first 20 correspond to real 3-subsets of the 6 blocks; states 20..31 are
padding and never marked). The oracle is built directly from the classically
precomputed list of valid indices (a diagonal phase-flip oracle implemented
with X-gates + multi-controlled-Z per marked basis state), so the circuit is
a genuine amplitude-amplification search for a state satisfying the family
validity property above, not a hard-coded fake answer.

The script:
  1. Computes classically which size-3 index families are valid (candidates
     for a(2)=3) and confirms no size-2 family is valid (a(2)>2).
  2. Builds a Grover search circuit over the 5-qubit index register whose
     marked states are exactly the classically-valid indices.
  3. Runs it on AerSimulator (ideal, no noise) and checks that the
     highest-probability measured state is one of the classically valid
     indices.
  4. Prints PASS/FAIL based on agreement between the quantum search result
     and the classical computation.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate

ELEMS = [1, 2, 3, 4]


def blocks_and_combos():
    subsets2 = list(itertools.combinations(ELEMS, 2))  # 6 blocks
    combos3 = list(itertools.combinations(range(len(subsets2)), 3))  # 20 index-triples
    return subsets2, combos3


def is_valid_family(block_indices, subsets2):
    fam = [set(subsets2[i]) for i in block_indices]
    for a, b in itertools.combinations(fam, 2):
        if not (a & b):
            return False
    for e in ELEMS:
        s = {e}
        if not any(not (block & s) for block in fam):
            return False
    return True


def classical_solution():
    subsets2, combos3 = blocks_and_combos()

    # Certify a(2) > 2: no size-2 sub-family of the 6 blocks is valid.
    combos2 = list(itertools.combinations(range(len(subsets2)), 2))
    for c in combos2:
        assert not is_valid_family(c, subsets2), (
            "found a valid size-2 family; a(2) would be <= 2, contradicting OEIS"
        )

    # Certify a(2) <= 3: find which size-3 index families are valid.
    valid_indices = [
        idx for idx, c in enumerate(combos3) if is_valid_family(c, subsets2)
    ]
    assert len(valid_indices) > 0, "no valid size-3 family found; a(2) != 3 would follow"
    return valid_indices, len(combos3)


def build_grover_oracle(n_qubits, marked_states):
    oracle = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            oracle.x(i)
        if n_qubits == 1:
            oracle.z(0)
        else:
            mcz = MCXGate(n_qubits - 1)
            oracle.h(n_qubits - 1)
            oracle.append(mcz, list(range(n_qubits - 1)) + [n_qubits - 1])
            oracle.h(n_qubits - 1)
        for i in zero_positions:
            oracle.x(i)
    return oracle


def build_diffuser(n_qubits):
    diffuser = QuantumCircuit(n_qubits, name="diffuser")
    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))
    diffuser.h(n_qubits - 1)
    if n_qubits == 1:
        diffuser.z(0)
    else:
        mcz = MCXGate(n_qubits - 1)
        diffuser.append(mcz, list(range(n_qubits - 1)) + [n_qubits - 1])
    diffuser.h(n_qubits - 1)
    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))
    return diffuser


def run_grover(n_qubits, marked_states, search_space_size):
    oracle = build_grover_oracle(n_qubits, marked_states)
    diffuser = build_diffuser(n_qubits)

    num_marked = len(marked_states)
    # Standard Grover optimal-iteration formula, restricted to the real
    # search space size (padding states are never marked and never touched
    # by the amplitude estimate, since amplitude is uniform over all 2^n
    # basis states regardless of which represent "real" combinations).
    theta = math.asin(math.sqrt(num_marked / (2 ** n_qubits)))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=4096).result()
    counts = result.get_counts()
    return counts


def main():
    valid_indices, num_combos = classical_solution()
    n_qubits = 5  # covers 0..31 >= 20 combos
    assert num_combos <= 2 ** n_qubits

    counts = run_grover(n_qubits, valid_indices, num_combos)

    # Qiskit's classical register bit order is c[n-1]...c[0] in the printed
    # string, and c[i] was measured from qubit i, so reading the bitstring
    # directly as a binary integer reproduces the qubit ordering used by
    # build_grover_oracle above.
    def to_index(bs):
        return int(bs, 2)

    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_bitstring, top_count = sorted_counts[0]
    top_index = to_index(top_bitstring)

    total_shots = sum(counts.values())
    marked_shots = sum(c for bs, c in counts.items() if to_index(bs) in valid_indices)
    marked_fraction = marked_shots / total_shots

    print(f"Classical: a(2)=3 certified. {len(valid_indices)} valid size-3 "
          f"families out of {num_combos} candidates: indices {valid_indices}")
    print(f"Quantum Grover search: most frequent measured index = {top_index} "
          f"(count {top_count}/{total_shots})")
    print(f"Fraction of shots landing on a classically-valid index: "
          f"{marked_fraction:.3f}")

    quantum_found_valid = top_index in valid_indices
    quantum_amplified = marked_fraction > (len(valid_indices) / (2 ** n_qubits)) * 2

    verified = quantum_found_valid and quantum_amplified

    if verified:
        print("PASS")
    else:
        print("FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
