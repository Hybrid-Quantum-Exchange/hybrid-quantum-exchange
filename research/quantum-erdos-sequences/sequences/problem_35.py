"""
Erdos problem #35 -- quantum-testable instance.

Source metadata (data/problems.yaml in the erdosproblems repo, entry
`number: "35"`): tags = ["number theory", "additive basis"], oeis = ["N/A"].
No OEIS sequence id is attached to problem #35 in the source data, so this
script cannot anchor a property to a specific OEIS entry. Per the task's
fallback instructions, this is the best honest attempt at a genuine quantum
circuit for the underlying mathematical object named by the tags (additive
bases / Sidon sets), with the limitation stated plainly: the link to
"problem 35" is via its tags only, not via an OEIS id (since none exists).

Classical property being tested
--------------------------------
A finite instance of the "additive basis" theme: Sidon sets (also called
B_2 sets), which are exactly the finite sets of integers whose pairwise
sums are all distinct -- the extremal/counting objects central to the
Erdos-Turan additive-basis literature that problem #35's tags point to.

Universe: elems = [0, 1, 2, 3] (4 elements, represented by 4 qubits, one
qubit per element indicating "element is in the subset").  For each of the
2**4 = 16 subsets, a subset with >= 2 elements is a Sidon set iff every
pairwise sum a+b (a < b, both in the subset) is distinct from every other
pairwise sum in that subset.

This script:
  1. Computes, from first principles in plain Python, the exact set of
     bitstrings (subsets of {0,1,2,3}, size >= 2) that are Sidon sets. This
     is the classical ground truth, "the classical answer".
  2. Builds a Grover search circuit over the 4-qubit space whose oracle
     phase-flips exactly the Sidon-set bitstrings (a real arithmetic
     property check baked into a fixed oracle derived from the classical
     computation -- not a copied literal value).
  3. Runs the circuit on the ideal AerSimulator, with the Grover iteration
     count chosen (also computed from first principles, from the standard
     Grover formula) to maximize the probability of measuring a marked
     (Sidon-set) bitstring.
  4. Compares the simulator's measured distribution against the classical
     Sidon-set list and prints PASS/FAIL.

Limitation: because problem #35 carries no OEIS id, the "sequence" tested
here is the classical enumeration of Sidon subsets of {0,1,2,3} that this
script derives itself, not a specific numbered OEIS sequence.
"""

import math
from itertools import combinations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_sidon_bitstrings(elems):
    """Return the set of bitstrings (subset of `elems`, size >= 2) that
    form a Sidon set (all pairwise sums distinct), computed directly."""
    n = len(elems)
    marked = []
    for bits in range(2 ** n):
        subset = [elems[i] for i in range(n) if (bits >> i) & 1]
        if len(subset) < 2:
            continue
        sums = []
        is_sidon = True
        for a, b in combinations(subset, 2):
            s = a + b
            if s in sums:
                is_sidon = False
                break
            sums.append(s)
        if is_sidon:
            marked.append(bits)
    return marked


def build_oracle(n_qubits, marked_bitstrings):
    """Phase-flip exactly the states in marked_bitstrings, via a
    multi-controlled-Z per bitstring (X-sandwich around 0-bits)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bits in marked_bitstrings:
        zero_positions = [i for i in range(n_qubits) if not (bits >> i) & 1]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z across all n_qubits (controls = first n-1, target = last)
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
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    elems = [0, 1, 2, 3]
    n_qubits = len(elems)
    N = 2 ** n_qubits

    marked = classical_sidon_bitstrings(elems)
    M = len(marked)
    marked_set = set(marked)
    marked_subsets = {
        bits: [elems[i] for i in range(n_qubits) if (bits >> i) & 1]
        for bits in marked
    }
    print(f"Universe elems = {elems}, N = {N} subsets, classical Sidon-set "
          f"(size>=2) bitstrings, M = {M}:")
    for bits in marked:
        print(f"  bits={bits:04b}  subset={marked_subsets[bits]}")

    # Optimal Grover iteration count computed from first principles, then
    # picked by directly maximizing sin^2((2t+1)*theta) over small t.
    theta = math.asin(math.sqrt(M / N))
    best_t, best_p = 0, 0.0
    for t in range(0, 6):
        p = math.sin((2 * t + 1) * theta) ** 2
        if p > best_p:
            best_p, best_t = p, t
    print(f"Chosen Grover iterations = {best_t} (predicted success "
          f"probability = {best_p:.4f})")

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)
    for _ in range(best_t):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit ordering: classical register bit 0 (elems[0]) is the
    # rightmost character of the returned bitstring.
    marked_shots = 0
    for bitstring, c in counts.items():
        bits_val = int(bitstring[::-1], 2)
        if bits_val in marked_set:
            marked_shots += c

    observed_p = marked_shots / shots
    print(f"Observed probability of measuring a Sidon-set bitstring: "
          f"{observed_p:.4f} ({marked_shots}/{shots} shots)")

    # Verification: the simulator's empirical probability of landing on a
    # classically-verified Sidon-set bitstring must be close to the
    # first-principles Grover prediction, and clearly above the uniform
    # baseline (M/N), demonstrating genuine amplitude amplification of the
    # classically-defined marked set.
    baseline = M / N
    tolerance = 0.08
    verified = (
        abs(observed_p - best_p) < tolerance
        and observed_p > baseline + 0.05
    )

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
