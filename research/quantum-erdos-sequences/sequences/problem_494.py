"""
Erdos problem #494 -- quantum-testable companion script.

Source metadata (from erdosproblems.com data, data/problems.yaml, entry
`number: "494"`):
    prize: no
    status: proved (last update 2025-10-14)
    tags: ["analysis", "additive combinatorics"]
    oeis: ["N/A"]

LIMITATION, stated up front and honestly: problem 494 carries no OEIS
sequence id in the source data (oeis: ["N/A"]). There is therefore no
concrete integer sequence attached to this problem to build a
membership/term-defining quantum oracle around, as the task instructions
ask for. Per the fallback instructions for this case, this script does
NOT fabricate an OEIS value or pretend a sequence exists. Instead it
builds the closest honest, genuinely-quantum artifact available: a real
Grover search circuit over a small, finite, computable instance of the
core notion in problem 494's own tags -- "additive combinatorics" -- an
area problem 494 is explicitly tagged with, even though the specific
statement of problem 494 is not itself computationally instantiated
here.

Classical property actually tested (finite, computable, checked in this
script from first principles, independent of any OEIS lookup):

    Let U = {1, 2, 3, 4}. A subset S subseteq U is "sum-free" if there
    is no solution to a + b = c with a, b, c in S (a and b need not be
    distinct, i.e. 2a = c also disqualifies S). We search the 2^4 = 16
    subsets of U for a sum-free subset of size >= 2.

    This script:
      1. Enumerates all 16 subsets of U classically and determines,
         by brute-force first principles (checking every a, b, c triple
         in each subset), exactly which subsets are sum-free of size
         >= 2. This is the classical ground truth.
      2. Builds a 4-qubit Grover search circuit whose oracle marks
         exactly those classically-determined sum-free subsets
         (encoded as computational basis states, bit i = 1 means
         element i+1 in U is in the subset), and whose diffuser is the
         standard Grover diffusion operator.
      3. Runs the circuit on the ideal AerSimulator, measures, and
         checks that the most frequently sampled bitstring decodes to
         a subset that is genuinely sum-free of size >= 2 (re-verified
         classically against the same first-principles check).
      4. Prints PASS if the quantum search's top outcome is a true
         sum-free subset (of size >= 2) of U, else FAIL.

This is a real (if generic) instance of a Grover search oracle built
from an explicitly, independently verified classical predicate -- not
a copied OEIS value -- offered as the best honest attempt given that
problem 494 itself has no attached OEIS sequence to target.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


UNIVERSE = [1, 2, 3, 4]
N = len(UNIVERSE)  # 4 qubits, bit i <-> UNIVERSE[i]


def is_sum_free(subset):
    """First-principles check: no a + b = c for a, b, c in subset."""
    s = set(subset)
    for a in s:
        for b in s:
            if (a + b) in s:
                return False
    return True


def classical_sum_free_subsets():
    """Brute-force, from first principles, every sum-free subset of
    UNIVERSE with size >= 2. Returns dict: bitstring -> subset."""
    marked = {}
    for bits in itertools.product([0, 1], repeat=N):
        subset = [UNIVERSE[i] for i in range(N) if bits[i] == 1]
        if len(subset) >= 2 and is_sum_free(subset):
            # qiskit bit ordering: qubit 0 is least-significant (rightmost)
            bitstring = "".join(str(b) for b in reversed(bits))
            marked[bitstring] = subset
    return marked


def build_oracle(marked_bitstrings, n_qubits):
    """Phase-flip oracle marking exactly the given basis states."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bitstring in marked_bitstrings:
        # bitstring[0] is qubit n-1 ... bitstring[-1] is qubit 0 (qiskit convention)
        zero_positions = [i for i, c in enumerate(reversed(bitstring)) if c == "0"]
        if zero_positions:
            qc.x(zero_positions)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
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


def build_grover_circuit(marked_bitstrings, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(marked_bitstrings, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    marked = classical_sum_free_subsets()
    print(f"Universe: {UNIVERSE}")
    print(f"Classically found {len(marked)} sum-free subsets (size >= 2) "
          f"out of {2 ** N} total subsets:")
    for bs, subset in sorted(marked.items()):
        print(f"  bitstring={bs} subset={subset}")

    assert marked, "No sum-free subsets found classically -- cannot search."

    M = len(marked)
    optimal_iterations = max(1, math.floor((math.pi / 4) * math.sqrt(2 ** N / M)))

    qc = build_grover_circuit(list(marked.keys()), N, optimal_iterations)
    qc_t = transpile(qc, AerSimulator())

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc_t, shots=shots).result()
    counts = result.get_counts()

    top_bitstring = max(counts, key=counts.get)
    top_count = counts[top_bitstring]
    top_prob = top_count / shots

    # Grover amplifies ALL M marked states together (there is no single
    # dominant basis state when M > 1), so the right check is the total
    # probability mass landing on the marked-solution set, not any one
    # bitstring's individual share.
    marked_mass = sum(counts.get(bs, 0) for bs in marked) / shots

    print(f"\nGrover iterations used: {optimal_iterations}")
    print(f"Top measured bitstring: {top_bitstring} "
          f"(count={top_count}/{shots}, prob={top_prob:.3f})")
    print(f"Total probability mass on the {len(marked)} marked solutions: "
          f"{marked_mass:.3f}")

    quantum_subset = [UNIVERSE[i] for i in range(N) if top_bitstring[N - 1 - i] == "1"]
    print(f"Decoded subset (top outcome): {quantum_subset}")

    # Re-verify classically, from first principles, independent of the
    # 'marked' dict used to build the oracle.
    classically_valid = len(quantum_subset) >= 2 and is_sum_free(quantum_subset)
    in_marked_set = top_bitstring in marked

    verified = classically_valid and in_marked_set and marked_mass > 0.9

    print(f"\nClassical re-check: size>=2 and sum-free = {classically_valid}")
    print(f"Matches oracle-marked set = {in_marked_set}")
    print(f"Marked-solution probability mass > 0.9 = {marked_mass > 0.9}")

    if verified:
        print("\nPASS")
    else:
        print("\nFAIL")


if __name__ == "__main__":
    main()
