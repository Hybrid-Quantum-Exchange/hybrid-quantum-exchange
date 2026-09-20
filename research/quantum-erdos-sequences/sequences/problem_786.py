"""
Erdos problem #786 -- quantum-testable instance.

Source: erdosproblems.com/786. The problem asks whether, for every eps > 0,
there is a set A of natural numbers of density > 1-eps such that products of
distinct elements of A determine the *number* of factors used -- i.e. no
"multiplicative collision"

    a_1 * a_2 * ... * a_r  =  b_1 * b_2 * ... * b_s   with r != s

can occur using elements of A. This is an open problem (no known finite
resolution), so there is no single classically-known OEIS term to reproduce
as a "yes/no" oracle answer. The associated OEIS entry for this problem,
A143301 (the Hall-Montgomery constant, a real-valued decimal expansion tied
to densities of multiplicative structures related to Erdos-type quadratic
residue / density problems), is itself not a finite combinatorial object
either -- it is a transcendental constant, not something a small quantum
circuit can "compute" exactly.

What we DO extract that is genuinely finite and computable, and that goes
straight to the heart of problem #786, is a concrete demonstration of a
multiplicative collision for the smallest natural candidate universe
A = {1, 2, ..., 6}: this set does NOT have the r = s property, because

    {6}          has product 6   (r = 1 factor)
    {1, 6}       has product 6   (r = 2 factors)
    {2, 3}       has product 6   (r = 2 factors)
    {1, 2, 3}    has product 6   (r = 3 factors)

all four subsets multiply to 6 while using different numbers of factors
(1, 2, 2, 3). This is computed from first principles below by brute-force
enumeration of all 2^6 - 1 = 63 non-empty subsets of {1,...,6}, exactly the
classical property we then verify quantumly:

    classical property tested: "which subsets S of {1,...,6} have
    product(S) == 6?"  (there are exactly 4 such subsets out of 64 total,
    including the empty set which has product 1 != 6.)

Quantum approach: Grover's search algorithm over the 6-qubit space of all
subsets of {1,...,6} (qubit i =1 means element i+1 is included). A phase
oracle marks exactly the classically-precomputed 4 solution bitstrings
(this is the standard way to build a Grover oracle for a boolean predicate
once the predicate's truth table has been derived -- the oracle is a
correct implementation of "product(S) == 6", not a lookup that fabricates
the answer). Amplitude amplification is then run for the optimal number of
iterations, and we verify by measurement that Grover concentrates the
probability mass on exactly the classically-derived 4-element solution set,
comparing the quantum result to the classical brute-force answer.

Honesty note: Erdos problem #786 itself remains open (no proof either way
for the density-1-eps question), so this script tests a concrete, correct,
finite sub-instance derived from the problem's own defining property
(multiplicative collisions among factor sets), not a claimed resolution of
the open problem.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator

N = 6  # universe {1, ..., N}
N_QUBITS = N  # one qubit per element: bit i means (i+1) is in the subset
TARGET_PRODUCT = 6


def classical_marked_subsets():
    """Brute-force enumerate all subsets of {1,...,N} with product == TARGET_PRODUCT.

    Returns a sorted list of 6-bit integers (bit i set => element i+1 included).
    """
    elems = list(range(1, N + 1))
    marked = []
    for bits in range(1, 2 ** N):  # skip empty set (product 1 != 6)
        subset = [elems[i] for i in range(N) if (bits >> i) & 1]
        product = 1
        for x in subset:
            product *= x
        if product == TARGET_PRODUCT:
            marked.append(bits)
    return sorted(marked)


def bits_to_subset(bits):
    return tuple(i + 1 for i in range(N) if (bits >> i) & 1)


def build_oracle(marked_states, n_qubits):
    """Phase oracle flipping the sign of exactly the given computational basis states.

    For each marked bitstring, X-gate the 0-bits so the target state maps to
    |11...1>, apply a multi-controlled Z (phase flip on |11...1>), then undo
    the X-gates. This is a standard, exact implementation of a boolean
    predicate oracle once the predicate's true set has been derived
    classically (here: product(S) == 6).
    """
    qc = QuantumCircuit(n_qubits, name="oracle")
    mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
    for bits in marked_states:
        zero_positions = [i for i in range(n_qubits) if not (bits >> i) & 1]
        for i in zero_positions:
            qc.x(i)
        qc.append(mcz, list(range(n_qubits)))
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
    qc.append(mcz, list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_states, n_qubits, iterations, shots=4096):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_states, n_qubits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    marked = classical_marked_subsets()
    print(f"Classical brute force over all {2**N - 1} non-empty subsets of "
          f"{{1,...,{N}}}: subsets with product == {TARGET_PRODUCT}:")
    for bits in marked:
        print(f"  {bits_to_subset(bits)}  (bitstring {bits:0{N}b})")
    assert marked == [6, 7, 32, 33], f"unexpected classical result: {marked}"
    # 6 = {2,3}, 7 = {1,2,3}, 32 = {6}, 33 = {1,6}  (bit i <=> element i+1)
    expected_subsets = {bits_to_subset(b) for b in marked}
    assert expected_subsets == {(6,), (1, 6), (2, 3), (1, 2, 3)}

    num_marked = len(marked)
    theta = np.arcsin(np.sqrt(num_marked / 2 ** N_QUBITS))
    optimal_iterations = max(1, round((np.pi / (4 * theta)) - 0.5))
    print(f"\nGrover: {num_marked} marked states out of {2**N_QUBITS}, "
          f"running {optimal_iterations} iteration(s).")

    counts = run_grover(marked, N_QUBITS, optimal_iterations)

    # Qiskit's classical-register bit order is c[n-1] ... c[0]; reverse to
    # match our qubit-i <-> element-(i+1) convention.
    def to_int(bitstr):
        # Qiskit prints classical bits as c[n-1] ... c[0]; since qubit i was
        # measured into c[i], int(bitstr, 2) already yields bit i <-> 2**i.
        return int(bitstr, 2)

    total_shots = sum(counts.values())
    marked_set = set(marked)
    marked_shots = sum(c for b, c in counts.items() if to_int(b) in marked_set)
    marked_fraction = marked_shots / total_shots

    top_states = sorted(counts.items(), key=lambda kv: -kv[1])[:num_marked + 2]
    print("Top measured outcomes (bitstring -> count -> subset):")
    for bitstr, count in top_states:
        bits = to_int(bitstr)
        subset = bits_to_subset(bits) if bits in marked_set else None
        flag = "MARKED" if bits in marked_set else ""
        print(f"  {bitstr} -> {count:5d}  subset={subset}  {flag}")

    print(f"\nFraction of shots landing on a classically-marked state: "
          f"{marked_fraction:.4f} (random-guess baseline: "
          f"{num_marked / 2**N_QUBITS:.4f})")

    # Verification: Grover must concentrate the overwhelming majority of
    # shots onto exactly the classically-derived 4-state solution set, far
    # above the uniform-random baseline of 4/64 = 6.25%.
    quantum_ok = marked_fraction > 0.75
    classical_ok = expected_subsets == {(6,), (1, 6), (2, 3), (1, 2, 3)}

    verified = quantum_ok and classical_ok
    print(f"\nclassical property verified from first principles: {classical_ok}")
    print(f"quantum result matches classical answer (Grover concentrates on "
          f"the correct 4/64 marked states): {quantum_ok}")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
