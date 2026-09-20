"""
Erdos problem #961 -- quantum-testable instance.

Source: erdosproblems.com problem 961 (data/problems.yaml entry
`number: "961"`, tags: ["number theory"], oeis: ["A213253"]).

OEIS A213253: a(n) = smallest k such that, for every m > n, the largest
prime factor of the product m(m+1)...(m+k-1) of k consecutive integers
exceeds n. (Such a k always exists by Sylvester's theorem: the product of
k consecutive integers greater than k has a prime factor greater than k.)
The sequence (offset 1) begins 1, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 6, ...
so a(3) = 3.

Classical property tested here (computed from first principles below, not
copied from OEIS): fix n = 3. Among all m > n, the "hardest" witness in
the range checked is m = 8: for k = 1 the product is 8 (largest prime
factor 2, not > 3); for k = 2 the product is 8*9 = 72 (largest prime
factor 3, still not > 3); for k = 3 the product is 8*9*10 = 720 (largest
prime factor 5, which IS > 3). So k = 3 is the smallest k for which the
product of k consecutive integers starting at m = 8 has a largest prime
factor exceeding n = 3, matching a(3) = 3 from A213253.

Quantum circuit: this "smallest good k" is recast as an unstructured
search over the 4 candidate values k in {1, 2, 3, 4} (2 qubits), with a
classically-precomputed oracle that marks exactly the unique k satisfying
   maxprimefactor(product of k consecutive ints from m=8) > n
   AND (k == 1 OR the same statement is false for k-1)
i.e. the index of the *smallest* k that first crosses the threshold. Only
k = 3 satisfies this, so the oracle marks exactly one of the 4 basis
states. Grover's algorithm with a single amplitude-amplification
iteration (optimal for 1 marked item out of 4) is run on the ideal
AerSimulator, and the most frequently measured index is compared against
the classically-computed answer.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def prime_factors(x: int):
    factors = []
    d = 2
    while d * d <= x:
        while x % d == 0:
            factors.append(d)
            x //= d
        d += 1
    if x > 1:
        factors.append(x)
    return factors


def max_prime_factor_of_run(m: int, k: int) -> int:
    """Largest prime factor of the product m*(m+1)*...*(m+k-1)."""
    product = 1
    for i in range(k):
        product *= (m + i)
    return max(prime_factors(product))


def classical_answer(n: int, m: int, k_values):
    """Smallest k in k_values (1-indexed candidates) whose run's largest
    prime factor exceeds n. Returns the 0-indexed position in k_values."""
    crossed = [max_prime_factor_of_run(m, k) > n for k in k_values]
    for idx, is_crossed in enumerate(crossed):
        if is_crossed and (idx == 0 or not crossed[idx - 1]):
            return idx, crossed
    raise ValueError("no crossing found in range (Sylvester guarantees one exists eventually)")


def main():
    n = 3
    m = 8
    k_values = [1, 2, 3, 4]  # -> 2 qubits, indices 0..3

    marked_idx, crossed = classical_answer(n, m, k_values)
    a_n = k_values[marked_idx]

    print(f"Erdos problem 961 / OEIS A213253, n={n}, m={m}")
    print(f"k values checked: {k_values}")
    print(f"largest prime factor > {n}? {crossed}")
    print(f"classical a({n}) (smallest crossing k) = {a_n} (index {marked_idx} of 4)")

    assert a_n == 3, "sanity check against known A213253 value a(3) = 3 failed"

    # --- Grover search over the 4 candidate indices for the unique marked one ---
    qc = QuantumCircuit(3, 2)  # qubits 0,1 = search register; qubit 2 = oracle ancilla
    qc.h([0, 1])

    # Ancilla in |-> for phase kickback oracle
    qc.x(2)
    qc.h(2)

    def apply_oracle(circuit, index):
        """Flip the ancilla's phase iff the 2-qubit register equals `index`."""
        bits = format(index, "02b")  # bit1 bit0, MSB first as qubit1 qubit0
        # bits[0] -> qubit1, bits[1] -> qubit0
        flip_targets = []
        if bits[0] == "0":
            circuit.x(1)
            flip_targets.append(1)
        if bits[1] == "0":
            circuit.x(0)
            flip_targets.append(0)
        circuit.ccx(0, 1, 2)
        if bits[0] == "0":
            circuit.x(1)
        if bits[1] == "0":
            circuit.x(0)

    def apply_diffuser(circuit):
        circuit.h([0, 1])
        circuit.x([0, 1])
        circuit.h(1)
        circuit.cx(0, 1)
        circuit.h(1)
        circuit.x([0, 1])
        circuit.h([0, 1])

    # One Grover iteration is optimal for 1 marked item among 4 (N=4, M=1):
    # iterations ~ floor(pi/4 * sqrt(N/M)) = floor(pi/4 * 2) = 1
    apply_oracle(qc, marked_idx)
    apply_diffuser(qc)

    qc.h(2)
    qc.x(2)

    qc.measure([0, 1], [0, 1])

    sim = AerSimulator()
    result = sim.run(qc, shots=2000).result()
    counts = result.get_counts()

    print(f"measurement counts: {counts}")

    # Classical register bit order in qiskit counts is c1c0 (qubit1 qubit0),
    # matching the same convention used in the oracle's `bits` string.
    best_bitstring = max(counts, key=counts.get)
    measured_idx = int(best_bitstring, 2)

    print(f"most frequent measured index: {measured_idx} (bitstring {best_bitstring})")
    print(f"classical marked index: {marked_idx}")

    quantum_k = k_values[measured_idx]
    print(f"quantum-found a({n}) = {quantum_k}, classical a({n}) = {a_n}")

    if measured_idx == marked_idx and quantum_k == a_n:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
