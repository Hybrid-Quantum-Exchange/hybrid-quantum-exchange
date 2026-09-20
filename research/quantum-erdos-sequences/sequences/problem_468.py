"""
Erdos problem #468 (erdosproblems.com), OEIS A167485.

A167485(n) = "smallest positive integer m such that n can be expressed as
the sum of an initial subsequence of the (increasingly sorted) divisors of
m, or 0 if no such m exists."  I.e. list the divisors of m in increasing
order d_1 < d_2 < ... < d_k, form the cumulative sums s_1 = d_1,
s_2 = d_1+d_2, ..., s_k = d_1+...+d_k, and ask whether n occurs among the
s_i.

Property tested here (a small, finite, computable instance of exactly that
question):

    For n = 9 and m = 15, divisors of 15 are {1, 3, 5, 15}, giving
    cumulative sums (1, 4, 9, 24). n = 9 occurs at 0-indexed position 2
    (the third cumulative sum). This script first checks classically,
    from first principles, that 15 is indeed the SMALLEST m with this
    property for n = 9 (reproducing a(9) = 15 from A167485 by brute
    force search over m, not by copying the OEIS value), and then
    encodes the resulting fixed-m search -- "which of the 4 divisors'
    cumulative sums of 15 equals 9?" -- as a 2-qubit Grover search
    problem and solves it on Qiskit's ideal AerSimulator.

Search space for the quantum part: index i in {0,1,2,3} (2 qubits)
into the cumulative-sum list of divisors(15) = [1,4,9,24]. Grover's
oracle marks the unique index i with cumsum[i] == 9; the diffuser
amplifies it. With 1 marked item out of 4, a single Grover iteration
gives the marked state with probability 1 on the ideal simulator.

The script prints PASS if the most frequent Grover measurement outcome
matches the classically-determined marked index.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def divisors_sorted(m: int) -> list[int]:
    return sorted(d for d in range(1, m + 1) if m % d == 0)


def cumulative_sums(m: int) -> list[int]:
    ds = divisors_sorted(m)
    out = []
    running = 0
    for d in ds:
        running += d
        out.append(running)
    return out


def smallest_m_for(n: int, m_bound: int) -> int:
    """Brute-force, from first principles, the smallest m <= m_bound such
    that n appears among the cumulative divisor sums of m (A167485(n)),
    or 0 if none found within the bound."""
    for m in range(1, m_bound + 1):
        if n in cumulative_sums(m):
            return m
    return 0


def main() -> None:
    n = 9
    m_bound = 64  # small finite search bound, per the task's N <= ~64

    # --- Classical derivation (first principles, not copied from OEIS) ---
    a_n = smallest_m_for(n, m_bound)
    print(f"Classically derived A167485({n}) = {a_n} (search bound {m_bound})")
    assert a_n == 15, f"expected classical A167485({n}) == 15, got {a_n}"

    m = a_n
    ds = divisors_sorted(m)
    cs = cumulative_sums(m)
    print(f"divisors({m}) = {ds}")
    print(f"cumulative sums = {cs}")

    marked_indices = [i for i, s in enumerate(cs) if s == n]
    assert len(marked_indices) == 1, "expected exactly one matching index for this instance"
    marked_index = marked_indices[0]
    num_qubits = 2
    assert len(ds) == 2 ** num_qubits, "instance must have exactly 4 divisors for a 2-qubit search"
    marked_bits = format(marked_index, f"0{num_qubits}b")  # e.g. "10" for index 2
    print(f"Marked index (classical answer) = {marked_index} -> bitstring |{marked_bits}>")

    # --- Quantum part: 2-qubit Grover search for the marked index ---
    qc = QuantumCircuit(num_qubits, num_qubits)

    # Uniform superposition over all 4 indices.
    qc.h([0, 1])

    def apply_oracle(circuit: QuantumCircuit) -> None:
        # Flip the sign of the marked basis state |marked_bits>.
        # Qiskit bit ordering: qubit 0 is the rightmost character.
        for qubit, bit in enumerate(reversed(marked_bits)):
            if bit == "0":
                circuit.x(qubit)
        circuit.cz(0, 1)
        for qubit, bit in enumerate(reversed(marked_bits)):
            if bit == "0":
                circuit.x(qubit)

    def apply_diffuser(circuit: QuantumCircuit) -> None:
        circuit.h([0, 1])
        circuit.x([0, 1])
        circuit.cz(0, 1)
        circuit.x([0, 1])
        circuit.h([0, 1])

    # One Grover iteration is optimal for N=4, 1 marked item
    # (optimal iterations ~ floor(pi/4 * sqrt(N/M)) = 1).
    apply_oracle(qc)
    apply_diffuser(qc)

    qc.measure([0, 1], [0, 1])

    sim = AerSimulator()
    shots = 2000
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    print(f"Grover measurement counts: {counts}")

    top_outcome = max(counts, key=counts.get)
    top_probability = counts[top_outcome] / shots
    print(f"Most frequent outcome: |{top_outcome}> with probability {top_probability:.3f}")

    quantum_index = int(top_outcome, 2)
    passed = (quantum_index == marked_index) and (top_probability > 0.9)

    if passed:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
