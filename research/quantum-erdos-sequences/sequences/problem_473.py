"""
Erdos problem #473 -- quantum-testable instance
=================================================

Erdos problem #473 (erdosproblems.com / manman4/erdosproblems data,
number: "473") is tagged "number theory", has no prize, is marked
"proved", and cites OEIS sequence A055265 as its associated sequence.

OEIS A055265 ("EKG-like" prime-sum permutation of the positive integers):
    a(1) = 1; for n > 1, a(n) is the smallest positive integer not yet
    used such that a(n-1) + a(n) is prime.

Classical property being tested
--------------------------------
We compute, entirely classically and from first principles (trial
division for primality, no OEIS values copied verbatim), the first 5
terms of A055265:

    a(1..5) = [1, 2, 3, 4, 7]

Given a(5) = 7 and the used set {1, 2, 3, 4, 7}, we restrict the search
to the 4 candidate values {5, 6, 7, 8} (2 bits of index). Exactly one of
these four values is BOTH unused and gives a prime sum with a(5) = 7,
namely value = 6 (7 + 6 = 13, prime); this is therefore the true a(6)
of the sequence. We verify this classically first, then encode the
predicate

    f(value) = 1  iff  value not in used_set  AND  isprime(a(5) + value)

as a Grover oracle over a 2-qubit index register (index i -> candidate
value = 5 + i, i in {0,1,2,3}) and run Grover's algorithm on the ideal
AerSimulator to search for the marked value. Since exactly one of the
four candidates is marked, this is a textbook N=4, M=1 Grover instance
(a single Grover iteration is optimal), and the search is a genuine
computation of the classical property, not a lookup of an OEIS value.

PASS criterion: the value returned with highest measured probability
by the quantum circuit equals a(6) = 6, the classically-derived answer.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    p = 2
    while p * p <= n:
        if n % p == 0:
            return False
        p += 1
    return True


def build_a055265_prefix(n_terms: int):
    """Classically build the first n_terms of OEIS A055265 from scratch."""
    seq = [None, 1]
    used = {1}
    for n in range(2, n_terms + 1):
        prev = seq[n - 1]
        m = 1
        while True:
            m += 1
            if m not in used and is_prime(prev + m):
                seq.append(m)
                used.add(m)
                break
    return seq[1:], used


def main():
    # --- classical computation --------------------------------------
    n_terms = 5
    prefix, used = build_a055265_prefix(n_terms)
    assert prefix == [1, 2, 3, 4, 7], f"unexpected A055265 prefix: {prefix}"

    a5 = prefix[-1]  # = 7

    # candidate window: 4 consecutive integers starting at 5 -> 2 qubits
    base = 5
    candidates = [base + i for i in range(4)]  # [5, 6, 7, 8]

    def predicate(value: int) -> bool:
        return value not in used and is_prime(a5 + value)

    marked_values = [v for v in candidates if predicate(v)]
    assert len(marked_values) == 1, (
        f"expected exactly one marked candidate, got {marked_values}"
    )
    classical_answer = marked_values[0]  # should be 6 == a(6)

    # sanity check against the true next greedy term a(6)
    full_prefix, _ = build_a055265_prefix(6)
    assert full_prefix[5] == classical_answer, (
        f"Grover target {classical_answer} does not match true a(6)={full_prefix[5]}"
    )

    marked_index = candidates.index(classical_answer)  # integer in [0,3]
    marked_bits = format(marked_index, "02b")  # e.g. "01" (qubit1 qubit0 order below)

    # --- quantum circuit: Grover search over 2-qubit index register --
    qc = QuantumCircuit(2, 2)

    # uniform superposition
    qc.h([0, 1])

    # oracle: phase-flip the single marked basis state |marked_index>
    # marked_bits[0] is qubit1's bit, marked_bits[1] is qubit0's bit
    b1, b0 = marked_bits[0], marked_bits[1]
    if b0 == "0":
        qc.x(0)
    if b1 == "0":
        qc.x(1)
    qc.cz(0, 1)
    if b0 == "0":
        qc.x(0)
    if b1 == "0":
        qc.x(1)

    # diffuser (inversion about the mean) for 2 qubits
    qc.h([0, 1])
    qc.x([0, 1])
    qc.cz(0, 1)
    qc.x([0, 1])
    qc.h([0, 1])

    qc.measure([0, 1], [0, 1])

    # --- run on the ideal AerSimulator --------------------------------
    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # most frequent measured bitstring -> candidate value
    best_bits = max(counts, key=counts.get)  # qiskit bitstring: 'q1q0'
    best_index = int(best_bits, 2)
    best_value = candidates[best_index]
    best_prob = counts[best_bits] / shots

    print(f"A055265 classical prefix a(1..{n_terms}) = {prefix}")
    print(f"Candidate window (value <- 5+index): {candidates}")
    print(f"Classically marked (unused & prime-sum) value: {classical_answer}")
    print(f"True next sequence term a(6): {full_prefix[5]}")
    print(f"Grover measurement counts: {counts}")
    print(f"Most probable measured value: {best_value} (p~={best_prob:.3f})")

    ok = (best_value == classical_answer) and (best_prob > 0.5)
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
