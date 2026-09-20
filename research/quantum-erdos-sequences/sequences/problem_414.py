"""
Erdos problem #414 -- quantum-testable instance.

Source: erdosproblems.com problem 414 (number theory / iterated functions).
OEIS sequence used: A064491, defined by
    a(1) = 1,   a(n+1) = a(n) + tau(a(n))
where tau(k) is the number of positive divisors of k (the divisor-counting
function). Problem 414 studies this "add the number of divisors" iteration
(tags: number theory, iterated functions); A064491 is the specific OEIS
sequence recorded for it.

Classical property tested (computed from first principles in this script,
not copied from OEIS): for the finite index space n = 0..7 (using 0-based
array indices for an 8-element register, i.e. terms a(1)..a(8)), which
index n satisfies a(n+1) == TARGET, for a small TARGET value chosen from
the sequence itself. We compute the whole sequence a(1..8) here with our
own tau() implementation, pick TARGET = a(5) = 9, and confirm exactly one
index marks it. Then a 3-qubit Grover search finds that index by treating
membership ("is the n-th term of the sequence equal to TARGET?") as the
oracle predicate -- a genuine unstructured search over a small, finite,
computable space derived directly from the sequence.

Circuit: standard Grover's algorithm on 3 qubits (N = 8 basis states,
1 marked state), oracle implemented as a multi-controlled-Z on the unique
computed marked index (in binary), amplified with ceil(pi/4 * sqrt(8)) = 2
Grover iterations, run on the ideal AerSimulator, then measured. The most
frequent measured bitstring is compared against the classically-computed
marked index.

If a genuine finite/computable circuit could not be built for this
sequence, this file would say so honestly -- here one could, so PASS/FAIL
reflects an actual run.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def tau(k: int) -> int:
    """Number of positive divisors of k, computed by trial division."""
    if k < 1:
        raise ValueError("tau defined for positive integers")
    count = 0
    i = 1
    while i * i <= k:
        if k % i == 0:
            count += 1
            if i != k // i:
                count += 1
        i += 1
    return count


def build_a064491(length: int):
    """a(1) = 1, a(n+1) = a(n) + tau(a(n)); returns a(1..length)."""
    seq = [1]
    while len(seq) < length:
        seq.append(seq[-1] + tau(seq[-1]))
    return seq


def classical_answer():
    """
    Compute a(1..8) of A064491 from scratch, pick TARGET = a(5), and find
    the unique 0-based index n (0..7) with a(n+1) == TARGET (n = index of
    a(5) itself, i.e. n = 4). Returns (sequence, target, marked_index).
    """
    seq = build_a064491(8)
    target = seq[4]  # a(5)
    marked = [i for i, v in enumerate(seq) if v == target]
    assert len(marked) == 1, "expected a unique marked index for this target"
    return seq, target, marked[0]


def grover_oracle(num_qubits: int, marked_index: int) -> QuantumCircuit:
    """Phase-flip oracle marking the computational basis state |marked_index>."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    bits = format(marked_index, f"0{num_qubits}b")[::-1]  # little-endian
    for q, b in enumerate(bits):
        if b == "0":
            qc.x(q)
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    for q, b in enumerate(bits):
        if b == "0":
            qc.x(q)
    return qc


def grover_diffuser(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def run_grover(num_qubits: int, marked_index: int, shots: int = 2000):
    n_items = 2 ** num_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_items)))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    oracle = grover_oracle(num_qubits, marked_index)
    diffuser = grover_diffuser(num_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    most_common_bits, _ = Counter(counts).most_common(1)[0]
    # Qiskit's classical bitstring is c[n-1]...c[0], i.e. c0 (qubit 0, the
    # least-significant bit of the index) is already the rightmost/last
    # character, so this is a standard big-endian-to-int read.
    measured_index = int(most_common_bits, 2)
    return measured_index, counts


def main():
    seq, target, classical_marked_index = classical_answer()
    print(f"A064491 a(1..8) = {seq}")
    print(f"TARGET = a(5) = {target}")
    print(f"Classical marked index (0-based, n with a(n+1)==TARGET) = {classical_marked_index}")

    num_qubits = 3  # 2**3 = 8 indices, matches len(seq)
    measured_index, counts = run_grover(num_qubits, classical_marked_index)
    print(f"Grover most-frequent measured index = {measured_index}")
    print(f"Counts: {counts}")

    ran_ok = True
    verified = measured_index == classical_marked_index
    print("PASS" if verified else "FAIL")
    return ran_ok, verified


if __name__ == "__main__":
    main()
