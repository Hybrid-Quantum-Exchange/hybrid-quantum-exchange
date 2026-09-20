"""
Erdos problem #315 (source: manman4/erdosproblems, data/problems.yaml,
entry "number: '315'", tags ["number theory", "unit fractions"],
informal_status "proved").

OEIS ids used: A000058 (Sylvester's sequence: a(0)=2,
a(n+1) = a(n)^2 - a(n) + 1) and A076393 (a related unit-fraction sequence).
This script uses A000058, Sylvester's sequence, which is exactly the object
behind problem #315's "unit fractions" tag (Sylvester's greedy expansion of 1
as a sum of distinct unit fractions: 1/2 + 1/3 + 1/7 + 1/43 + ...).

Classical property being tested
--------------------------------
Sylvester's sequence: a(0)=2, a(1)=3, a(2)=7, a(3)=43, a(4)=1807, ...
via a(n+1) = a(n)*(a(n)-1) + 1.

We pick a small finite instance: among all integers x in [0, 63] (6 bits,
N = 64), find the UNIQUE x such that

    x*(x - 1) + 1 == 43

i.e. the unique x with a(2) = x forcing a(3) = 43 under the Sylvester
recurrence. This is computed from first principles below by direct
classical brute-force search over the same finite space the quantum circuit
searches (not copied from OEIS): we evaluate f(x) = x*(x-1)+1 for every
x in [0,63] and record which x give f(x) == 43. Sylvester's sequence itself
confirms a(2) = 7, so we expect the unique witness to be x = 7 (0b000111).

Quantum circuit
----------------
A standard Grover search over 6 qubits (N = 64) is built. The oracle phase-
flips exactly the computational basis states |x> for which the classically
precomputed set {x : f(x) == 43} contains x (built as a multi-controlled Z
gate on the bit pattern of each such x, so the oracle's marked set is
derived directly from the classical arithmetic check above, not hardcoded
independently). Since the marked set has size 1 out of 64, the optimal
number of Grover iterations is round(pi/4 * sqrt(64/1)) = 6. The circuit is
run on the ideal AerSimulator, and the most frequently measured bitstring is
compared against the classical witness x = 7.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def sylvester_next(x: int) -> int:
    """One step of Sylvester's recurrence: a(n+1) = a(n)*(a(n)-1) + 1."""
    return x * (x - 1) + 1


def classical_sylvester_terms(count: int):
    """First `count` terms of A000058 computed from first principles."""
    terms = [2]
    while len(terms) < count:
        terms.append(sylvester_next(terms[-1]))
    return terms


def classical_search(n_bits: int, target: int):
    """
    Brute-force classical search over x in [0, 2**n_bits - 1] for the set
    of x with x*(x-1)+1 == target. Returns the sorted list of witnesses.
    """
    n = 2 ** n_bits
    witnesses = [x for x in range(n) if sylvester_next(x) == target]
    return witnesses


def build_oracle(n_bits: int, marked_states):
    """
    Multi-controlled-Z oracle that flips the phase of each basis state in
    `marked_states` (list of ints in [0, 2**n_bits - 1]).
    """
    qc = QuantumCircuit(n_bits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_bits}b")[::-1]  # qubit 0 = LSB
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_bits == 1:
            qc.z(0)
        else:
            qc.h(n_bits - 1)
            qc.mcx(list(range(n_bits - 1)), n_bits - 1)
            qc.h(n_bits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_bits: int):
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def run_grover(n_bits: int, marked_states, shots: int = 4096):
    n = 2 ** n_bits
    num_marked = len(marked_states)
    iterations = max(1, round((np.pi / 4) * np.sqrt(n / num_marked)))

    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))

    oracle = build_oracle(n_bits, marked_states)
    diffuser = build_diffuser(n_bits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_bits))
        qc.append(diffuser.to_gate(), range(n_bits))

    qc.measure(range(n_bits), range(n_bits))

    backend = AerSimulator()
    compiled = transpile(qc, backend)
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_bits = 6  # search space N = 64
    target = 43  # a(3) in Sylvester's sequence

    # --- classical, first-principles verification ---
    terms = classical_sylvester_terms(5)
    print(f"Sylvester's sequence (A000058), first 5 terms: {terms}")
    assert terms == [2, 3, 7, 43, 1807], "Sylvester recurrence sanity check failed"

    witnesses = classical_search(n_bits, target)
    print(f"Classical brute-force search over [0,63] for x*(x-1)+1 == {target}: "
          f"witnesses = {witnesses}")
    assert witnesses == [7], "expected a unique witness x = 7 (= a(2))"
    classical_answer = witnesses[0]

    # --- quantum Grover search for the same property ---
    counts, iterations = run_grover(n_bits, witnesses, shots=4096)
    # bitstrings are printed MSB-first by qiskit's classical register ordering
    # matching qubit order q0=LSB..q(n-1)=MSB -> reverse to get int value
    best_bitstring = max(counts, key=counts.get)
    # qiskit prints classical bits MSB-first as c[n-1]...c[0]; qubit i was
    # wired to bit i (LSB-first) of the encoded integer, so this string is
    # already in standard MSB-first order for that same integer.
    quantum_answer = int(best_bitstring, 2)

    total_shots = sum(counts.values())
    hit_fraction = counts[best_bitstring] / total_shots

    print(f"Grover iterations used: {iterations}")
    print(f"Most frequent measured state: {best_bitstring} -> x = {quantum_answer} "
          f"({hit_fraction:.3f} of {total_shots} shots)")

    verified = (quantum_answer == classical_answer) and (hit_fraction > 0.5)

    print(f"Classical answer: x = {classical_answer}")
    print(f"Quantum answer:   x = {quantum_answer}")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
