"""
Erdos problem #137 -- quantum-testable sequence lane.

Source metadata (erdosproblems.com data, `data/problems.yaml`, number "137"):
    prize: no
    status: open
    tags: ["number theory", "powerful"]
    oeis: ["N/A"]   -- the erdosproblems.com dataset records NO OEIS id for
                       problem #137 itself. Its `tags` field, however, names
                       the classical concept the problem is about: "powerful"
                       numbers. The canonical OEIS sequence for that concept
                       is A001694 (powerful numbers: n such that every prime
                       dividing n divides it at least twice, i.e. for every
                       prime p | n, p^2 | n). Since the problem entry itself
                       carries no id, this script uses A001694 as the closest
                       real, checkable sequence implied by the entry's tags,
                       and says so plainly rather than inventing a fake id.

Property tested (finite, computable, chosen instance N = 16):
    For each integer n in [1, 16], is n a POWERFUL NUMBER (A001694)?
    n is powerful iff for every prime p dividing n, p^2 also divides n
    (equivalently: n can be written as a^2 * b^3 for positive integers a, b;
    the two definitions are classically known to be equivalent, but this
    script uses the prime-exponent definition directly, computed from first
    principles by trial-division factorization -- no OEIS values are copied).

    Classically, in [1, 16] the powerful numbers are: 1, 4, 8, 9, 16.
    This is computed IN THIS SCRIPT (see `classical_powerful_numbers`),
    not copied from OEIS.

Quantum approach:
    A 4-qubit Grover search over the 16 basis states |0000>..|1111>
    (representing n-1 for n = 1..16, i.e. index i = n-1). The oracle is
    built directly from the classical powerful-number test (computed
    above) by applying a multi-controlled-Z phase flip to exactly the
    marked basis states -- the same construction any Grover oracle uses
    when the marking predicate is evaluated classically to build the
    phase circuit (the standard approach for "mark these known states"
    oracles when no cheaper in-circuit arithmetic oracle is being hand
    rolled). The diffusion operator is the standard Grover diffuser.
    The number of Grover iterations is chosen near the optimal
    floor(pi/4 * sqrt(N/M)) for N=16 basis states and M marked states.

    After running on AerSimulator, the amplified basis states (i.e. the
    most frequently measured outcomes) are decoded back to integers n
    and compared against the classically computed powerful-number set.
    PASS requires that the classical powerful numbers in [1,16] are
    exactly the set of outcomes carrying the top measured probability
    mass (i.e. Grover genuinely amplified the correct marked states).

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N = 16          # search space: integers 1..16
NUM_QUBITS = 4  # 2^4 = 16 basis states, index i encodes n = i + 1


def classical_powerful_numbers(limit: int):
    """Return the sorted list of powerful numbers n in [1, limit].

    n is powerful iff every prime p dividing n satisfies p^2 | n.
    Computed here from first principles via trial-division factorization
    -- nothing is looked up from OEIS.
    """
    powerful = []
    for n in range(1, limit + 1):
        m = n
        is_powerful = True
        p = 2
        while p * p <= m:
            if m % p == 0:
                exponent = 0
                while m % p == 0:
                    m //= p
                    exponent += 1
                if exponent < 2:
                    is_powerful = False
                    break
            p += 1
        if is_powerful and m > 1:
            # m is a leftover prime factor with exponent exactly 1
            is_powerful = False
        if is_powerful:
            powerful.append(n)
    return powerful


def build_oracle(marked_indices, num_qubits):
    """Phase-flip oracle: applies -1 to each basis state in marked_indices.

    Built from the classical predicate (computed above), using X gates to
    remap each marked index onto the all-ones pattern, a multi-controlled-Z
    (via H + multi-controlled-X + H on the target qubit) to flip its phase,
    then undoing the X gates. This is the standard way to realize a Grover
    oracle for an explicit, classically-known marked set.
    """
    qc = QuantumCircuit(num_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{num_qubits}b")
        # flip qubits that should be 0 in this basis state, so that the
        # marked state corresponds to all-ones on the (flipped) register
        zero_positions = [q for q, b in enumerate(reversed(bits)) if b == "0"]
        for q in zero_positions:
            qc.x(q)

        # multi-controlled Z on all qubits (phase flip when all are |1>)
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)

        for q in zero_positions:
            qc.x(q)
    return qc


def build_diffuser(num_qubits):
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def build_grover_circuit(marked_indices, num_qubits, iterations):
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))

    oracle = build_oracle(marked_indices, num_qubits)
    diffuser = build_diffuser(num_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(num_qubits))
        qc.append(diffuser.to_gate(), range(num_qubits))

    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def main():
    classical_answer = classical_powerful_numbers(N)
    marked_indices = sorted(n - 1 for n in classical_answer)
    m = len(marked_indices)

    print(f"Classical powerful numbers in [1, {N}] (A001694, computed here): "
          f"{classical_answer}")

    optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(N / m)))
    print(f"Grover iterations used: {optimal_iterations} "
          f"(N={N} basis states, M={m} marked states)")

    qc = build_grover_circuit(marked_indices, NUM_QUBITS, optimal_iterations)

    backend = AerSimulator()
    transpiled = transpile(qc, backend)
    shots = 8192
    result = backend.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    # decode bitstrings (qiskit returns little-endian classical bit order
    # in the string, i.e. c[num_qubits-1] ... c[0]) into integers n = i+1
    decoded_counts = Counter()
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        n = idx + 1
        decoded_counts[n] += count

    # the M states with the highest measured counts are the circuit's
    # "found" answer
    top_n = [n for n, _ in decoded_counts.most_common(m)]
    quantum_answer = sorted(top_n)

    total_marked_prob = sum(decoded_counts[n] for n in classical_answer) / shots
    print(f"Top-{m} measured outcomes (quantum answer): {quantum_answer}")
    print(f"Total probability mass on classically-correct marked states: "
          f"{total_marked_prob:.4f}")

    passed = (quantum_answer == classical_answer) and (total_marked_prob > 0.9)

    if passed:
        print("PASS")
    else:
        print("FAIL")

    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
