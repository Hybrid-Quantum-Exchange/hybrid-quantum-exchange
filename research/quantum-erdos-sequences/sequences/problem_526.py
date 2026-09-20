"""
Erdos problem #526 -- quantum-testable sequence attempt.

LIMITATION (read this first): Erdos problem #526, as recorded in the
manman4/erdosproblems dataset (data/problems.yaml, entry "number: '526'"),
carries no OEIS sequence id -- its "oeis" field is literally ["N/A"]. The
problem's tags are ["probability", "geometry"] and its informal status is
"solved" (as of 2025-08-31), but the dataset gives no integer sequence, no
finite combinatorial object, and no numeric statement attached to problem
526 that could be turned into a small, honest, quantum-computable property.
Fabricating an OEIS id or inventing a "property of the sequence" for a
problem that has neither would not be genuine -- there is no sequence here
to test.

Because the task instructions allow (and require, in this situation) an
honest fallback rather than a faked pass tied to problem 526 itself, this
script instead builds a small, real, self-verifying Grover-search quantum
circuit on a generic, independently-checkable finite arithmetic property
(primality of 3-bit integers, 0..7), and is explicit that this instance is
NOT derived from problem #526's mathematical content -- there is none in
the dataset to derive it from. The circuit itself is real Qiskit code,
genuinely run on AerSimulator, and its result is genuinely compared against
a classical computation performed from first principles in this script.

Classical property tested (unrelated to #526, chosen only as a valid small
Grover instance): for N = 8 (3 qubits, values 0..7), find the unique
element of {0..7} that is BOTH prime AND greater than 5. Classically:
primes in [0,7] = {2,3,5,7}; primes > 5 = {7}. So the marked state is
x = 7 (binary 111), a single unique solution -- ideal for a textbook
Grover search with one Grover iteration on 3 qubits.

Result: PASS/FAIL is reported honestly below; this script does not claim
to verify anything about Erdos problem 526's actual (geometry/probability)
content, since no computable sequence property for #526 exists in source.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_marked_states(n_bits: int):
    """From first principles: values in [0, 2**n_bits) that are prime and > 5."""
    marked = [x for x in range(2 ** n_bits) if is_prime(x) and x > 5]
    return marked


def build_oracle(n_bits: int, marked_value: int) -> QuantumCircuit:
    """Phase-flip oracle marking exactly `marked_value` (a multi-controlled Z)."""
    qc = QuantumCircuit(n_bits, name="oracle")
    bits = format(marked_value, f"0{n_bits}b")[::-1]  # little-endian qubit order
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]
    for i in zero_positions:
        qc.x(i)
    qc.h(n_bits - 1)
    qc.mcx(list(range(n_bits - 1)), n_bits - 1)
    qc.h(n_bits - 1)
    for i in zero_positions:
        qc.x(i)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    qc.h(n_bits - 1)
    qc.mcx(list(range(n_bits - 1)), n_bits - 1)
    qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def main():
    n_bits = 3  # N = 8

    marked = classical_marked_states(n_bits)
    print(f"Classical search space: 0..{2**n_bits - 1}")
    print(f"Classical answer (prime and > 5): {marked}")

    if len(marked) != 1:
        print("FAIL: expected a unique classical solution for a single Grover iteration")
        return False

    target = marked[0]

    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))

    oracle = build_oracle(n_bits, target)
    diffuser = build_diffuser(n_bits)

    # One Grover iteration is optimal for N=8, M=1 (~pi/4 * sqrt(8) ~= 2.2 -> round to 2,
    # but 1 iteration already gives high success probability and keeps the circuit small;
    # we verify empirically below rather than assuming).
    n_iterations = round(np.pi / 4 * np.sqrt(2 ** n_bits / 1)) or 1

    for _ in range(n_iterations):
        qc.append(oracle.to_instruction(), range(n_bits))
        qc.append(diffuser.to_instruction(), range(n_bits))

    qc.measure(range(n_bits), range(n_bits))
    qc = qc.decompose()

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Bitstrings from Qiskit are big-endian classical-register order; qubit 0 is
    # rightmost. Convert to integer directly since register order matches qubit order.
    best_bitstring = max(counts, key=counts.get)
    quantum_answer = int(best_bitstring, 2)
    success_prob = counts.get(best_bitstring, 0) / shots

    print(f"Grover iterations used: {n_iterations}")
    print(f"Measurement counts: {counts}")
    print(f"Most frequent measured value: {quantum_answer} (success prob {success_prob:.3f})")
    print(f"Classical target value: {target}")

    verified = (quantum_answer == target) and (success_prob > 0.7)

    if verified:
        print("PASS")
    else:
        print("FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
