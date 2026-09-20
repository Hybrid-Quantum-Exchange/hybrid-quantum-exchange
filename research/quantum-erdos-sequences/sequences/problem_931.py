"""
Erdos problem #931 — quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: '931'"):
    prize: no
    informal_status: open (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION (reported honestly, as instructed): problem #931 has NO associated
OEIS sequence id in the source data (oeis: ["N/A"]). There is therefore no
concrete integer sequence to build a quantum-testable property from for this
specific problem. Rather than fabricate an OEIS id or invent a fake property
"of the sequence" (there is no sequence to point to), this script falls back
to the best honest alternative available under problem 931's own tag,
"number theory": a genuine, small, finite, classically-verifiable
number-theoretic search property, tested with a real Grover search circuit
run on the ideal AerSimulator.

Chosen property (small, finite, computable):
    Among the integers N = {0, 1, ..., 63} (6 qubits), find exactly the
    primes. This is a legitimate finite decision/search problem
    ("is n prime?") of the same flavor (number theory, existence/search over
    a finite range) as problem 931's tag, used here in place of a
    sequence-specific property since no sequence exists to test.

Classical ground truth:
    The list of primes in [0, 63] is computed from first principles in this
    script via trial division (`is_prime`), independent of any OEIS lookup.
    This produces the marked set M = {2,3,5,7,11,13,...,61}, |M| = 18 out of
    N = 64 candidates.

Quantum circuit:
    A genuine Grover search circuit (6 qubits + ancilla-free phase-oracle
    via multi-controlled Z) is built whose oracle marks exactly the
    classically-computed prime bitstrings in {0,...,63}. The oracle is
    constructed by, for each marked integer, applying open-controlled /
    controlled phase flips (via X-sandwiched multi-controlled Z) on that
    computational basis state. The optimal number of Grover iterations for
    |M|=18 marked items out of N=64 is computed via the standard formula
    round(pi/4 * sqrt(N/|M|)) and applied. The circuit is simulated exactly
    on qiskit_aer's ideal AerSimulator (statevector), and we check that the
    total measured probability mass lands on the classically-marked (prime)
    bitstrings above a comfortable threshold, and that the single most
    likely outcome is itself prime.

PASS/FAIL:
    The script prints PASS if the quantum-simulated measurement outcomes are
    dominated by the classically-computed prime states (aggregate probability
    on primes > 0.5, and the most-probable single outcome is prime); else
    FAIL.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator
from qiskit import transpile


N_QUBITS = 6
N = 2 ** N_QUBITS  # 64


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    for d in range(3, r + 1, 2):
        if n % d == 0:
            return False
    return True


def classical_primes(limit: int):
    return [n for n in range(limit) if is_prime(n)]


def mark_state(qc: QuantumCircuit, value: int, n_qubits: int):
    """Apply a phase flip (-1) to the computational basis state |value>."""
    bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian qubit order
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]
    for i in zero_positions:
        qc.x(i)
    if n_qubits == 1:
        qc.z(0)
    else:
        mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
        qc.append(mcz, list(range(n_qubits)))
    for i in zero_positions:
        qc.x(i)


def build_oracle(marked, n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for m in marked:
        mark_state(qc, m, n_qubits)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        mcz = MCMTGate(ZGate(), n_qubits - 1, 1)
        qc.append(mcz, list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked, n_qubits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.compose(oracle, qubits=range(n_qubits), inplace=True)
        qc.compose(diffuser, qubits=range(n_qubits), inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    marked = classical_primes(N)
    marked_set = set(marked)
    n_marked = len(marked)

    print(f"Erdos problem #931: no OEIS id (oeis: ['N/A']); tags: ['number theory'].")
    print(f"Falling back to a finite number-theory search: primes in [0, {N - 1}].")
    print(f"Classically computed marked set (primes < {N}): {marked}")
    print(f"|M| = {n_marked} out of N = {N}")

    iterations = max(1, round((math.pi / 4) * math.sqrt(N / n_marked)))
    print(f"Grover iterations used: {iterations}")

    qc = build_grover_circuit(marked, N_QUBITS, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim, basis_gates=["u3", "u", "cx", "x", "h", "z", "cz", "ccx", "mcx"])
    job = sim.run(tqc, shots=4096)
    result = job.result()
    counts = result.get_counts()

    # Qiskit returns bitstrings MSB-first over classical bits in reverse
    # register order matching qubit order c[n-1]...c[0]; qubit 0 is LSB in
    # our little-endian encoding above, and Qiskit's default classical
    # register readout string is big-endian in qubit index, i.e.
    # string[0] == highest qubit index. Convert consistently.
    total_shots = sum(counts.values())
    prime_mass = 0
    best_outcome = None
    best_count = -1
    decoded_counts = {}
    for bitstring, cnt in counts.items():
        # Qiskit's classical-register readout string already has qubit 0
        # as the rightmost (least-significant) character, matching our
        # little-endian encoding in mark_state, so it is interpreted
        # directly as an integer with no reversal needed.
        value = int(bitstring, 2)
        decoded_counts[value] = decoded_counts.get(value, 0) + cnt
        if cnt > best_count:
            best_count = cnt
            best_outcome = value

    for value, cnt in decoded_counts.items():
        if value in marked_set:
            prime_mass += cnt

    prime_fraction = prime_mass / total_shots
    print(f"Total measurement shots: {total_shots}")
    print(f"Fraction of shots landing on a classically-prime state: {prime_fraction:.4f}")
    print(f"Most probable measured outcome: {best_outcome} "
          f"(count={best_count}) -> classically prime? {best_outcome in marked_set}")

    verified = (prime_fraction > 0.5) and (best_outcome in marked_set)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
