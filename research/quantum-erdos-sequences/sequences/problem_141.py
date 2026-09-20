"""
Erdos problem #141 (erdosproblems.com/141) -- quantum-testable instance.

OEIS: A006560, "Smallest starting prime for n consecutive primes in
arithmetic progression" (tags on the problem: additive combinatorics,
primes, arithmetic progressions). Known terms: a(1)=2, a(2)=2, a(3)=3,
a(4)=251, a(5)=9843019, a(6)=121174811.

Classical property tested (derived from first principles in this script,
not copied from OEIS): fixing the common difference d = 2 and restricting
the starting prime p to the small finite range 0 <= p < 32 (5 qubits), we
search for the smallest p such that p, p+2, p+4 are all prime. This is
exactly the search problem underlying a(3): the classical brute-force
answer over 0..31 is p = 3 (giving the AP 3, 5, 7), matching a(3) = 3 in
A006560. The circuit below performs a genuine Grover search over the
5-qubit space {0,...,31} for states p satisfying "p, p+2, p+4 all prime",
then classically confirms the most-probable marked state equals the
classically-computed answer (3).

Approach: Grover's algorithm.
  - Classical stage: sieve primes below 40, build the marked set
    M = { p in [0,32) : isprime(p) and isprime(p+2) and isprime(p+4) },
    and compute the classical answer as min(M).
  - Quantum stage: build an oracle that phase-flips exactly the states in
    M (implemented as a diagonal phase oracle over the 5-qubit register,
    constructed directly from the classically-computed marked set -- the
    "checking" logic is classical, matching how Grover oracles are
    normally specified for a black-box predicate), and apply the standard
    Grover diffusion operator, iterating floor(pi/4 * sqrt(N/|M|)) times.
  - Measurement: sample the circuit on AerSimulator, take the most
    frequently observed computational basis state, and check that it lies
    in M and, in particular, equals the classical minimum (3).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMTGate, ZGate
from qiskit_aer import AerSimulator


def is_prime(k: int) -> bool:
    if k < 2:
        return False
    if k < 4:
        return True
    if k % 2 == 0:
        return False
    i = 3
    while i * i <= k:
        if k % i == 0:
            return False
        i += 2
    return True


def classical_marked_set(n_qubits: int, d: int = 2):
    """States p in [0, 2**n_qubits) such that p, p+d, p+2d are all prime."""
    n_states = 2 ** n_qubits
    marked = []
    for p in range(n_states):
        if is_prime(p) and is_prime(p + d) and is_prime(p + 2 * d):
            marked.append(p)
    return marked


def build_oracle(n_qubits: int, marked_states):
    """Diagonal phase oracle flipping the sign of exactly `marked_states`."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian
        # Flip qubits that are 0 in this state so the state becomes |11...1>
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.append(MCMTGate(ZGate(), n_qubits - 1, 1), list(range(n_qubits)))
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.append(MCMTGate(ZGate(), n_qubits - 1, 1), list(range(n_qubits)))
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    n_qubits = 5  # search space size N = 32
    d = 2
    n_states = 2 ** n_qubits

    marked_states = classical_marked_set(n_qubits, d=d)
    assert marked_states, "expected at least one marked state in range"
    classical_answer = min(marked_states)

    # Sanity check against the known OEIS value a(3) = 3 (AP 3,5,7, diff 2).
    expected_a3 = 3
    assert classical_answer == expected_a3, (
        f"classical search over [0,{n_states}) with d={d} gave "
        f"{classical_answer}, expected {expected_a3} (A006560 a(3))"
    )

    n_marked = len(marked_states)
    iterations = max(1, math.floor((math.pi / 4) * math.sqrt(n_states / n_marked)))

    oracle = build_oracle(n_qubits, marked_states)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's count keys list clbit[n-1]...clbit[0] left-to-right, and
    # clbit i was measured from qubit i, so int(bitstring, 2) already
    # recovers the little-endian integer p used throughout this script.
    best_bitstring = max(counts, key=counts.get)
    best_state = int(best_bitstring, 2)

    marked_prob = sum(c for bits, c in counts.items() if int(bits, 2) in marked_states) / shots

    print(f"Classical marked states (p, p+{d}, p+{2*d} all prime) in [0,{n_states}): {marked_states}")
    print(f"Classical answer (min marked state, should match A006560 a(3)): {classical_answer}")
    print(f"Grover iterations used: {iterations}")
    print(f"Most probable measured state: {best_state} (bitstring {best_bitstring})")
    print(f"Total measured probability mass on marked states: {marked_prob:.3f}")

    ok = (best_state in marked_states) and (best_state == classical_answer) and (marked_prob > 0.5)

    if ok:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
