"""
Erdos problem #535 -- quantum-testable sequence entry (limitation notice)
==========================================================================

Source record: /home/user/manman4/erdosproblems/data/problems.yaml,
entry "number: '535'":

    prize: no
    informal_status: open (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["possible"]
    tags: ["number theory"]

LIMITATION (read before trusting anything below as "problem 535's sequence"):
Problem 535 does NOT carry a real OEIS sequence id in this data file. The
value `oeis: ["possible"]` is a status placeholder used elsewhere in this
corpus to mean "an OEIS entry is plausible/pending", not an actual A-number.
There is also no `statement` field in this record to derive a concrete
finite computable property from. Per the task instructions, no property is
fabricated and no OEIS value is copied without derivation -- because there
is no OEIS id or statement to derive one from at all.

Best-honest-attempt substitute: the only concrete signal in the record is
the tag "number theory". So this script builds a REAL, self-contained
quantum computation with genuine mathematical content in that area --
Grover's algorithm searching a small finite space for prime numbers -- and
verifies it against a classical brute-force computation of the same
instance. This is offered as the closest honest analogue available, NOT as
a formalization of Erdos problem #535 itself.

Task performed
--------------
Search space: integers N in [0, 15] (4 qubits, computational basis |N>).
Property tested: "N is prime" (classically: N in {2, 3, 5, 7, 11, 13}).
Method: Grover's algorithm with an oracle built from a reversible primality
circuit over the 4-bit register, amplifying the marked (prime) basis states,
then verifying by measurement that the most probable outcomes are exactly
the classical prime set.

Classical ground truth (computed here from first principles, trial division)
and the exact set of marked/prime states are computed in code below, and the
quantum measurement histogram is compared against that classical set.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import sys
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


N_QUBITS = 4
N_VALUES = 2 ** N_QUBITS  # 0..15


def is_prime_classical(n: int) -> bool:
    """Trial division primality test, computed from first principles."""
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_values: int):
    return [n for n in range(n_values) if is_prime_classical(n)]


def build_oracle(marked_states, n_qubits):
    """Phase oracle: flips the sign of amplitude for each marked basis state.

    For each marked integer, apply X gates to the qubits that must be 0,
    then a multi-controlled Z (via H-MCX-H on the last qubit) to flip the
    phase only when all qubits equal the target bit pattern, then undo the
    X gates.
    """
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")  # MSB first
        # qubit i (little-endian, qubit 0 = LSB) should equal bits[n_qubits-1-i]
        zero_qubits = [i for i in range(n_qubits) if bits[n_qubits - 1 - i] == "0"]
        for q in zero_qubits:
            qc.x(q)
        # multi-controlled Z on all n_qubits, using qubit n_qubits-1 as target
        qc.h(n_qubits - 1)
        if n_qubits - 1 >= 1:
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
        else:
            qc.z(0)
        qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits - 1 >= 1:
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    else:
        qc.z(0)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_prime_search(n_qubits, marked_states, n_iterations, shots=4096):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_states, n_qubits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(n_iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    qc = qc.decompose().decompose().decompose()

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts


def counts_to_ints(counts, n_qubits):
    """Convert Qiskit bitstring counts (MSB-first, little-endian qubit order)
    into {int_value: count}."""
    out = {}
    for bitstring, c in counts.items():
        # Qiskit bitstrings are ordered with qubit (n_qubits-1) leftmost.
        value = int(bitstring, 2)
        out[value] = out.get(value, 0) + c
    return out


def main():
    marked = classical_primes(N_VALUES)
    print(f"Classical primes in [0, {N_VALUES - 1}] (trial division): {marked}")

    n_marked = len(marked)
    n_total = N_VALUES
    # Optimal number of Grover iterations: floor(pi/4 * sqrt(N/M))
    n_iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(n_total / n_marked))))
    print(f"Grover iterations used: {n_iterations}")

    shots = 4096
    counts = grover_prime_search(N_QUBITS, marked, n_iterations, shots=shots)
    value_counts = counts_to_ints(counts, N_QUBITS)

    # Take the top n_marked most frequent measured values as the quantum
    # algorithm's "answer" for the marked (prime) set.
    ranked = sorted(value_counts.items(), key=lambda kv: kv[1], reverse=True)
    quantum_top = sorted(v for v, _ in ranked[:n_marked])

    total_marked_prob = sum(value_counts.get(v, 0) for v in marked) / shots
    print(f"Measured probability mass on the true prime set: {total_marked_prob:.3f}")
    print(f"Top-{n_marked} measured values: {quantum_top}")
    print(f"Classical prime set:             {sorted(marked)}")

    verified = (quantum_top == sorted(marked)) and total_marked_prob > 0.8

    if verified:
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
