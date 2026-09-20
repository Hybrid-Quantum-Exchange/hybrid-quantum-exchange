"""
Erdos problem #974 -- quantum-testable lane.

LIMITATION (read first): Erdos problem #974's entry in
erdosproblems/data/problems.yaml lists oeis: ["N/A"] and tags: ["analysis"].
There is no OEIS sequence attached to this problem, so there is no
"sequence membership / early term" property of *this problem's* sequence
that a small circuit could compute -- there is nothing to compute it from.
Fabricating one would violate the task's own instruction not to invent a
property with no real mathematical content.

Per the task's fallback instruction ("if after reasonable effort no genuine
quantum circuit can be constructed for this problem's sequence ... write the
script anyway with your best honest attempt, note the limitation clearly"),
this script instead runs a genuine, self-contained Grover-search circuit on
a small, real, independently-classically-verified number-theoretic property
-- primality of 4-bit integers (0..15) -- and is honest that this property is
NOT derived from problem 974's own (nonexistent) OEIS sequence. It exists to
prove the quantum-verification machinery genuinely works, not to claim any
connection to problem 974's mathematical content beyond that disclosed
substitution.

Property actually tested: "which integers N in {0, ..., 15} are prime?"
Classical answer (computed here from first principles, trial division):
  primes among 0..15 = {2, 3, 5, 7, 11, 13}   (marked states, 4-qubit basis)

Method: exact Grover search (4 qubits, oracle built from the classically
computed prime set, single-iteration-count chosen from the standard Grover
formula for this marked-state count) run on the ideal AerSimulator. The
script checks that the states with the highest measured probability are
exactly the classically-computed prime set, and prints PASS/FAIL.

Dependencies: qiskit, qiskit_aer, numpy only (already installed).
"""

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_primes(n_max: int):
    """Trial-division primality test, first principles, 0..n_max inclusive."""
    primes = []
    for n in range(n_max + 1):
        if n < 2:
            continue
        is_prime = True
        for d in range(2, int(math.isqrt(n)) + 1):
            if n % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    return primes


def build_oracle(n_qubits: int, marked_states):
    """Phase-flip oracle: multi-controlled Z on each marked basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits: int, marked_states, shots: int = 4096):
    n_items = 2 ** n_qubits
    n_marked = len(marked_states)

    # Standard optimal Grover iteration count.
    theta = math.asin(math.sqrt(n_marked / n_items))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_states)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    qc = qc.decompose().decompose().decompose()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_qubits = 4
    n_max = (2 ** n_qubits) - 1  # 0..15

    marked = classical_primes(n_max)
    print(f"Classical primality (trial division), n in 0..{n_max}: primes = {marked}")

    counts, iterations = run_grover(n_qubits, marked)
    print(f"Grover iterations used: {iterations}")

    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    print("Measurement counts (bitstring -> count), most frequent first:")
    for bitstring, c in sorted_counts:
        value = int(bitstring, 2)
        print(f"  {bitstring} (n={value}): {c}")

    # Take the top len(marked) measured outcomes and compare to the
    # classically computed marked set.
    top_states = {int(bs, 2) for bs, _ in sorted_counts[: len(marked)]}
    classical_set = set(marked)

    verified = top_states == classical_set
    print(f"Top-{len(marked)} measured states: {sorted(top_states)}")
    print(f"Classical prime set:               {sorted(classical_set)}")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
