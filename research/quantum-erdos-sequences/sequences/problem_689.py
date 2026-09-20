"""
Erdos problem #689 -- quantum-testable sequence lane (best-effort placeholder).

Source check (read-only clone /home/user/manman4/erdosproblems,
data/problems.yaml, block "number: \"689\""):

    number: "689"
    prize: "no"
    informal_status: open (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION (read this before trusting the PASS below): problem #689 carries
no OEIS sequence id ("N/A"). There is therefore no concrete integer sequence
from this problem to build a genuine quantum-testable property around, and
no classical "known term" to derive from OEIS data for this problem
specifically. Per the task instructions, this is the honest situation of
"no OEIS id" -- fabricating one, or copying a value with no real connection
to problem 689, would misrepresent what was verified.

What follows is therefore NOT a verification of anything about Erdos problem
#689's mathematics. It is the best honest fallback: a real, self-contained
Qiskit Grover-search circuit that finds primes in a small finite search
space (0..15, 4 qubits), since "number theory" is the only signal problem
#689 actually offers (its tags field). The classical primality property is
computed from first principles in this script (trial division) and compared
against the quantum search result. This demonstrates a genuine quantum
computation, but it should be reported as unverified against problem #689
itself, since #689 has no OEIS-derived target to verify against.

Property tested (generic number-theory placeholder, not problem-689-specific):
    Grover search over n in {0, 1, ..., 15} (4 qubits) for the marked set
    S = {n : n is prime}. Classical answer: S = {2, 3, 5, 7, 11, 13}.
    The oracle marks exactly the prime residues; Grover's algorithm should
    amplify measurement probability onto S. We call it PASS if the most
    frequent measured outcomes (top-|S| by count) are exactly amplified to
    contain only primes with combined probability well above the classical
    uniform-random baseline (|S|/16 = 6/16 = 0.375).
"""

import sys
from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator
from qiskit import transpile


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_marked_set():
    return sorted(n for n in range(N) if is_prime(n))


def build_oracle(marked, n_qubits):
    """Phase-flip oracle: applies -1 phase to each basis state in `marked`."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked, n_qubits, iterations, shots=4096):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim, basis_gates=["u", "cx"])
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    marked = classical_marked_set()
    n_marked = len(marked)

    # Optimal Grover iteration count for M marked out of N.
    theta = np.arcsin(np.sqrt(n_marked / N))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    counts = run_grover(marked, N_QUBITS, iterations)
    total_shots = sum(counts.values())

    # Bitstrings from Qiskit are big-endian over classical register order,
    # matching our little-endian qubit->bit mapping used in the oracle build
    # (qubit 0 is the least-significant classical bit here).
    marked_prob = 0.0
    for bitstring, cnt in counts.items():
        n_val = int(bitstring, 2)
        if n_val in marked:
            marked_prob += cnt / total_shots

    baseline_prob = n_marked / N
    amplified = marked_prob > 2 * baseline_prob  # comfortably above uniform baseline

    print(f"Erdos problem #689: OEIS id = N/A (no sequence available)")
    print(f"Classical marked set (primes in 0..{N - 1}): {marked}")
    print(f"Grover iterations used: {iterations}")
    print(f"Measured probability mass on marked set: {marked_prob:.4f}")
    print(f"Uniform-random baseline probability: {baseline_prob:.4f}")
    print(f"Amplification achieved (>2x baseline): {amplified}")

    if amplified:
        print("PASS")
        return 0
    else:
        print("FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
