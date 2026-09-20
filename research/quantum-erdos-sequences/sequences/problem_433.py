"""
Erdos problem #433 — quantum-testable sequence lane.

Source metadata (from erdosproblems/data/problems.yaml, entry `number: "433"`):
    prize: "no"
    status: "proved (Lean)"
    oeis: ["possible"]
    tags: ["number theory"]

LIMITATION, stated honestly up front: the `oeis` field for problem #433 in the
source data is the literal placeholder string "possible", not a real OEIS
sequence id. There is no concrete OEIS A-number attached to this problem in
the data, so no specific integer sequence could be derived or verified for
it. Per the task instructions ("if no genuine quantum circuit can be
constructed for this problem's sequence ... write the script anyway with
your best honest attempt, note the limitation clearly"), this script falls
back to the only real mathematical content available: the problem's own
`tags: ["number theory"]" field. It builds a genuine Grover-search quantum
circuit over a small, explicit, fully-classically-checked number-theoretic
property (primality), rather than fabricating an OEIS-linked claim.

Chosen finite/computable property (NOT claimed to be OEIS-verified for
problem 433, since no OEIS id exists here):
    Search space: integers N in [0, 15] (4 qubits).
    Property: N is prime.
    Classical answer (computed here from first principles, no external
    libraries): trial division primality test over [0, 15] gives the prime
    set {2, 3, 5, 7, 11, 13}, i.e. 6 marked states out of 16.

Quantum method: exact Grover's algorithm.
    - 4-qubit register, oracle phase-flips the six prime basis states.
    - Optimal number of Grover iterations for M=6 marked out of N=16 is
      round((pi/4) * sqrt(N/M)) = round((pi/4) * sqrt(16/6)) = 2.
    - Run on the ideal AerSimulator, measure, and check that the
      highest-probability measured states are exactly the classically
      computed prime set.

PASS/FAIL: the script prints PASS if the set of most-frequently-measured
basis states (top-6, matching |primes|) equals the classical prime set
{2,3,5,7,11,13}, and each individually has measured probability well above
the uniform baseline (1/16); else FAIL.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


def classical_is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(limit_exclusive: int):
    return sorted(n for n in range(limit_exclusive) if classical_is_prime(n))


NUM_QUBITS = 4
N = 2 ** NUM_QUBITS  # 16
PRIMES = classical_primes(N)  # expected [2, 3, 5, 7, 11, 13]
M = len(PRIMES)


def oracle_circuit() -> QuantumCircuit:
    """Phase-flip every basis state |n> with n in PRIMES."""
    qc = QuantumCircuit(NUM_QUBITS, name="oracle")
    for p in PRIMES:
        # format(p, ...) is big-endian (leftmost char = highest qubit index);
        # reversing it gives bit i == qubit i's value, matching qc.x(i) etc.
        bits = format(p, f"0{NUM_QUBITS}b")[::-1]
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all qubits: phase flip |11...1>
        qc.h(NUM_QUBITS - 1)
        qc.append(MCXGate(NUM_QUBITS - 1), list(range(NUM_QUBITS - 1)) + [NUM_QUBITS - 1])
        qc.h(NUM_QUBITS - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def diffuser_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(NUM_QUBITS, name="diffuser")
    qc.h(range(NUM_QUBITS))
    qc.x(range(NUM_QUBITS))
    qc.h(NUM_QUBITS - 1)
    qc.append(MCXGate(NUM_QUBITS - 1), list(range(NUM_QUBITS - 1)) + [NUM_QUBITS - 1])
    qc.h(NUM_QUBITS - 1)
    qc.x(range(NUM_QUBITS))
    qc.h(range(NUM_QUBITS))
    return qc


def build_grover_circuit(iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
    qc.h(range(NUM_QUBITS))
    oracle = oracle_circuit()
    diffuser = diffuser_circuit()
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(NUM_QUBITS))
        qc.append(diffuser.to_instruction(), range(NUM_QUBITS))
    qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))
    return qc


def main():
    # The textbook formula round((pi/4)*sqrt(N/M)) gives a starting guess;
    # for these small, exactly-known parameters we sweep a small range of
    # iteration counts and pick the one that maximizes the *classically
    # computed* success probability on the ideal statevector, then run that
    # circuit's measurement on the AerSimulator as the actual quantum test.
    from qiskit.quantum_info import Statevector

    def success_probability(it: int) -> float:
        probe = QuantumCircuit(NUM_QUBITS)
        probe.h(range(NUM_QUBITS))
        oracle = oracle_circuit()
        diffuser = diffuser_circuit()
        for _ in range(it):
            probe.append(oracle.to_instruction(), range(NUM_QUBITS))
            probe.append(diffuser.to_instruction(), range(NUM_QUBITS))
        probs = Statevector(probe).probabilities()
        return sum(probs[p] for p in PRIMES)

    candidates = range(1, 6)
    iterations = max(candidates, key=success_probability)
    qc = build_grover_circuit(iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # qiskit count keys: leftmost char = highest qubit index (standard binary read)
    freq = Counter()
    for bitstring, c in counts.items():
        n = int(bitstring, 2)
        freq[n] += c

    ranked = sorted(freq.items(), key=lambda kv: -kv[1])
    top_measured = sorted(n for n, _ in ranked[:M])

    baseline = shots / N
    all_above_baseline = all(freq.get(p, 0) > 2 * baseline for p in PRIMES)

    print(f"Erdos problem #433 (tags: number theory) — quantum lane fallback")
    print(f"OEIS id in source data: 'possible' (not a real A-number; see docstring)")
    print(f"Classical primes in [0,{N-1}]: {PRIMES}")
    print(f"Grover iterations used: {iterations}")
    print(f"Top-{M} measured states: {top_measured}")
    print(f"Counts for primes: {[freq.get(p, 0) for p in PRIMES]} (baseline ~{baseline:.1f})")

    verified = (top_measured == PRIMES) and all_above_baseline
    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
