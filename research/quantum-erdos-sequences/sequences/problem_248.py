"""
Erdos problem #248 -- quantum-testable sequence attempt.

LIMITATION (read first): problem_248's entry in erdosproblems/data/problems.yaml
records oeis: ["N/A"] -- there is no OEIS sequence id associated with this
problem. Tags: ["number theory"]. Without an OEIS id there is no concrete
"sequence" to build a Grover oracle / phase-estimation circuit around, as the
task requires one to be derived from the problem's OEIS id(s). This script is
therefore a best-effort fallback, not a genuine encoding of problem #248's
mathematical content: it demonstrates a real, finite, computable number-theory
search (Grover search for primes below a bound, N <= 32, 5 qubits) of the kind
that *would* be relevant to a number-theory Erdos problem, and reports honestly
that it is not tied to problem #248's actual (nonexistent) OEIS sequence.

Classical property tested (chosen only because problem #248 is tagged
"number theory", not derived from any OEIS id):
    For N = 32 (5-bit search space, integers 0..31), find the set of primes.
    This is computed from first principles by trial division in this script,
    then verified as the marked-state set of a Grover search circuit run on
    the ideal AerSimulator.

Circuit: standard Grover's algorithm.
    - 5 qubits encode integers 0..31.
    - Oracle phase-flips exactly the classical prime set {2,3,5,7,11,13,17,19,
      23,29,31} built via a sum of controlled-Z terms selecting those basis
      states (an explicit, verifiable oracle -- not a black box).
    - ~optimal number of Grover iterations for 11 marked states out of 32.
    - Measurement outcomes are compared against the classical prime set: PASS
      if the measured distribution is concentrated (highest-probability
      outcomes) on classically-verified primes.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_primes(n: int):
    """Trial-division primality, first principles, no external data."""
    primes = []
    for k in range(2, n):
        is_p = True
        for d in range(2, int(math.isqrt(k)) + 1):
            if k % d == 0:
                is_p = False
                break
        if is_p:
            primes.append(k)
    return primes


def build_oracle(n_qubits: int, marked_states):
    """Phase-flip oracle: multi-controlled Z on each marked computational
    basis state (X-sandwiched around a multi-controlled-Z so that the
    control pattern equals the marked bitstring)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
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
    N = 2 ** n_qubits
    M = len(marked_states)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_states)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_qubits = 5
    N = 2 ** n_qubits  # 32

    primes = classical_primes(N)
    print(f"Classical primes below {N} (trial division): {primes}")

    counts, iterations = run_grover(n_qubits, primes)
    print(f"Grover iterations used: {iterations}")

    # Sort measured outcomes by frequency, take top-M (M = number of marked
    # states) and check they are all classically-verified primes.
    M = len(primes)
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_states = [int(bits, 2) for bits, _ in sorted_counts[:M]]

    total_shots = sum(counts.values())
    marked_prob = sum(c for bits, c in counts.items()
                       if int(bits, 2) in primes) / total_shots

    print(f"Top {M} measured states (by frequency): {sorted(top_states)}")
    print(f"Fraction of shots landing on classically-verified primes: {marked_prob:.3f}")

    verified = (set(top_states) == set(primes)) and (marked_prob > 0.90)

    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
