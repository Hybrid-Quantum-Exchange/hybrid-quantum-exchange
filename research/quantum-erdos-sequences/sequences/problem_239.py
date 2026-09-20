"""
Erdos problem #239 -- quantum-testable lane.

Source metadata (data/problems.yaml, entry "number: \"239\"", manman4/erdosproblems,
read 2026-09-19):
    prize: no
    informal_status: proved
    formal_status: Lean
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION (reported honestly, per instructions): problem #239 has no OEIS
sequence id attached (oeis: ["N/A"]). There is therefore no specific integer
sequence from this problem to build a membership/search oracle around, and no
sequence-derived property to verify a quantum result against. This script is
the best honest attempt possible under that constraint: rather than fabricate
a connection to problem #239's actual (unspecified) number-theoretic content,
it builds a genuine, self-contained, verifiable quantum computation on a
small, well-defined finite instance in the same subject area the problem is
tagged with ("number theory") -- primality -- so the harness still gets a
real circuit with a checkable PASS/FAIL, rather than nothing or a faked
result.

Concrete instance: search space is the 4-bit integers 0..15. The classical
target set is {n in [0,15] : n is prime} = {2, 3, 5, 7, 11, 13} (computed
below by trial division, not copied from any table). We build a Grover
search circuit whose oracle marks exactly the primality-satisfying basis
states, run it on the ideal AerSimulator, and check that the state(s) Grover
amplifies to high probability are exactly members of the classically
computed prime set.

verified_against_classical: True in the sense that the oracle's marked set
is independently recomputed classically and the quantum sampling result is
checked against it. It is NOT a verification of any specific term of an
actual OEIS sequence belonging to problem 239, because problem 239 has none.

No OEIS id was used, because none exists for this problem.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_BITS = 4
N = 1 << N_BITS  # 16


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_prime_set(n_bits: int):
    return sorted(n for n in range(1 << n_bits) if is_prime(n))


def build_oracle(qc: QuantumCircuit, qubits, marked_values, n_bits):
    """Phase-flip oracle: applies -1 to each basis state in marked_values."""
    for val in marked_values:
        bits = format(val, f"0{n_bits}b")
        # flip qubits that are 0 in this value so the multi-controlled Z
        # triggers exactly on |val>
        for i, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(qubits[i])
        if n_bits == 1:
            qc.z(qubits[0])
        else:
            qc.h(qubits[-1])
            qc.mcx(qubits[:-1], qubits[-1])
            qc.h(qubits[-1])
        for i, b in enumerate(reversed(bits)):
            if b == "0":
                qc.x(qubits[i])


def build_diffuser(qc: QuantumCircuit, qubits, n_bits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def grover_circuit(marked_values, n_bits, iterations):
    qc = QuantumCircuit(n_bits, n_bits)
    qubits = list(range(n_bits))
    qc.h(qubits)
    for _ in range(iterations):
        build_oracle(qc, qubits, marked_values, n_bits)
        build_diffuser(qc, qubits, n_bits)
    qc.measure(qubits, qubits)
    return qc


def main():
    marked = classical_prime_set(N_BITS)
    print(f"Classical prime set in [0, {N - 1}] (trial division): {marked}")

    m = len(marked)
    # optimal Grover iteration count for M marked out of N states
    iterations = max(1, round((np.pi / 4) * np.sqrt(N / m)))
    print(f"Grover iterations used: {iterations} (N={N}, M={m})")

    qc = grover_circuit(marked, N_BITS, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # decode measured bitstrings (qiskit orders classical bits reversed)
    decoded = {}
    for bitstring, c in counts.items():
        val = int(bitstring, 2)
        decoded[val] = decoded.get(val, 0) + c

    total = sum(decoded.values())
    marked_prob = sum(decoded.get(v, 0) for v in marked) / total
    print(f"Total probability mass landing on classically-prime states: {marked_prob:.4f}")

    top_values = sorted(decoded.items(), key=lambda kv: -kv[1])[:m]
    top_set = sorted(v for v, _ in top_values)
    print(f"Top-{m} most-sampled values from the quantum run: {top_set}")

    # success criteria: Grover should concentrate most probability mass on
    # exactly the marked (prime) states, and the m most frequent measured
    # values should exactly equal the classical prime set.
    ok_probability = marked_prob > 0.80
    ok_top_set = top_set == marked

    passed = ok_probability and ok_top_set

    print(f"ok_probability(>0.80)={ok_probability} ok_top_set_matches_classical={ok_top_set}")
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
