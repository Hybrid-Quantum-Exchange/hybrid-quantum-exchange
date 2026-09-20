"""
Erdos problem #225 -- quantum-testable lane.

Source metadata (from data/problems.yaml, manman4/erdosproblems, entry
"number: \"225\""):
    prize: no
    status: proved (2025-08-31)
    oeis: ["N/A"]
    tags: ["analysis"]

LIMITATION (read before trusting the PASS below): problem 225 carries no
OEIS sequence id ("N/A") and is tagged "analysis" with no finite integer
sequence attached to it in the source data. There is therefore no genuine
finite/computable number-theoretic property of "the problem 225 sequence"
to build a quantum circuit around -- there is no such sequence. Fabricating
one (e.g. inventing a fake OEIS id or a property with no connection to the
problem) would violate the instruction to not fake mathematical content, so
this script does NOT claim to verify anything about Erdos problem 225
itself.

What this script actually does, honestly: it is a best-effort placeholder
quantum lane. It implements a real, non-trivial Grover search circuit for a
genuinely computable, verifiable finite property -- "which 4-bit integers
in [0, 15] are prime" -- and checks the quantum result against a classical
brute-force primality check computed from first principles in this script.
This exercises the same kind of finite/small-search-space quantum
methodology the library calls for, but it is NOT a verification of any
Erdos-225-specific classical fact, because no such finite computable fact
exists in the source data for this problem.

Accordingly: ran_ok can be True and the internal (non-225-specific) PASS/
FAIL check can succeed, but this must be reported as NOT a verification
against a classical answer for problem 225 -- there is no such classical
answer to check against.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_primes_below(n: int) -> list[int]:
    """Brute-force primality check, from first principles, for 0..n-1."""
    primes = []
    for k in range(n):
        if k < 2:
            continue
        is_prime = True
        for d in range(2, int(math.isqrt(k)) + 1):
            if k % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(k)
    return primes


def build_prime_oracle(n_qubits: int, marked: list[int]) -> QuantumCircuit:
    """Phase-flip oracle marking each value in `marked` (as an n_qubits
    computational basis state) with a -1 phase, via multi-controlled Z
    preceded/followed by X gates on the bits that are 0 in that value."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked:
        # bits[i] is bit i of `value` (qubit i encodes bit i, matching
        # qiskit's little-endian qubit-to-classical-register convention).
        bits = format(value, f"0{n_qubits}b")[::-1]
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover_prime_search(n_qubits: int, marked: list[int], shots: int = 4096):
    n = 2 ** n_qubits
    m = len(marked)
    iterations = max(1, round((math.pi / 4) * math.sqrt(n / m)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_prime_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_qubits = 4  # search space {0, ..., 15}
    n = 2 ** n_qubits

    classical_answer = classical_primes_below(n)
    print(f"Classical primes in [0, {n - 1}]: {classical_answer}")

    counts, iterations = run_grover_prime_search(n_qubits, classical_answer)
    print(f"Grover iterations used: {iterations}")

    # Interpret measured bitstrings (little-endian per build_prime_oracle)
    # as integers, and take the values that were sampled most often.
    int_counts = Counter()
    for bitstring, freq in counts.items():
        # Qiskit counts keys are "c[n-1]...c[0]", i.e. already in standard
        # big-endian integer order (c[i] measures qubit i, which is the
        # same bit-i-of-value convention used in build_prime_oracle).
        value = int(bitstring, 2)
        int_counts[value] += freq

    top_k = int_counts.most_common(len(classical_answer))
    measured_top_values = sorted(v for v, _ in top_k)

    ok = measured_top_values == sorted(classical_answer)

    print(f"Top {len(classical_answer)} measured values: {measured_top_values}")
    print(f"Classical primes:                 {sorted(classical_answer)}")

    if ok:
        print("PASS")
    else:
        print("FAIL")

    print()
    print("NOTE: This PASS/FAIL is for the internal Grover-primality demo")
    print("only. Erdos problem #225 has no OEIS id and no finite sequence")
    print("in the source data, so this does NOT verify a classical fact")
    print("about problem 225 itself. See the module docstring.")

    return ok


if __name__ == "__main__":
    passed = main()
    raise SystemExit(0 if passed else 1)
