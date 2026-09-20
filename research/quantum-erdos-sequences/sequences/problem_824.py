"""
Erdos problem #824 -- quantum-testable lane (best-effort, limitation noted).

Source metadata (from erdosproblems/data/problems.yaml, entry "number: \"824\""):
    prize: "no"
    informal_status: open (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["possible"]
    tags: ["number theory"]

LIMITATION (read before trusting anything below as "problem 824's sequence"):
The yaml's `oeis` field for problem 824 is the literal string "possible", not
an OEIS A-number. There is no actual OEIS sequence id attached to this
problem in the source data, and the problems.yaml file carries no statement
text for #824 beyond the tag "number theory" -- so there is no finite,
checkable sequence-membership property of the *real* problem #824 that this
script can honestly claim to test. Fabricating one and presenting it as
"problem 824's property" would violate the task's own instruction not to
invent unfounded content.

Per the task's fallback instruction ("write the script anyway with your best
honest attempt, note the limitation clearly"), this script instead builds a
REAL, genuinely computed small-instance quantum circuit for a property that
sits squarely in the problem's stated tag ("number theory"): primality.

Chosen finite, computable property:
    For N = 16 (4-bit search space, values 0..15), find the set of primes
    P = { n in [0,16) : n is prime }.
    Classically: P = {2, 3, 5, 7, 11, 13} (computed below by trial division,
    from first principles, not copied from any table).

Quantum method: Grover's algorithm.
    A 4-qubit oracle flips the phase of computational basis states |n> whose
    integer value n is prime (built directly from the classically-computed
    set P above, i.e. the oracle marks exactly the classically-verified
    primality bits -- this is a standard "database search" instantiation of
    Grover, not a black box unrelated to the classical check). We run Grover
    with the optimal number of iterations for |P|=6 marked items out of 16,
    measure, and check that the measured outcomes are concentrated on P with
    high probability, verifying the quantum search actually finds members of
    the classically-defined prime set.

PASS/FAIL: compare the set of most-frequently measured basis states (top
|P| by count) against the classical set P. PASS if they match exactly and
the total measured probability mass on P exceeds 0.8 (which is what
Grover's amplitude amplification with a correctly chosen iteration count is
expected to deliver for this instance).
"""

import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_primes_below(n: int):
    """Trial-division primality, computed from first principles."""
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


def build_oracle(num_qubits: int, marked_values):
    """Phase oracle: flips sign of |v> for each v in marked_values."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for v in marked_values:
        bits = format(v, f"0{num_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(num_qubits: int):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def main():
    N = 16
    num_qubits = 4

    # 1. Classical ground truth, computed here, not copied from a table.
    classical_primes = classical_primes_below(N)
    M = len(classical_primes)
    print(f"Classical primes below {N} (trial division): {classical_primes}")

    # 2. Grover iteration count optimal for M marked items out of N.
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Grover iterations used: {iterations}")

    oracle = build_oracle(num_qubits, classical_primes)
    diffuser = build_diffuser(num_qubits)

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical register bit order is c[n-1]...c[0]; convert keys to ints.
    int_counts = {}
    for bitstring, c in counts.items():
        v = int(bitstring, 2)
        int_counts[v] = int_counts.get(v, 0) + c

    # 3. Compare quantum result to classical answer.
    ranked = sorted(int_counts.items(), key=lambda kv: -kv[1])
    top_values = sorted(v for v, _ in ranked[:M])
    prob_mass_on_primes = sum(c for v, c in int_counts.items() if v in classical_primes) / shots

    print(f"Quantum measurement counts (value: count): {dict(sorted(int_counts.items()))}")
    print(f"Top {M} measured values: {top_values}")
    print(f"Probability mass on classical prime set: {prob_mass_on_primes:.4f}")

    verified = (top_values == sorted(classical_primes)) and (prob_mass_on_primes > 0.8)

    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
