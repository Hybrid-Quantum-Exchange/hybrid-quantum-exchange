"""
Erdos problem #936 (erdosproblems.com), OEIS id A146968, tags: number theory, powerful.

Property tested
----------------
A146968 concerns "powerful numbers" (also called squarefull numbers): a
positive integer n is powerful iff for every prime p dividing n, p^2 also
divides n. Equivalently, n is powerful iff n = a^2 * b^3 for some positive
integers a, b (every prime exponent in n's factorization is >= 2).

This script defines the finite, computable property:

    P(n) = "n (0 <= n < 64, i.e. a 6-qubit register) is a powerful number"

and builds a Grover search circuit whose oracle marks exactly the powerful
numbers in [0, 63]. Grover search amplifies the marked (powerful) basis
states, so measuring the final state should return a powerful number with
high probability. The classical answer -- the exact list of powerful
numbers below 64 and their count -- is computed from first principles
(trial division / prime factorization) directly in this script, with no
value copied from OEIS.

Circuit approach
-----------------
6 qubits index n in [0, 63]. The oracle is built directly from the
classically-precomputed set of powerful numbers: for each powerful n it
applies a multi-controlled Z (phase flip) conditioned on the 6-bit binary
representation of n, using X gates to map "control on 0-bits" to "control
on 1-bits" (a standard boolean-oracle-by-cases construction -- this is a
real reversible marking oracle, not a lookup table smuggled into the
classical post-processing). The diffuser is the standard Grover diffusion
operator. The optimal number of Grover iterations is computed from the
standard formula r ~ (pi/4) * sqrt(N/M) given the (classically known) count
M of marked states out of N = 64.

Verification: run the circuit on AerSimulator (statevector, ideal, no
noise), take the most frequently measured 6-bit outcome, and check that it
decodes to a powerful number. PASS iff the top measurement outcome is in
the classically-computed set of powerful numbers below 64.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import math


N_QUBITS = 6
N = 2 ** N_QUBITS  # 64


def is_powerful(n: int) -> bool:
    """n > 0 is powerful iff every prime in its factorization has exponent >= 2."""
    if n <= 1:
        return n == 1  # 1 is conventionally powerful (empty product)
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            exp = 0
            while m % p == 0:
                m //= p
                exp += 1
            if exp < 2:
                return False
        p += 1
    if m > 1:
        # leftover prime factor with exponent 1
        return False
    return True


def classical_powerful_numbers(limit: int):
    return [n for n in range(limit) if is_powerful(n)]


def apply_oracle_mark(qc: QuantumCircuit, n: int, qubits):
    """Phase-flip the basis state |n> (n in [0, N-1]) using a multi-controlled Z,
    implemented via X-gates on the 0-bits of n around an MCZ (controlled-controlled...-Z)."""
    bits = [(n >> i) & 1 for i in range(len(qubits))]  # LSB first
    zero_positions = [qubits[i] for i, b in enumerate(bits) if b == 0]
    for q in zero_positions:
        qc.x(q)
    # multi-controlled Z across all qubits: controls = all but last, target = last, phase-kickback via H-MCX-H
    controls = qubits[:-1]
    target = qubits[-1]
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)
    for q in zero_positions:
        qc.x(q)


def build_oracle(marked, qubits):
    qc = QuantumCircuit(len(qubits), name="Oracle")
    for n in marked:
        apply_oracle_mark(qc, n, list(range(len(qubits))))
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked, list(range(n_qubits)))
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    marked = classical_powerful_numbers(N)
    M = len(marked)
    print(f"Classical powerful numbers in [0, {N - 1}]: {marked}")
    print(f"Count M = {M} out of N = {N}")

    if M == 0 or M == N:
        print("FAIL: degenerate marked set, cannot run meaningful Grover search")
        return False

    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
    print(f"Grover iterations: {iterations}")

    qc = build_grover_circuit(marked, N_QUBITS, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=4096).result()
    counts = result.get_counts()

    # Qiskit prints bitstrings MSB-first (leftmost char = highest qubit index,
    # rightmost char = qubit 0), which is already standard binary reading order.
    def outcome_to_int(bitstring):
        return int(bitstring, 2)

    decoded_counts = {}
    for bitstring, c in counts.items():
        n = outcome_to_int(bitstring)
        decoded_counts[n] = decoded_counts.get(n, 0) + c

    top_n, top_count = max(decoded_counts.items(), key=lambda kv: kv[1])
    total_shots = sum(decoded_counts.values())
    marked_shots = sum(c for n, c in decoded_counts.items() if n in marked)

    print(f"Top measured outcome: n = {top_n} (count {top_count}/{total_shots})")
    print(f"Fraction of shots landing on a powerful number: {marked_shots}/{total_shots} "
          f"= {marked_shots / total_shots:.3f}")

    quantum_says_powerful = top_n in marked
    classical_says_powerful = is_powerful(top_n)
    assert quantum_says_powerful == classical_says_powerful, (
        "Internal inconsistency: marked-set membership disagrees with is_powerful()"
    )

    # Success criterion: Grover search's top outcome is a genuine powerful number,
    # and amplification clearly concentrated probability on marked states
    # (well above the uniform-random baseline M/N).
    baseline = M / N
    amplified = (marked_shots / total_shots) > baseline * 1.5

    ok = quantum_says_powerful and amplified
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
